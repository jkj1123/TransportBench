"""
Pure POD-ROM (Stage 1 of train_w_pod.py, NO HyperDeepONet residual
correction) check for Gibbs-ringing near the cylinder/shock -- answers
"does Task2_Cylinder's POD-ROM also show the same ringing artifact seen in
Capsule / shocktube / ellipse this session?"

Exactly reuses train_w_pod.py's Stage 1 (train-only normalize -> SVD ->
RBFInterpolator(Kn,Ma)->coeffs), with pod_r=32 (its default), but skips
Stage 2 entirely -- this is the pure linear POD-ROM baseline, closed-form,
no training loop.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader_fast import CylinderDatasetFast

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
POD_R = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42); np.random.seed(42)

dataset = CylinderDatasetFast('../cylinder_full_2400_400_seed42.pt', mode='deeponet', device=device)
train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
train_idx, test_idx = train_idx.to(device), test_idx.to(device)

x_branch_train = dataset.branch_all[train_idx]
x_branch_test = dataset.branch_all[test_idx]
x_trunk = dataset.trunk_shared
field_train = dataset.target_all[train_idx]
field_test = dataset.target_all[test_idx]
fluid_mask = dataset.fluid_mask  # [HW], bool
H, W = dataset.H, dataset.W
n_train, HW, C = field_train.shape
n_test = field_test.shape[0]
print(f"train={n_train} test={n_test} grid={H}x{W} channels={C}  fluid points={fluid_mask.sum().item()}/{HW}")

mean_c = field_train.mean(dim=(0, 1), keepdim=True)
std_c = field_train.std(dim=(0, 1), keepdim=True) + 1e-12

Xtr = ((field_train - mean_c) / std_c).reshape(n_train, -1)
Xm = Xtr.mean(dim=0, keepdim=True)
Xtr_c = Xtr - Xm

U, S, Vt = torch.linalg.svd(Xtr_c, full_matrices=False)
r = POD_R
V_r = Vt[:r].T.contiguous()
cumvar = (S[:r] ** 2).sum() / (S ** 2).sum()
print(f"POD rank r={r}  train cumulative variance explained={cumvar.item()*100:.3f}%")

true_coeff_train = Xtr_c @ V_r
rbf = RBFInterpolator(x_branch_train.cpu().numpy(), true_coeff_train.cpu().numpy(),
                       kernel='thin_plate_spline', smoothing=1e-6)


def u_pod(x_branch):
    with torch.no_grad():
        coeff = torch.from_numpy(rbf(x_branch.cpu().numpy())).float().to(device)
        x_norm_flat = coeff @ V_r.T + Xm
        x_norm = x_norm_flat.reshape(-1, HW, C)
        return x_norm * std_c + mean_c


u_pod_test = u_pod(x_branch_test)

fm = fluid_mask.reshape(1, HW, 1).float()
pred_masked = (u_pod_test * fm).reshape(n_test, -1)
true_masked = (field_test * fm).reshape(n_test, -1)
rel2 = torch.norm(pred_masked - true_masked, dim=1) / (torch.norm(true_masked, dim=1) + 1e-8)
rel2 = rel2.cpu().numpy()
print(f"Pure POD-ROM (no residual correction), fluid-masked test rel L2: "
      f"mean={rel2.mean()*100:.4f}%  median={np.median(rel2)*100:.4f}%  "
      f"min={rel2.min()*100:.4f}%  max={rel2.max()*100:.4f}%")

with open(os.path.join(OUT_DIR, 'metrics.txt'), 'w') as f:
    f.write(f"Task2_Cylinder pure POD-ROM (Stage 1 only, r={POD_R}, no HyperDeepONet residual)\n")
    f.write(f"train={n_train} test={n_test}  cumvar={cumvar.item()*100:.3f}%\n\n")
    f.write(f"Fluid-masked test rel L2: mean={rel2.mean()*100:.4f}%  median={np.median(rel2)*100:.4f}%  "
            f"min={rel2.min()*100:.4f}%  max={rel2.max()*100:.4f}%\n")

params_test = x_branch_test.cpu().numpy()
order = np.argsort(rel2)
show_idx = {'best': order[0], 'median': order[len(order)//2], 'worst': order[-1],
            'highest_Ma': int(np.argmax(params_test[:, 1]))}
CHAN_NAMES = ['rho', 'u', 'v', 'p']
for tag, idx in show_idx.items():
    Kn, Ma = params_test[idx]
    fig, axes = plt.subplots(C, 3, figsize=(12, 2.6 * C))
    m2d = fluid_mask.reshape(H, W).cpu().numpy()
    for c in range(C):
        true_g = field_test[idx, :, c].reshape(H, W).cpu().numpy()
        pred_g = u_pod_test[idx, :, c].reshape(H, W).cpu().numpy()
        true_g_m = np.where(m2d, true_g, np.nan)
        pred_g_m = np.where(m2d, pred_g, np.nan)
        diff_g = pred_g_m - true_g_m
        vmin, vmax = np.nanmin(true_g_m), np.nanmax(true_g_m)
        emax = np.nanmax(np.abs(diff_g))
        for ax, data, title, cmap, vlim in [
            (axes[c, 0], true_g_m, f'{CHAN_NAMES[c]} true', 'viridis', (vmin, vmax)),
            (axes[c, 1], pred_g_m, f'{CHAN_NAMES[c]} pred (pure POD-ROM)', 'viridis', (vmin, vmax)),
            (axes[c, 2], diff_g, f'{CHAN_NAMES[c]} error', 'coolwarm', (-emax, emax)),
        ]:
            im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1], aspect='auto')
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Task2_Cylinder pure POD-ROM, {tag} case idx={idx} (Kn={Kn:.3f}, Ma={Ma:.3f}), rel2={rel2[idx]*100:.3f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'pod_only_{tag}_idx{idx}.png'), dpi=140)
    plt.close()

print(f"Saved to {OUT_DIR}")
