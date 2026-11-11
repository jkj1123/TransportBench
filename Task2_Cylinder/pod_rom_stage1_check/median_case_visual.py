"""
Pick a median-difficulty test case (by the pure POD-ROM's own rel L2
distribution, same 80 test cases as compare_distributions.py) and produce:
  1. median_case_true_pred_error.png: true / POD-ROM prediction / error,
     side by side, for the rho channel.
  2. median_case_cross_section.png: a vertical cross-section at x = mid-domain
     (the column at W//2), true vs POD-ROM prediction vs y, same case.
Also reports that case's rel L2 under POD-ROM, HyperDeepONet, and
Fusion_DeepONet for direct comparison (same checkpoints as
compare_distributions.py).
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader_fast import CylinderDatasetFast
from model_hyperdeeponet import HyperDeepONet
from model_fusion_deeponet import Fusion_DeepONet

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42); np.random.seed(42)

BACKUP = '/home/sseut1123/run/Transportbench_07_15/TransportBench_10_25/Task2_Cylinder'
CHAN_NAMES = ['rho', 'u', 'v', 'p']
CHANNEL = 0

dataset = CylinderDatasetFast('../cylinder_full_2400_400_seed42.pt', mode='deeponet', device=device)
train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
train_idx, test_idx = train_idx.to(device), test_idx.to(device)

x_branch_train = dataset.branch_all[train_idx]
x_branch_test = dataset.branch_all[test_idx]
x_trunk = dataset.trunk_shared
field_train = dataset.target_all[train_idx]
field_test = dataset.target_all[test_idx]
fluid_mask = dataset.fluid_mask
H, W = dataset.H, dataset.W
n_train, HW, C = field_train.shape
n_test = field_test.shape[0]
fm = fluid_mask.reshape(1, HW, 1).float()
mask2d = fluid_mask.reshape(H, W).cpu().numpy()


def rel2_masked(pred, true):
    p = (pred * fm).reshape(pred.shape[0], -1)
    t = (true * fm).reshape(true.shape[0], -1)
    return (torch.norm(p - t, dim=1) / (torch.norm(t, dim=1) + 1e-8)).cpu().numpy()


# ---------- POD-ROM (r=32) ----------
mean_c = field_train.mean(dim=(0, 1), keepdim=True)
std_c = field_train.std(dim=(0, 1), keepdim=True) + 1e-12
Xtr = ((field_train - mean_c) / std_c).reshape(n_train, -1)
Xm = Xtr.mean(dim=0, keepdim=True)
Xtr_c = Xtr - Xm
U, S, Vt = torch.linalg.svd(Xtr_c, full_matrices=False)
r = 32
V_r = Vt[:r].T.contiguous()
true_coeff_train = Xtr_c @ V_r
rbf = RBFInterpolator(x_branch_train.cpu().numpy(), true_coeff_train.cpu().numpy(),
                       kernel='thin_plate_spline', smoothing=1e-6)
with torch.no_grad():
    coeff = torch.from_numpy(rbf(x_branch_test.cpu().numpy())).float().to(device)
    x_norm_flat = coeff @ V_r.T + Xm
    u_pod_test = x_norm_flat.reshape(n_test, HW, C) * std_c + mean_c
rel2_pod = rel2_masked(u_pod_test, field_test)

# ---------- HyperDeepONet ----------
hyper = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=19, num_outputs=4,
                       trunk_depth=3, branch_depth=3, activation='GELU').to(device)
ckpt = torch.load(os.path.join(BACKUP, 'output/hyperdeeponet_cyl400_param20k_decay100/best_model.pth'),
                   map_location=device, weights_only=False)
hyper.load_state_dict(ckpt if 'model_state_dict' not in ckpt else ckpt['model_state_dict'])
hyper.eval()
with torch.no_grad():
    pred_hyper = hyper(x_branch_test, x_trunk)
rel2_hyper = rel2_masked(pred_hyper, field_test)

# ---------- Fusion_DeepONet ----------
fusion = Fusion_DeepONet(branch_dim=2, trunk_dim=2, hidden_dim=38, num_outputs=4,
                          depth=5, activation='GELU').to(device)
ckpt2 = torch.load(os.path.join(BACKUP, 'output/fusion_deeponet_cyl400_param20k_decay100/best_model.pth'),
                    map_location=device, weights_only=False)
fusion.load_state_dict(ckpt2 if 'model_state_dict' not in ckpt2 else ckpt2['model_state_dict'])
fusion.eval()
with torch.no_grad():
    pred_fusion = fusion(x_branch_test, x_trunk)
rel2_fusion = rel2_masked(pred_fusion, field_test)

# ---------- just pick an arbitrary representative case (no special selection) ----------
median_i = 10
Kn, Ma = x_branch_test[median_i].cpu().numpy()
print(f"Case: local test idx={median_i}  Kn={Kn:.3f}  Ma={Ma:.3f}")
print(f"  rel2  POD-ROM={rel2_pod[median_i]*100:.3f}%  "
      f"HyperDeepONet={rel2_hyper[median_i]*100:.3f}%  Fusion_DeepONet={rel2_fusion[median_i]*100:.3f}%")

true_g = field_test[median_i, :, CHANNEL].reshape(H, W).cpu().numpy()
pred_g = u_pod_test[median_i, :, CHANNEL].reshape(H, W).cpu().numpy()
true_m = np.where(mask2d, true_g, np.nan)
pred_m = np.where(mask2d, pred_g, np.nan)
err_m = pred_m - true_m

# ---------- Image 1: true / pred / error ----------
vmin, vmax = np.nanmin(true_m), np.nanmax(true_m)
emax = np.nanmax(np.abs(err_m))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, data, title, cmap, vlim in [
    (axes[0], true_m, f'{CHAN_NAMES[CHANNEL]} true', 'viridis', (vmin, vmax)),
    (axes[1], pred_m, f'{CHAN_NAMES[CHANNEL]} pred (POD-ROM, r=32)', 'viridis', (vmin, vmax)),
    (axes[2], err_m, f'{CHAN_NAMES[CHANNEL]} error (pred-true)', 'coolwarm', (-emax, emax)),
]:
    im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1], aspect='auto')
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f'Task2_Cylinder POD-ROM, representative test case (idx={median_i}, Kn={Kn:.3f}, Ma={Ma:.3f})\n'
             f'rel2: POD-ROM={rel2_pod[median_i]*100:.2f}%  HyperDeepONet={rel2_hyper[median_i]*100:.2f}%  '
             f'Fusion_DeepONet={rel2_fusion[median_i]*100:.2f}%')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'median_case_true_pred_error.png'), dpi=150)
plt.close()

# ---------- Image 2: cross-section along the y=0 centerline, plotted vs x ----------
trunk2d = x_trunk.reshape(H, W, 2).cpu().numpy()
yrow = int(np.argmin(np.abs(trunk2d[:, W // 2, 1])))  # row index closest to y=0
x_coords = trunk2d[yrow, :, 0]
true_slice = true_g[yrow, :]
pred_slice = pred_g[yrow, :]
mask_slice = mask2d[yrow, :]
true_slice_m = np.where(mask_slice, true_slice, np.nan)
pred_slice_m = np.where(mask_slice, pred_slice, np.nan)

plt.figure(figsize=(7, 5))
plt.plot(x_coords, true_slice_m, 'k-', lw=1.8, label='true')
plt.plot(x_coords, pred_slice_m, 'r--', lw=1.8, label='POD-ROM (r=32)')
plt.xlabel('x'); plt.ylabel(CHAN_NAMES[CHANNEL])
plt.title(f'Task2_Cylinder representative case (idx={median_i}, Kn={Kn:.3f}, Ma={Ma:.3f})\n'
          f'cross-section at y=0 (row {yrow}, centerline)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'median_case_cross_section.png'), dpi=150)
plt.close()

with open(os.path.join(OUT_DIR, 'median_case_summary.txt'), 'w') as f:
    f.write(f"Arbitrary representative test case (idx=10, no special selection)\n")
    f.write(f"local test idx={median_i}  Kn={Kn:.4f}  Ma={Ma:.4f}\n\n")
    f.write(f"rel L2 (fluid-masked, full field):\n")
    f.write(f"  POD-ROM (r=32):      {rel2_pod[median_i]*100:.4f}%\n")
    f.write(f"  HyperDeepONet:       {rel2_hyper[median_i]*100:.4f}%\n")
    f.write(f"  Fusion_DeepONet:     {rel2_fusion[median_i]*100:.4f}%\n")

print(f"Saved to {OUT_DIR}")
