"""
Compare the per-test-case rel L2 DISTRIBUTION of pure POD-ROM (Stage 1 of
train_w_pod.py, no neural correction, r=32) against the actual trained
HyperDeepONet and Fusion_DeepONet models, all on the same 80 held-out test
cases (cylinder_full_2400_400_seed42.pt, 0.8/0.2 split, seed=42), same
fluid-masked rel L2 metric. Checkpoints are the existing ~20K-param
cyl400_param20k_decay100 runs (pre-dating fit_train_normalizer, so this script
intentionally does NOT call it either, to match the checkpoints' training-time
data pipeline).
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


def rel2_masked(pred, true):
    p = (pred * fm).reshape(pred.shape[0], -1)
    t = (true * fm).reshape(true.shape[0], -1)
    return (torch.norm(p - t, dim=1) / (torch.norm(t, dim=1) + 1e-8)).cpu().numpy()


# ---------- 1. Pure POD-ROM (r=32), same as check_pod_only_ringing.py ----------
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

# ---------- 2. HyperDeepONet (hidden_dim=19, ~18.8K params) ----------
hyper = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=19, num_outputs=4,
                       trunk_depth=3, branch_depth=3, activation='GELU').to(device)
ckpt = torch.load(os.path.join(BACKUP, 'output/hyperdeeponet_cyl400_param20k_decay100/best_model.pth'),
                   map_location=device, weights_only=False)
hyper.load_state_dict(ckpt if 'model_state_dict' not in ckpt else ckpt['model_state_dict'])
hyper.eval()
with torch.no_grad():
    pred_hyper = hyper(x_branch_test, x_trunk)
rel2_hyper = rel2_masked(pred_hyper, field_test)

# ---------- 3. Fusion_DeepONet (hidden_dim=38, ~19.6K params) ----------
fusion = Fusion_DeepONet(branch_dim=2, trunk_dim=2, hidden_dim=38, num_outputs=4,
                          depth=5, activation='GELU').to(device)
ckpt2 = torch.load(os.path.join(BACKUP, 'output/fusion_deeponet_cyl400_param20k_decay100/best_model.pth'),
                    map_location=device, weights_only=False)
fusion.load_state_dict(ckpt2 if 'model_state_dict' not in ckpt2 else ckpt2['model_state_dict'])
fusion.eval()
with torch.no_grad():
    pred_fusion = fusion(x_branch_test, x_trunk)
rel2_fusion = rel2_masked(pred_fusion, field_test)

# ---------- compare ----------
results = {
    'POD-ROM (r=32, no NN)': rel2_pod,
    'HyperDeepONet (~18.8K)': rel2_hyper,
    'Fusion_DeepONet (~19.6K)': rel2_fusion,
}

print(f"n_test = {n_test}")
for name, arr in results.items():
    print(f"{name}: mean={arr.mean()*100:.3f}%  median={np.median(arr)*100:.3f}%  "
          f"min={arr.min()*100:.3f}%  max={arr.max()*100:.3f}%")

thresholds = [0.005, 0.01, 0.02, 0.05, 0.10]
with open(os.path.join(OUT_DIR, 'distribution_comparison.txt'), 'w') as f:
    f.write("Task2_Cylinder: rel L2 distribution comparison on the same 80 test cases\n")
    f.write("(cyl400, hidden_dim tuned for ~20K params each; POD-ROM r=32)\n\n")
    for name, arr in results.items():
        f.write(f"{name}:\n")
        f.write(f"  mean={arr.mean()*100:.4f}%  median={np.median(arr)*100:.4f}%  "
                f"std={arr.std()*100:.4f}%  min={arr.min()*100:.4f}%  max={arr.max()*100:.4f}%\n")
    f.write("\nFraction of test cases (n=80) exceeding each rel2 threshold:\n")
    f.write(f"{'threshold':>10s}" + "".join(f"{n:>26s}" for n in results.keys()) + "\n")
    for th in thresholds:
        row = f"{th*100:>9.1f}%"
        for name, arr in results.items():
            cnt = int((arr > th).sum())
            row += f"{cnt:>12d} ({cnt/n_test*100:5.1f}%)   "
        f.write(row + "\n")

plt.figure(figsize=(7, 5))
bins = np.logspace(np.log10(1e-4), np.log10(0.2), 30)
for name, arr in results.items():
    plt.hist(np.clip(arr, 1e-4, None), bins=bins, alpha=0.5, label=name)
plt.xscale('log')
plt.xlabel('rel L2 (log scale)'); plt.ylabel('count (of 80 test cases)')
plt.title('Task2_Cylinder: rel L2 distribution\nPOD-ROM vs HyperDeepONet vs Fusion_DeepONet')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'distribution_comparison.png'), dpi=150)

print(f"Saved to {OUT_DIR}")
