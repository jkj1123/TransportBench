"""
1. Task2_Cylinder POD-ROM: test rel L2 vs POD rank r (sweep), same 80 held-out
   test cases as compare_distributions.py -- saved as a plot.
2. Comparison table: POD-ROM (r=32) vs Fusion_DeepONet vs HyperDeepONet vs
   DeepONet, rel L2 on the same 80 test cases -- saved as a table image.
   (DeepONet has no 20K-tier checkpoint available; the existing 200K one is
   used instead and labeled accordingly.)
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
from model_deeponet import BoltzmannDeepONet

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
n_train, HW, C = field_train.shape
n_test = field_test.shape[0]
fm = fluid_mask.reshape(1, HW, 1).float()


def rel2_masked(pred, true):
    p = (pred * fm).reshape(pred.shape[0], -1)
    t = (true * fm).reshape(true.shape[0], -1)
    return (torch.norm(p - t, dim=1) / (torch.norm(t, dim=1) + 1e-8)).cpu().numpy()


# ---------- POD basis (full rank, truncate per r) ----------
mean_c = field_train.mean(dim=(0, 1), keepdim=True)
std_c = field_train.std(dim=(0, 1), keepdim=True) + 1e-12
Xtr = ((field_train - mean_c) / std_c).reshape(n_train, -1)
Xm = Xtr.mean(dim=0, keepdim=True)
Xtr_c = Xtr - Xm
U, S, Vt = torch.linalg.svd(Xtr_c, full_matrices=False)


def pod_rom_rel2(r):
    V_r = Vt[:r].T.contiguous()
    coeff_train = Xtr_c @ V_r
    rbf = RBFInterpolator(x_branch_train.cpu().numpy(), coeff_train.cpu().numpy(),
                           kernel='thin_plate_spline', smoothing=1e-6)
    with torch.no_grad():
        coeff = torch.from_numpy(rbf(x_branch_test.cpu().numpy())).float().to(device)
        u_pod = (coeff @ V_r.T + Xm).reshape(n_test, HW, C) * std_c + mean_c
    return rel2_masked(u_pod, field_test)


# ---------- 1. rank sweep ----------
candidate_rs = [2, 5, 8, 12, 16, 20, 24, 28, 32, 40, 50, 64, 80, 100, 150, 200, 320]
candidate_rs = [r for r in candidate_rs if r < n_train]
sweep_mean, sweep_median = [], []
for r in candidate_rs:
    rel2_r = pod_rom_rel2(r)
    sweep_mean.append(rel2_r.mean() * 100)
    sweep_median.append(np.median(rel2_r) * 100)
    print(f"r={r:3d}: mean={sweep_mean[-1]:.4f}%  median={sweep_median[-1]:.4f}%")

plt.figure(figsize=(7, 5))
plt.plot(candidate_rs, sweep_mean, 'o-', label='mean test rel L2')
plt.plot(candidate_rs, sweep_median, 's--', label='median test rel L2')
plt.axvline(32, color='gray', ls=':', label='r=32 (used elsewhere)')
plt.xlabel('POD rank r'); plt.ylabel('test rel L2 (%)')
plt.yscale('log')
plt.title('Task2_Cylinder POD-ROM: test rel L2 vs POD rank\n(same 80 held-out test cases)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pod_rank_sweep.png'), dpi=150)
plt.close()
print(f"Saved rank sweep plot.")

# ---------- 2. comparison table ----------
rel2_pod32 = pod_rom_rel2(32)

hyper = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=19, num_outputs=4,
                       trunk_depth=3, branch_depth=3, activation='GELU').to(device)
ckpt = torch.load(os.path.join(BACKUP, 'output/hyperdeeponet_cyl400_param20k_decay100/best_model.pth'),
                   map_location=device, weights_only=False)
hyper.load_state_dict(ckpt if 'model_state_dict' not in ckpt else ckpt['model_state_dict'])
hyper.eval()
with torch.no_grad():
    pred_hyper = hyper(x_branch_test, x_trunk)
rel2_hyper = rel2_masked(pred_hyper, field_test)

fusion = Fusion_DeepONet(branch_dim=2, trunk_dim=2, hidden_dim=38, num_outputs=4,
                          depth=5, activation='GELU').to(device)
ckpt2 = torch.load(os.path.join(BACKUP, 'output/fusion_deeponet_cyl400_param20k_decay100/best_model.pth'),
                    map_location=device, weights_only=False)
fusion.load_state_dict(ckpt2 if 'model_state_dict' not in ckpt2 else ckpt2['model_state_dict'])
fusion.eval()
with torch.no_grad():
    pred_fusion = fusion(x_branch_test, x_trunk)
rel2_fusion = rel2_masked(pred_fusion, field_test)

deeponet = BoltzmannDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=123, num_outputs=4,
                              depth=5, activation='GELU').to(device)
ckpt3 = torch.load(os.path.join(BACKUP, 'output/deeponet_cyl400_param200k_decay100/best_model.pth'),
                    map_location=device, weights_only=False)
deeponet.load_state_dict(ckpt3 if 'model_state_dict' not in ckpt3 else ckpt3['model_state_dict'])
deeponet.eval()
with torch.no_grad():
    pred_deeponet = deeponet(x_branch_test, x_trunk)
rel2_deeponet = rel2_masked(pred_deeponet, field_test)

hyper200k = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=44, num_outputs=4,
                           trunk_depth=3, branch_depth=3, activation='GELU').to(device)
ckpt4 = torch.load(os.path.join(BACKUP, 'output/hyperdeeponet_cyl400_param200k_decay100/best_model.pth'),
                    map_location=device, weights_only=False)
hyper200k.load_state_dict(ckpt4 if 'model_state_dict' not in ckpt4 else ckpt4['model_state_dict'])
hyper200k.eval()
with torch.no_grad():
    pred_hyper200k = hyper200k(x_branch_test, x_trunk)
rel2_hyper200k = rel2_masked(pred_hyper200k, field_test)

rows = [
    ("POD-ROM (r=32)", "0", rel2_pod32),
    ("Fusion_DeepONet", "19,554", rel2_fusion),
    ("HyperDeepONet", "18,757", rel2_hyper),
    ("HyperDeepONet", "196,332", rel2_hyper200k),
    ("DeepONet", "199,014", rel2_deeponet),
]

print("\nComparison (n=80 test cases, fluid-masked rel L2):")
for name, params, arr in rows:
    print(f"  {name:>16s} ({params:>9s}): mean={arr.mean()*100:.3f}%  median={np.median(arr)*100:.3f}%  "
          f"max={arr.max()*100:.3f}%")

fig, ax = plt.subplots(figsize=(8, 2.2))
ax.axis('off')
col_labels = ["Model", "Params", "Mean rel L2", "Median rel L2", "Max rel L2"]
cell_text = [[name, params, f"{arr.mean()*100:.3f}%", f"{np.median(arr)*100:.3f}%", f"{arr.max()*100:.3f}%"]
             for name, params, arr in rows]
table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
ax.set_title("Task2_Cylinder: rel L2 comparison (same 80 test cases)\n"
             "DeepONet has no 20K checkpoint; the existing ~200K run is shown instead",
             fontsize=10, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_comparison_table.png'), dpi=150, bbox_inches='tight')
plt.close()

with open(os.path.join(OUT_DIR, 'model_comparison_table.txt'), 'w') as f:
    f.write("Task2_Cylinder: rel L2 comparison (same 80 held-out test cases)\n\n")
    f.write(f"{'Model':>18s} {'Params':>10s} {'Mean':>10s} {'Median':>10s} {'Max':>10s}\n")
    for name, params, arr in rows:
        f.write(f"{name:>18s} {params:>10s} {arr.mean()*100:9.3f}% {np.median(arr)*100:9.3f}% {arr.max()*100:9.3f}%\n")
    f.write("\nNote: DeepONet has no 20K-param checkpoint available; the existing ~199K-param "
            "run is shown instead. The two HyperDeepONet rows (18,757 vs 196,332 params) show "
            "that a 10x larger HyperDeepONet improves but still does not beat POD-ROM or the "
            "20K Fusion_DeepONet.\n")

print(f"Saved table image + text to {OUT_DIR}")
