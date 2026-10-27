"""
Visual check: did any of the Task1_Airfoil residual experiments (500K tier
deeponet/hyperdeeponet/mscale_hyperdeeponet/mscaledeeponet, 20K tier
residual_fusion_deeponet and residual_fusion_garbor_deeponet) actually learn
anything, or did they all collapse to predicting ~0 residual everywhere
(rel2(resid)~1.0)? Loads each saved checkpoint, runs it on one fixed test
case, and plots true residual vs predicted residual vs error side by side.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

# 'bwr' has a true pure-white center at 0 (unlike 'coolwarm', whose center is
# a pale grayish tint); NaN (masked-out / solid-airfoil) pixels are also
# forced to white instead of matplotlib's default gray.
CMAP = copy.copy(plt.get_cmap('bwr'))
CMAP.set_bad('white')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader_fast import AirfoilDatasetFast
from model_deeponet import BoltzmannDeepONet
from model_hyperdeeponet import HyperDeepONet
from model_hyper_mscale_deeponet import HyperMscaleDeepONet
from model_mscale_deeponet import MscaleDeepONet
from model_wavelet_deeponet import WaveletDeepONet
from model_fusion_deeponet import Fusion_DeepONet
from model_residual_fusion_deeponet import Residual_Fusion_DeepONet
from model_residual_fusion_garbor_deeponet import Residual_Fusion_Garbor_DeepONet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
H = W = 128
CHANNEL = 0

dataset = AirfoilDatasetFast('airfoil_unified_128x128.pt', mode='deeponet', device=device)
train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
train_idx, test_idx = train_idx.to(device), test_idx.to(device)
dataset.fit_train_normalizer(train_idx)

target_train = dataset.target_all[train_idx]
mask_train = dataset.geo_mask_flat[train_idx]
mask_sum = mask_train.sum(dim=0).clamp(min=1.0)
mean_field_flat = (target_train * mask_train).sum(dim=0) / mask_sum
residual_all = dataset.target_all - mean_field_flat.unsqueeze(0)

x_trunk = dataset.trunk_all[test_idx[0]]  # shared grid, same for every sample
geo_mask = dataset.geo_mask_flat[test_idx[0]].reshape(H, W).cpu().numpy()

MODELS = [
    ('deeponet_residual_500k', lambda hd: BoltzmannDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd, num_outputs=4, depth=5, activation='GELU')),
    ('hyperdeeponet_residual_500k', lambda hd: HyperDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd, num_outputs=4, trunk_depth=3, branch_depth=3, activation='GELU')),
    ('mscale_hyperdeeponet_residual_500k', lambda hd: HyperMscaleDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd, num_outputs=4, depth=4, activation='GELU')),
    ('mscaledeeponet_residual_500k', lambda hd: MscaleDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd, num_outputs=4, scales=[1,2,4,8,16], depth=4, activation='GELU')),
    ('wavelet_deeponet_residual_500k', lambda hd: WaveletDeepONet(branch_dim=674, trunk_dim=2, branch_hidden=hd, trunk_hidden=hd, branch_depth=2, trunk_depth=2, basis_size=hd, num_outputs=4)),
    ('fusion_deeponet_residual_20k', lambda hd: Fusion_DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd, num_outputs=4, depth=5, activation='GELU')),
    ('residual_fusion_deeponet_residual_20k', lambda hd: Residual_Fusion_DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd, num_outputs=4, depth=5, activation='GELU')),
    ('residual_fusion_garbor_deeponet_residual_20k', lambda hd: Residual_Fusion_Garbor_DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd, num_outputs=4, depth=5, activation='GELU')),
    ('residual_fusion_garbor_deeponet_residual_200k', lambda hd: Residual_Fusion_Garbor_DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd, num_outputs=4, depth=5, activation='GELU')),
]

# pick one test case with a reasonably strong true residual (not a trivial near-zero case)
true_resid_test = residual_all[test_idx]
strength = true_resid_test[:, :, CHANNEL].abs().amax(dim=1)
case_local = int(torch.argmax(strength).item())
case_global = test_idx[case_local].item()
print(f"Using test case local_idx={case_local} (global idx={case_global}), "
      f"true residual max|.|={strength[case_local].item():.4f}")

true_resid = residual_all[test_idx[case_local]].reshape(H, W, 4)[:, :, CHANNEL].cpu().numpy()
true_resid_m = np.where(geo_mask == 1, true_resid, np.nan)

x_branch_case = dataset.branch_all[test_idx[case_local]:test_idx[case_local] + 1]

# Zoom to the bounding box of the actual signal (the shock line only occupies
# a thin diagonal sliver of the 128x128 domain -- cropping to it instead of
# showing the whole mostly-blank image makes the differences far easier to see.
sig_mask = np.nan_to_num(np.abs(true_resid_m)) > 0.02 * np.nanmax(np.abs(true_resid_m))
rows_sig, cols_sig = np.where(sig_mask)
pad = 6
r0, r1 = max(rows_sig.min() - pad, 0), min(rows_sig.max() + pad, H)
c0, c1 = max(cols_sig.min() - pad, 0), min(cols_sig.max() + pad, W)


def crop(a):
    return a[r0:r1, c0:c1]


true_resid_m = crop(true_resid_m)
geo_mask_c = crop(geo_mask)

n_models = len(MODELS)
fig, axes = plt.subplots(n_models, 3, figsize=(11, 3.4 * n_models))

# Percentile-based vmax (not the raw max) so one extreme pixel doesn't wash
# out the color scale for the rest of the shock line.
rvmax = np.nanpercentile(np.abs(true_resid_m), 99)

for row, (name, ctor) in enumerate(MODELS):
    ckpt_path = f'output/{name}/best_model.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ctor(ckpt['hidden_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    min_c = ckpt['min_c'].to(device)
    max_c = ckpt['max_c'].to(device)
    range_c = max_c - min_c

    with torch.no_grad():
        pred_norm = model(x_branch_case, x_trunk)  # [1, 16384, 4]
        pred_phys = (pred_norm + 1.0) / 2.0 * range_c + min_c
        pred_resid = pred_phys[0, :, CHANNEL].reshape(H, W).cpu().numpy()
    pred_resid_full = np.where(geo_mask == 1, pred_resid, np.nan)
    pred_resid_m = crop(pred_resid_full)
    err_m = pred_resid_m - true_resid_m

    # rel2 computed on the FULL (uncropped) field, matching training's metric
    pred_resid_full_masked = np.where(geo_mask == 1, pred_resid, np.nan)
    true_resid_full = np.where(geo_mask == 1, true_resid, np.nan)
    rel2_this = float(np.linalg.norm(np.nan_to_num(pred_resid_full_masked - true_resid_full)) /
                       (np.linalg.norm(np.nan_to_num(true_resid_full)) + 1e-8))

    for col, (data, title, cmap, vlim) in enumerate([
        (true_resid_m, 'true residual' if row == 0 else '', CMAP, (-rvmax, rvmax)),
        (pred_resid_m, 'pred residual' if row == 0 else '', CMAP, (-rvmax, rvmax)),
        (err_m, 'error' if row == 0 else '', CMAP, (-rvmax, rvmax)),
    ]):
        ax = axes[row, col]
        im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1], interpolation='nearest')
        if col == 0:
            ax.set_ylabel(name, fontsize=9)
        if row == 0:
            ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    axes[row, 2].set_title(f'error (rel2={rel2_this*100:.2f}%)', fontsize=9)

fig.suptitle(f'Task1_Airfoil residual experiments: true vs predicted residual, zoomed to shock region\n'
             f'(channel {CHANNEL}, test case global_idx={case_global}, color scale = 99th pct of |true|)',
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = 'pod_rom_stage1_check_visual.png'
plt.savefig(out_path, dpi=160)
print(f"Saved {out_path}")
