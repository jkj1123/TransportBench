"""
Eval script paired with train_residual_deeponet_pca14.py: loads a checkpoint
trained with a PCA-reduced geometry branch input, reconstructs the full
predicted field (mean_field_flat + denormalized predicted residual) on the
test split, and reports the masked relative-L2 error plus an optional
prediction-vs-ground-truth visualization for one test sample.
"""
import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_deeponet import BoltzmannDeepONet
from data_loader_fast import AirfoilDatasetFast


def get_args():
    parser = argparse.ArgumentParser(description="Eval: BoltzmannDeepONet on mean-subtracted residual, PCA-reduced geometry branch input")
    parser.add_argument('--run_dir', type=str, required=True,
                         help='Output dir containing best_model.pth, e.g. output/deeponet_residual_50k_pca14')
    parser.add_argument('--data_path', type=str, default='airfoil_unified_128x128.pt')
    parser.add_argument('--sample_idx', type=int, default=0,
                         help='Index into the test split to visualize (0 = first test sample).')
    parser.add_argument('--no_viz', action='store_true', help='Skip saving the prediction visualization image.')
    return parser.parse_args()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_path = os.path.join(args.run_dir, 'best_model.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    dataset = AirfoilDatasetFast(args.data_path, mode='deeponet', device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)
    dataset.fit_train_normalizer(train_idx)

    mean_field_flat = ckpt['mean_field_flat'].to(device)
    min_c = ckpt['min_c'].to(device)
    max_c = ckpt['max_c'].to(device)
    range_c = (max_c - min_c).clamp(min=1e-8)
    hidden_dim = ckpt['hidden_dim']
    pca_dim = ckpt['pca_dim']
    pca_mean = torch.from_numpy(ckpt['pca_mean']).to(device=device, dtype=dataset.branch_all.dtype)
    pca_components = torch.from_numpy(ckpt['pca_components']).to(device=device, dtype=dataset.branch_all.dtype)

    # Reproduce sklearn PCA.transform: (x - mean) @ components.T
    branch_all_pca = (dataset.branch_all - pca_mean) @ pca_components.T

    model = BoltzmannDeepONet(branch_dim=pca_dim, trunk_dim=2, hidden_dim=hidden_dim, num_outputs=4,
                               depth=5, activation='GELU').to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Full test-set masked relative-L2, same metric as training.
    with torch.no_grad():
        xb = branch_all_pca[test_idx]
        xt = dataset.trunk_all[test_idx]
        mask = dataset.geo_mask_flat[test_idx]
        y_raw = dataset.target_all[test_idx] - mean_field_flat.unsqueeze(0)

        pred_list = [model(xb[i:i+1], xt[i]) for i in range(xb.shape[0])]
        pred_norm = torch.cat(pred_list, dim=0)
        pred_phys = (pred_norm + 1.0) / 2.0 * range_c + min_c

        pred_masked = (pred_phys * mask).reshape(xb.shape[0], -1)
        y_masked = (y_raw * mask).reshape(xb.shape[0], -1)
        per_sample_rel2 = torch.norm(pred_masked - y_masked, p=2, dim=1) / \
                           (torch.norm(y_masked, p=2, dim=1) + 1e-8)

    print(f"Checkpoint: {ckpt_path}")
    print(f"PCA dim: {pca_dim} | hidden_dim: {hidden_dim}")
    print(f"Test samples: {xb.shape[0]}")
    print(f"rel2(resid) mean={per_sample_rel2.mean().item():.4f}  "
          f"min={per_sample_rel2.min().item():.4f}  max={per_sample_rel2.max().item():.4f}")

    if args.no_viz:
        return

    # Visualization for one test sample: true/predicted full field + true/predicted residual.
    i = args.sample_idx
    sample_global_idx = test_idx[i].item()
    coords = dataset.trunk_all[sample_global_idx].cpu().numpy().reshape(128, 128, 2)
    mask_np = dataset.geo_mask_flat[sample_global_idx].cpu().numpy().reshape(128, 128)
    true_full = dataset.target_all[sample_global_idx].cpu().numpy().reshape(128, 128, 4)
    true_resid = (dataset.target_all[sample_global_idx] - mean_field_flat).cpu().numpy().reshape(128, 128, 4)
    pred_full = (mean_field_flat.unsqueeze(0) + pred_phys[i:i+1])[0].cpu().numpy().reshape(128, 128, 4)
    pred_resid = pred_phys[i].cpu().numpy().reshape(128, 128, 4)
    X, Y = coords[..., 0], coords[..., 1]

    channels = ['rho', 'u', 'v', 'T']
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    for c, ch in enumerate(channels):
        tf = np.where(mask_np <= 0.5, np.nan, true_full[..., c])
        pf = np.where(mask_np <= 0.5, np.nan, pred_full[..., c])
        tr = np.where(mask_np <= 0.5, np.nan, true_resid[..., c])
        pr = np.where(mask_np <= 0.5, np.nan, pred_resid[..., c])

        vmin_f, vmax_f = np.nanmin(tf), np.nanmax(tf)
        vmax_r = np.nanmax(np.abs(tr))
        vmax_r = vmax_r if vmax_r > 1e-8 else 1.0

        im0 = axes[c, 0].pcolormesh(X, Y, tf, shading='auto', cmap='viridis', vmin=vmin_f, vmax=vmax_f)
        axes[c, 0].set_title(f'{ch}: True field'); plt.colorbar(im0, ax=axes[c, 0])
        im1 = axes[c, 1].pcolormesh(X, Y, pf, shading='auto', cmap='viridis', vmin=vmin_f, vmax=vmax_f)
        axes[c, 1].set_title(f'{ch}: Predicted field'); plt.colorbar(im1, ax=axes[c, 1])
        im2 = axes[c, 2].pcolormesh(X, Y, tr, shading='auto', cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r)
        axes[c, 2].set_title(f'{ch}: True residual'); plt.colorbar(im2, ax=axes[c, 2])
        im3 = axes[c, 3].pcolormesh(X, Y, pr, shading='auto', cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r)
        axes[c, 3].set_title(f'{ch}: Predicted residual'); plt.colorbar(im3, ax=axes[c, 3])
        for ax in axes[c]:
            ax.set_aspect('equal')

    fig.suptitle(f'Task1 DeepONet residual, PCA({pca_dim})-reduced geometry branch -- test sample idx {sample_global_idx}\n'
                 f'mean rel2(resid) over test set = {per_sample_rel2.mean().item():.4f}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(args.run_dir, 'prediction_viz.png')
    plt.savefig(out_path, dpi=130)
    print(f"Saved visualization to {out_path}")


if __name__ == '__main__':
    main()
