"""
Evaluation script for the POD-ROM + HyperDeepONet residual hybrid trained
by train_w_pod.py:

    u_pred(Kn, Ma, x) = u_pod(Kn, Ma, x) + unnormalize(HyperDeepONet(Kn, Ma, x))

Loads the single combined checkpoint saved by train_w_pod.py (POD basis,
channel stats, RBF coefficient interpolator, residual normalize_constant,
HyperDeepONet weights), rebuilds the full pipeline, evaluates Rel L2 on
the TEST split (same 0.8/0.2, seed=42 split as every other experiment in
this project), and optionally saves a true/pred/diff comparison plot for
one test sample.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model_hyperdeeponet import HyperDeepONet
from data_loader_fast import CylinderDatasetFast


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate the POD-ROM + HyperDeepONet residual hybrid")
    parser.add_argument('--checkpoint', type=str, default='output/w_pod_hyperdeeponet/best_model.pth')
    parser.add_argument('--data_path', type=str, default='./cylinder_full_2400_400_seed42.pt')
    parser.add_argument('--plot', action='store_true', help='Save a true/pred/diff comparison plot for one test sample.')
    parser.add_argument('--plot_idx', type=int, default=-1, help='Test-set index to plot (-1 = pick the highest-Ma test sample).')
    parser.add_argument('--out_png', type=str, default=None, help='Output path for the plot (default: alongside the checkpoint).')
    return parser.parse_args()


def load_pipeline(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    hyper = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=ckpt['hyper_hidden_dim'],
                           num_outputs=4, trunk_depth=3, branch_depth=3, activation='GELU').to(device)
    hyper.load_state_dict(ckpt['hyper_state_dict'])
    hyper.eval()

    V_r = ckpt['V_r'].to(device)
    Xm = ckpt['Xm'].to(device)
    mean_c = ckpt['mean_c'].to(device)
    std_c = ckpt['std_c'].to(device)
    normalize_constant = ckpt['normalize_constant'].to(device)

    return dict(rbf=ckpt['rbf'], hyper=hyper, V_r=V_r, Xm=Xm, mean_c=mean_c, std_c=std_c,
                normalize_constant=normalize_constant, pod_r=ckpt['pod_r'])


@torch.no_grad()
def predict(pipeline, x_branch, x_trunk, HW, C):
    coeff_np = pipeline['rbf'](x_branch.cpu().numpy())
    coeff = torch.from_numpy(coeff_np).float().to(x_branch.device)
    B = coeff.shape[0]
    x_norm_flat = coeff @ pipeline['V_r'].T + pipeline['Xm']
    u_pod = x_norm_flat.reshape(B, HW, C) * pipeline['std_c'] + pipeline['mean_c']

    r_hyper_norm = pipeline['hyper'](x_branch, x_trunk)
    r_hyper = pipeline['normalize_constant'] * r_hyper_norm

    return u_pod + r_hyper, u_pod, r_hyper


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = CylinderDatasetFast(args.data_path, mode='deeponet', device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    test_idx = test_idx.to(device)

    x_branch_test = dataset.branch_all[test_idx]
    x_trunk = dataset.trunk_shared
    field_test = dataset.target_all[test_idx]
    H, W = dataset.H, dataset.W
    n_test, HW, C = field_test.shape

    pipeline = load_pipeline(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}  (POD rank r={pipeline['pod_r']})")

    u_pred, u_pod_only, r_hyper_only = predict(pipeline, x_branch_test, x_trunk, HW, C)

    pred_flat = u_pred.reshape(n_test, -1)
    true_flat = field_test.reshape(n_test, -1)
    per_sample_rel2 = torch.norm(pred_flat - true_flat, p=2, dim=1) / (torch.norm(true_flat, p=2, dim=1) + 1e-8)
    rel2_combined = per_sample_rel2.mean().item()

    pod_only_flat = u_pod_only.reshape(n_test, -1)
    per_sample_rel2_pod = torch.norm(pod_only_flat - true_flat, p=2, dim=1) / (torch.norm(true_flat, p=2, dim=1) + 1e-8)
    rel2_pod_only = per_sample_rel2_pod.mean().item()

    print(f"Test set (n={n_test}):")
    print(f"  u_pod alone (no residual correction)        Rel L2 = {rel2_pod_only*100:.4f}%")
    print(f"  u_pod + r_hyper (full hybrid)                Rel L2 = {rel2_combined*100:.4f}%")
    print(f"  median Rel L2 (hybrid)                        = {per_sample_rel2.median().item()*100:.4f}%")

    if args.plot:
        params_test = x_branch_test.cpu().numpy()
        if args.plot_idx == -1:
            idx = int(np.argmax(params_test[:, 1]))  # highest Ma
        else:
            idx = args.plot_idx

        true_g = field_test[idx, :, 0].reshape(H, W).cpu().numpy()
        pred_g = u_pred[idx, :, 0].reshape(H, W).cpu().numpy()
        diff_g = pred_g - true_g
        Kn, Ma = params_test[idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(true_g, cmap='viridis', aspect='auto')
        axes[0].set_title(f'True density (test idx={idx}, Kn={Kn:.3f}, Ma={Ma:.3f})')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(pred_g, cmap='viridis', aspect='auto')
        axes[1].set_title(f'Pred density (u_pod + r_hyper), rel2={per_sample_rel2[idx].item()*100:.3f}%')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmax = np.percentile(np.abs(diff_g), 98)
        im2 = axes[2].imshow(diff_g, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[2].set_title('pred - true (p98 clip)')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        out_png = args.out_png or os.path.join(os.path.dirname(args.checkpoint), 'eval_comparison.png')
        plt.savefig(out_png, dpi=150, facecolor='white')
        print(f"Saved comparison plot to {out_png}")


if __name__ == '__main__':
    main()
