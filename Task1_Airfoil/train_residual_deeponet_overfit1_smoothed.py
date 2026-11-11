"""
Combines the two single-variable ablations tried separately on Task1:
train_residual_deeponet_overfit1.py (single-sample overfit, no
generalization pressure) and train_residual_deeponet_smoothed.py (Gaussian
smoothing of the residual target, to remove the pixel-level DSMC noise seen
in overfit1's True-residual panel). Isolates whether removing noise helps
once gradient interference from other samples is also removed, directly
testing the "gradient interference across the 80-sample batch" hypothesis
raised after overfit1's visualization.

Same optimizer/scheduler/weight_decay/hidden_dim recipe as
train_residual_deeponet.py; the only deliberate changes are: (1) the
dataset size used in the training loop (1 sample instead of 80), and (2)
Gaussian-smoothing the residual target before normalization.
"""
import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from model_deeponet import BoltzmannDeepONet
from data_loader_fast import AirfoilDatasetFast


def get_args():
    parser = argparse.ArgumentParser(description="Task I: BoltzmannDeepONet overfit sanity check on ONE residual sample")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='airfoil_unified_128x128.pt')
    parser.add_argument('--run_tag', type=str, default='residual_overfit1')
    parser.add_argument('--hidden_dim', type=int, default=41,
                        help='BoltzmannDeepONet hidden_dim; 41 -> ~50K params for branch_dim=674, depth=5.')
    parser.add_argument('--sample_idx', type=int, default=-1,
                         help='Global sample index to overfit on. -1 (default) uses the first TRAIN-split sample.')
    parser.add_argument('--lr_decay_step', type=int, default=0,
                        help='0 (default) = no decay, matching the spirit of a pure overfit test.')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.91)
    parser.add_argument('--smooth_sigma', type=float, default=1.5,
                        help='Gaussian filter sigma (in grid cells) applied to the residual target before normalization.')
    return parser.parse_args()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Starting Task I overfit-sanity-check | Model: BoltzmannDeepONet (RESIDUAL target, 1 sample) | Device: {device}")

    save_dir = os.path.join('output', f"deeponet_{args.run_tag}")
    os.makedirs(save_dir, exist_ok=True)

    dataset = AirfoilDatasetFast(args.data_path, mode='deeponet', device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)
    dataset.fit_train_normalizer(train_idx)

    target_train = dataset.target_all[train_idx]
    mask_train = dataset.geo_mask_flat[train_idx]
    mask_sum = mask_train.sum(dim=0).clamp(min=1.0)
    mean_field_flat = (target_train * mask_train).sum(dim=0) / mask_sum

    residual_all = dataset.target_all - mean_field_flat.unsqueeze(0)

    n_samples = residual_all.shape[0]
    residual_np = residual_all.reshape(n_samples, 128, 128, 4).cpu().numpy()
    residual_smoothed_np = np.empty_like(residual_np)
    for i in range(n_samples):
        for c in range(4):
            residual_smoothed_np[i, :, :, c] = gaussian_filter(
                residual_np[i, :, :, c], sigma=args.smooth_sigma, mode='nearest')
    residual_all = torch.from_numpy(residual_smoothed_np).reshape(n_samples, -1, 4).to(residual_all.device, residual_all.dtype)
    print(f"Applied Gaussian smoothing (sigma={args.smooth_sigma}) to the residual target.")

    residual_train_masked = residual_all[train_idx] * mask_train
    mask_train_bool = mask_train.expand_as(residual_train_masked) > 0
    min_c = torch.stack([residual_train_masked[..., c][mask_train_bool[..., c]].min()
                         for c in range(residual_train_masked.shape[-1])]).view(1, 1, -1)
    max_c = torch.stack([residual_train_masked[..., c][mask_train_bool[..., c]].max()
                         for c in range(residual_train_masked.shape[-1])]).view(1, 1, -1)
    range_c = (max_c - min_c).clamp(min=1e-8)

    residual_norm_all = 2.0 * (residual_all - min_c) / range_c - 1.0

    sample_idx = args.sample_idx if args.sample_idx >= 0 else train_idx[0].item()
    print(f"Overfitting on single sample, global index {sample_idx}")

    x_branch = dataset.branch_all[sample_idx:sample_idx + 1]
    x_trunk = dataset.trunk_all[sample_idx]
    y_norm = residual_norm_all[sample_idx:sample_idx + 1]
    y_raw = residual_all[sample_idx:sample_idx + 1]
    mask = dataset.geo_mask_flat[sample_idx:sample_idx + 1]

    model = BoltzmannDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=args.hidden_dim, num_outputs=4,
                               depth=5, activation='GELU').to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {n_params} ({n_params/1e3:.1f}K)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_decay_step > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    criterion = nn.MSELoss()

    rel2_log_path = os.path.join(save_dir, 'rel2_history.csv')
    with open(rel2_log_path, 'w') as f:
        f.write('epoch,loss,rel2\n')

    print("Training Started...")
    pbar = tqdm(range(args.epochs), desc="Overfit-1")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        pred = model(x_branch, x_trunk)
        loss = criterion(pred * mask, y_norm * mask)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred_phys = (pred + 1.0) / 2.0 * range_c + min_c
            pred_masked = (pred_phys * mask).reshape(1, -1)
            y_masked = (y_raw * mask).reshape(1, -1)
            rel2 = (torch.norm(pred_masked - y_masked, p=2, dim=1) /
                    (torch.norm(y_masked, p=2, dim=1) + 1e-8)).item()

        with open(rel2_log_path, 'a') as f:
            f.write(f'{epoch + 1},{loss.item():.6g},{rel2:.6g}\n')

        pbar.set_postfix({'loss': f'{loss.item():.4g}', 'rel2': f'{rel2:.4g}'})

    print(f"Training Complete! Final loss: {loss.item():.4g} | Final rel2(resid) on the overfit sample: {rel2:.4g}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'mean_field_flat': mean_field_flat.cpu(),
        'min_c': min_c.cpu(),
        'max_c': max_c.cpu(),
        'hidden_dim': args.hidden_dim,
        'sample_idx': sample_idx,
    }, os.path.join(save_dir, "final_model.pth"))


if __name__ == '__main__':
    main()
