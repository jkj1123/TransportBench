"""
Bridges overfit1 (1 sample, no generalization pressure -- learns real
shock structure, just slowly) and the full pipeline (80 samples -- branch
net collapses to predicting ~0 residual for every geometry, confirmed via
PCA(14)/PCA(4)-reduced branch inputs too, so it isn't an input-dimension
issue). Trains on exactly --n_samples distinct geometries simultaneously
(one batch per epoch, all N samples, same per-sample forward loop as the
full pipeline) and reports PER-SAMPLE rel2 and residual norms, to find the
N at which the branch net stops differentiating between geometries and
starts predicting the same near-zero residual for all of them.

Same mean_field_flat/normalization convention as train_residual_deeponet.py
(computed from the full 80-sample train split, for a realistic, comparable
residual target) and the same hidden_dim/lr/weight_decay recipe as
train_residual_deeponet_overfit1.py (the config that worked best for N=1).
"""
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from model_deeponet import BoltzmannDeepONet
from data_loader_fast import AirfoilDatasetFast


def get_args():
    parser = argparse.ArgumentParser(description="Task I: BoltzmannDeepONet on N simultaneous residual samples")
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='airfoil_unified_128x128.pt')
    parser.add_argument('--run_tag', type=str, default='residual_nsamples')
    parser.add_argument('--hidden_dim', type=int, default=41,
                        help='BoltzmannDeepONet hidden_dim; 41 -> ~50K params for branch_dim=674, depth=5.')
    parser.add_argument('--n_samples', type=int, default=2,
                         help='Number of distinct TRAIN-split geometries to train on simultaneously.')
    parser.add_argument('--lr_decay_step', type=int, default=0)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.91)
    return parser.parse_args()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Starting Task I N-sample test | N={args.n_samples} | Model: BoltzmannDeepONet (RESIDUAL target) | Device: {device}")

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

    residual_train_masked = residual_all[train_idx] * mask_train
    mask_train_bool = mask_train.expand_as(residual_train_masked) > 0
    min_c = torch.stack([residual_train_masked[..., c][mask_train_bool[..., c]].min()
                         for c in range(residual_train_masked.shape[-1])]).view(1, 1, -1)
    max_c = torch.stack([residual_train_masked[..., c][mask_train_bool[..., c]].max()
                         for c in range(residual_train_masked.shape[-1])]).view(1, 1, -1)
    range_c = (max_c - min_c).clamp(min=1e-8)

    residual_norm_all = 2.0 * (residual_all - min_c) / range_c - 1.0

    sample_indices = train_idx[:args.n_samples]
    print(f"Training on {args.n_samples} samples, global indices: {sample_indices.tolist()}")

    x_branch = dataset.branch_all[sample_indices]
    x_trunk_list = [dataset.trunk_all[i] for i in sample_indices]
    y_norm = residual_norm_all[sample_indices]
    y_raw = residual_all[sample_indices]
    mask = dataset.geo_mask_flat[sample_indices]

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
    header = 'epoch,loss,' + ','.join([f'rel2_s{i}' for i in range(args.n_samples)])
    with open(rel2_log_path, 'w') as f:
        f.write(header + '\n')

    print("Training Started...")
    pbar = tqdm(range(args.epochs), desc=f"N={args.n_samples}")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        pred_list = [model(x_branch[i:i+1], x_trunk_list[i]) for i in range(args.n_samples)]
        pred = torch.cat(pred_list, dim=0)
        loss = criterion(pred * mask, y_norm * mask)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred_phys = (pred + 1.0) / 2.0 * range_c + min_c
            pred_masked = (pred_phys * mask).reshape(args.n_samples, -1)
            y_masked = (y_raw * mask).reshape(args.n_samples, -1)
            per_sample_rel2 = (torch.norm(pred_masked - y_masked, p=2, dim=1) /
                               (torch.norm(y_masked, p=2, dim=1) + 1e-8))
            # Cross-sample prediction similarity: are the N predicted residuals
            # collapsing to near-identical (geometry-independent) outputs?
            pred_flat = pred_masked
            pairwise_diff_norm = (pred_flat[0] - pred_flat[-1]).norm().item() if args.n_samples > 1 else float('nan')
            pred_self_norm = pred_flat[0].norm().item()

        row = f'{epoch + 1},{loss.item():.6g},' + ','.join(f'{r.item():.6g}' for r in per_sample_rel2)
        with open(rel2_log_path, 'a') as f:
            f.write(row + '\n')

        pbar.set_postfix({
            'loss': f'{loss.item():.4g}',
            'rel2_mean': f'{per_sample_rel2.mean().item():.4g}',
            'pred0_vs_predN_diff/pred0_norm': f'{(pairwise_diff_norm / (pred_self_norm + 1e-8)):.3g}' if args.n_samples > 1 else 'n/a',
        })

    print(f"Training Complete! Final loss: {loss.item():.4g} | Final per-sample rel2: {[round(r.item(),4) for r in per_sample_rel2]}")
    if args.n_samples > 1:
        print(f"Final cross-sample prediction difference norm (sample0 vs sampleN) / sample0 pred norm: "
              f"{pairwise_diff_norm / (pred_self_norm + 1e-8):.4g}  "
              f"(near 0 means the model predicts ~the same residual for every geometry -- branch collapse)")


if __name__ == '__main__':
    main()
