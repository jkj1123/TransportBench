"""
Train a ~50K-parameter MscaleDeepONet directly on the MEAN-SUBTRACTED RESIDUAL
field for Task1_Airfoil, under the exact same experimental conditions as
train_residual_hyperdeeponet.py (same residual target, same per-channel
normalize_constant, same data split, same optimizer/scheduler/epoch budget),
to check whether a multi-scale trunk (explicitly designed to counter spectral
bias) fares any better than HyperDeepONet on this residual-only regression.
"""
import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_mscale_deeponet import MscaleDeepONet
from data_loader_fast import AirfoilDatasetFast, FastBatchIterator


def get_args():
    parser = argparse.ArgumentParser(description="Task I: MscaleDeepONet trained on mean-subtracted residual")
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_path', type=str, default='airfoil_unified_128x128.pt')
    parser.add_argument('--run_tag', type=str, default='residual_50k')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='MscaleDeepONet hidden_dim; 32 -> 50,469 params (~50K budget) for branch_dim=674.')
    parser.add_argument('--lr_decay_step', type=int, default=100,
                        help='Matches Task2_Cylinder/train_w_pod.py (latest config): decay every 100 epochs.')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.91,
                        help='Matches Task2_Cylinder/train_w_pod.py (latest config).')
    return parser.parse_args()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Starting Task I Training | Model: MscaleDeepONet (RESIDUAL target) | Device: {device}")
    if device == 'cuda':
        print(f"  torch.cuda.current_device()={torch.cuda.current_device()}  "
              f"name={torch.cuda.get_device_name(0)}")

    save_dir = os.path.join('output', f"mscaledeeponet_{args.run_tag}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    dataset = AirfoilDatasetFast(args.data_path, mode='deeponet', device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)

    target_train = dataset.target_all[train_idx]
    mask_train = dataset.geo_mask_flat[train_idx]
    mask_sum = mask_train.sum(dim=0).clamp(min=1.0)
    mean_field_flat = (target_train * mask_train).sum(dim=0) / mask_sum

    residual_all = dataset.target_all - mean_field_flat.unsqueeze(0)

    residual_train_masked = residual_all[train_idx] * mask_train
    std_c = residual_train_masked.std(dim=(0, 1), keepdim=True) + 1e-8
    print(f"residual per-channel std (train, normalize_constant): {std_c.flatten().tolist()}")

    residual_norm_all = residual_all / std_c

    train_tensors = (dataset.branch_all[train_idx], dataset.trunk_all[train_idx],
                      residual_norm_all[train_idx], dataset.geo_mask_flat[train_idx])
    test_tensors = (dataset.branch_all[test_idx], dataset.trunk_all[test_idx],
                     residual_norm_all[test_idx], dataset.geo_mask_flat[test_idx],
                     residual_all[test_idx])

    train_loader = FastBatchIterator(train_tensors, batch_size=args.batch_size, shuffle=True, device=device)
    test_loader = FastBatchIterator(test_tensors, batch_size=args.batch_size, shuffle=False, device=device)

    model = MscaleDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=args.hidden_dim, num_outputs=4,
                            scales=[1, 2, 4, 8, 16], depth=4, activation='GELU').to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {n_params} ({n_params/1e3:.1f}K)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    if args.lr_decay_step > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    best_test_rel2 = float('inf')
    history = {'train_loss': [], 'test_loss': []}
    rel2_log_path = os.path.join(save_dir, 'rel2_history.csv')
    with open(rel2_log_path, 'w') as f:
        f.write('epoch,wall_time_sec,test_rel2,best_test_rel2\n')
    train_start_time = time.time()

    print("Training Started...")
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        rel2_sum, rel2_count = 0.0, 0
        model.train()
        train_loss_acc = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            x_branch, x_trunk, y, mask = batch
            batch_size = x_branch.shape[0]
            pred_list = [model(x_branch[i:i+1], x_trunk[i]) for i in range(batch_size)]
            pred = torch.cat(pred_list, dim=0)
            loss = criterion(pred * mask, y * mask)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()

        if scheduler is not None:
            scheduler.step()
        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x_branch, x_trunk, y, mask, y_raw = batch
                batch_size = x_branch.shape[0]
                pred_list = [model(x_branch[i:i+1], x_trunk[i]) for i in range(batch_size)]
                pred = torch.cat(pred_list, dim=0)
                loss = criterion(pred * mask, y * mask)
                test_loss_acc += loss.item()

                pred_phys = pred * std_c
                pred_masked = (pred_phys * mask).reshape(pred.shape[0], -1)
                y_masked = (y_raw * mask).reshape(y_raw.shape[0], -1)
                per_sample_rel2 = torch.norm(pred_masked - y_masked, p=2, dim=1) / \
                                   (torch.norm(y_masked, p=2, dim=1) + 1e-8)
                rel2_sum += per_sample_rel2.sum().item()
                rel2_count += pred_masked.shape[0]

        avg_test_loss = test_loss_acc / len(test_loader)
        history['test_loss'].append(avg_test_loss)
        test_rel2 = rel2_sum / rel2_count
        if test_rel2 < best_test_rel2:
            best_test_rel2 = test_rel2
        wall_time_sec = time.time() - train_start_time
        with open(rel2_log_path, 'a') as f:
            f.write(f'{epoch + 1},{wall_time_sec:.2f},{test_rel2:.6g},{best_test_rel2:.6g}\n')

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean_field_flat': mean_field_flat.cpu(),
                'std_c': std_c.cpu(),
                'hidden_dim': args.hidden_dim,
            }, save_path)

        pbar.set_postfix({
            'train': f'{avg_train_loss:.4g}',
            'test': f'{avg_test_loss:.4g}',
            'best': f'{best_test_loss:.4g}',
            'rel2(resid)': f'{test_rel2:.4g}',
            'best_rel2': f'{best_test_rel2:.4g}',
        })

    total_wall_time = time.time() - train_start_time
    print(f"Training Complete! Best Test Loss: {best_test_loss:.4g} | Best Test Rel L2 (residual) : {best_test_rel2:.4g} | "
          f"Wall Time: {total_wall_time:.1f}s. Model saved to {save_path}")

    np.save(os.path.join(save_dir, 'history.npy'), history)
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train', alpha=0.8)
    plt.plot(history['test_loss'], label='Test', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('Masked MSE (residual target)')
    plt.legend(); plt.title('MscaleDeepONet on Task1 residual (mean-subtracted) target')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150)


if __name__ == '__main__':
    main()
