"""
Train Fusion_DeepONet_NoScalars (branch+trunk minor scalar params removed,
skip connections intact) under the exact same conditions as the existing
fusion_deeponet_cyl400_param100k_decay100 benchmark, for direct comparison.
"""
import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model_fusion_deeponet_noscalars import Fusion_DeepONet_NoScalars
from data_loader_fast import CylinderDatasetFast, FastBatchIterator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=87)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_path', type=str, default='./cylinder_full_2400_400_seed42.pt')
    parser.add_argument('--lr_decay_step', type=int, default=100)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.91)
    parser.add_argument('--run_tag', type=str, default='cyl400_param100k_decay100_noscalars')
    parser.add_argument('--rel2_log_interval', type=int, default=5)
    return parser.parse_args()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    torch.manual_seed(seed); random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    save_dir = os.path.join('output', f"fusion_deeponet_{args.run_tag}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    rel2_log_path = os.path.join(save_dir, 'rel2_history.csv')
    with open(rel2_log_path, 'w') as f:
        f.write('epoch,test_rel2_error,timestamp\n')

    run_info_path = os.path.join(save_dir, 'run_info.txt')
    with open(run_info_path, 'w') as f:
        f.write(f"model=fusion_deeponet_noscalars\nhidden_dim={args.hidden_dim}\n"
                f"data_path={args.data_path}\nepochs={args.epochs}\nbatch_size={args.batch_size}\n"
                f"lr={args.lr}\nlr_decay_step={args.lr_decay_step}\nlr_decay_gamma={args.lr_decay_gamma}\n"
                f"run_tag={args.run_tag}\n")

    dataset = CylinderDatasetFast(args.data_path, mode='deeponet', device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)

    train_tensors = (dataset.branch_all[train_idx], dataset.target_all[train_idx])
    test_tensors = (dataset.branch_all[test_idx], dataset.target_all[test_idx])
    train_loader = FastBatchIterator(train_tensors, batch_size=args.batch_size, shuffle=True, device=device)
    test_loader = FastBatchIterator(test_tensors, batch_size=args.batch_size, shuffle=False, device=device)

    model = Fusion_DeepONet_NoScalars(branch_dim=2, trunk_dim=2, hidden_dim=args.hidden_dim,
                                       num_outputs=4, depth=5, activation='GELU').to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params}")
    with open(run_info_path, 'a') as f:
        f.write(f"num_params={num_params}\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        compute_rel2 = (epoch + 1) % args.rel2_log_interval == 0
        rel2_sum, rel2_count = 0.0, 0
        model.train()
        train_loss_acc = 0.0
        for x_branch, y in train_loader:
            optimizer.zero_grad()
            x_branch2 = x_branch[:, :2]
            x_trunk = dataset.trunk_shared
            pred = model(x_branch2, x_trunk)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()
        scheduler.step()
        avg_train_loss = train_loss_acc / len(train_loader)

        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for x_branch, y in test_loader:
                x_branch2 = x_branch[:, :2]
                x_trunk = dataset.trunk_shared
                pred = model(x_branch2, x_trunk)
                loss = criterion(pred, y)
                test_loss_acc += loss.item()
                if compute_rel2:
                    pred_flat = pred.reshape(pred.shape[0], -1)
                    y_flat = y.reshape(y.shape[0], -1)
                    per_sample_rel2 = torch.norm(pred_flat - y_flat, p=2, dim=1) / \
                                       (torch.norm(y_flat, p=2, dim=1) + 1e-8)
                    rel2_sum += per_sample_rel2.sum().item()
                    rel2_count += pred_flat.shape[0]
        avg_test_loss = test_loss_acc / len(test_loader)

        if compute_rel2:
            test_rel2 = rel2_sum / rel2_count
            now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            with open(rel2_log_path, 'a') as f:
                f.write(f'{epoch + 1},{test_rel2:.6g},{now_str}\n')
            pbar.set_postfix({'train': f'{avg_train_loss:.4g}', 'test': f'{avg_test_loss:.4g}',
                               'rel2': f'{test_rel2:.4g}'})

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)

    print(f"Training Complete! Best Test Loss: {best_test_loss:.4g}. Model saved to {save_path}")


if __name__ == '__main__':
    main()
