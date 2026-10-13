"""
Overhead-fixed training script for Task I (Airfoil).

Same as train.py in every respect (args, model configs, optimizer, scheduler,
logging, checkpointing) EXCEPT the data path: uses AirfoilDatasetFast +
FastBatchIterator (data_loader_fast.py) instead of AirfoilDataset +
torch.utils.data.DataLoader + the _IndexedSubset/mask_source lookup dance,
which removes the per-epoch redo of reshape/permute + collate + mask lookup
for a split that never changes across a run. train.py/data_loader.py are
untouched.

The per-sample forward loop for coordinate-based models (`for i in
range(batch_size): pred_i = model(x_branch[i:i+1], x_trunk[i])`) is
deliberately kept as-is -- it's a model-batching detail, not a data-loading
one, and changing it would need re-verifying every baseline model's batched
trunk support.
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

from model_deeponet import BoltzmannDeepONet
from model_fno import FNO2d
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder
from model_pt import PointTransformerONet
from model_mscale_deeponet import MscaleDeepONet
from model_hyperdeeponet import HyperDeepONet
from model_hyper_mscale_deeponet import HyperMscaleDeepONet
from model_c_hyperdeeponet import c_HyperDeepONet
from model_fusion_deeponet import Fusion_DeepONet
from model_residual_fusion_deeponet import Residual_Fusion_DeepONet
from data_loader_fast import AirfoilDatasetFast, FastBatchIterator

def get_args():
    parser = argparse.ArgumentParser(description="TransportBench - Task I: Airfoil Flow (overhead-fixed loader)")
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt', 'mscale_deeponet', 'hyperdeeponet', 'c_hyperdeeponet', 'hyper_mscale_deeponet', 'fusion_deeponet', 'residual_fusion_deeponet'],
                        help='Choose the baseline model')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data/airfoil_unified_128x128.pt', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--lr_decay_step', type=int, default=0,
                        help='Decay LR every N epochs. 0 = disabled (paper default: no scheduler).')
    parser.add_argument('--lr_decay_gamma', type=float, default=1.0,
                        help='Multiplicative LR decay factor applied every --lr_decay_step epochs.')
    parser.add_argument('--run_tag', type=str, default='',
                        help='Optional suffix so this run writes to output/<model>_<run_tag>/ instead of '
                             'output/<model>/ (avoids clobbering a concurrent/previous run of the same model).')
    parser.add_argument('--hidden_dim', type=int, default=0,
                        help='Override hidden_dim for hyperdeeponet (0 = use model default, 76). '
                             'Lets a param-count sweep pick hidden_dim without touching the model file.')
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

    print(f"Starting Task I Training [fast loader] | Model: {args.model.upper()} | Device: {device}")

    model_dir_name = f"{args.model}_{args.run_tag}" if args.run_tag else args.model
    args.save_dir = os.path.join('output', model_dir_name)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"best_model.pth")

    data_mode = 'fno' if args.model in ['fno', 'unet', 'vit', 'ae'] else 'deeponet'

    dataset = AirfoilDatasetFast(args.data_path, mode=data_mode, device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)

    if data_mode == 'fno':
        train_tensors = (dataset.x_input_all[train_idx], dataset.target_all[train_idx])
        test_tensors = (dataset.x_input_all[test_idx], dataset.target_all[test_idx])
    else:
        train_tensors = (dataset.branch_all[train_idx], dataset.trunk_all[train_idx],
                          dataset.target_all[train_idx], dataset.geo_mask_flat[train_idx])
        test_tensors = (dataset.branch_all[test_idx], dataset.trunk_all[test_idx],
                         dataset.target_all[test_idx], dataset.geo_mask_flat[test_idx])

    train_loader = FastBatchIterator(train_tensors, batch_size=args.batch_size, shuffle=True, device=device)
    test_loader = FastBatchIterator(test_tensors, batch_size=args.batch_size, shuffle=False, device=device)

    # Initialize model
    if args.model == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=28, in_channels=3, out_channels=4)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=3, out_channels=4, base_dim=20)
    elif args.model == 'vit':
        model = VisionTransformer(embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = AutoEncoder(in_channels=3, out_channels=4, base_dim=24)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=256, num_outputs=4, depth=5)
    elif args.model == 'pt':
        model = PointTransformerONet(hidden_dim=256, num_outputs=4)
    elif args.model == 'mscale_deeponet':
        model = MscaleDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=181, num_outputs=4,
                               scales=[1, 2, 4, 8, 16], depth=4, activation='GELU')
    elif args.model == 'hyperdeeponet':
        hd_hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 76
        model = HyperDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=hd_hidden_dim, num_outputs=4,
                              trunk_depth=3, branch_depth=3, activation='GELU')
    elif args.model == 'c_hyperdeeponet':
        model = c_HyperDeepONet(branch_dim=674, trunk_dim=2, trunk_hidden_dim=256, branch_hidden_dim=225,
                                num_outputs=4, trunk_depth=4, branch_depth=3, activation='GELU',
                                chunk_in=1024, chunk_out=512)
    elif args.model == 'hyper_mscale_deeponet':
        model = HyperMscaleDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=67, num_outputs=4,
                                    depth=4, activation='GELU')
    elif args.model == 'fusion_deeponet':
        model = Fusion_DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=253, num_outputs=4,
                                depth=5, activation='GELU')
    elif args.model == 'residual_fusion_deeponet':
        model = Residual_Fusion_DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=253, num_outputs=4,
                                         depth=5, activation='GELU')

    model = model.to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    if args.lr_decay_step > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    history = {'train_loss': [], 'test_loss': []}

    best_test_rel2 = float('inf')
    rel2_log_path = os.path.join(args.save_dir, 'rel2_history.csv')
    with open(rel2_log_path, 'w') as f:
        f.write('epoch,wall_time_sec,test_rel2,best_test_rel2\n')
    train_start_time = time.time()

    print("Training Started...")
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        rel2_sum, rel2_count = 0.0, 0
        # --- Train Phase ---
        model.train()
        train_loss_acc = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            if data_mode == 'fno':
                x, y = batch
                pred = model(x)
                mask = x[:, 0:1, :, :]
                loss = criterion(pred * mask, y * mask)
            else:
                x_branch, x_trunk, y, mask = batch
                batch_size = x_branch.shape[0]

                pred_list = []
                for i in range(batch_size):
                    pred_i = model(x_branch[i:i+1], x_trunk[i])
                    pred_list.append(pred_i)
                pred = torch.cat(pred_list, dim=0)

                loss = criterion(pred * mask, y * mask)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()

        if scheduler is not None:
            scheduler.step()
        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Test Phase ---
        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if data_mode == 'fno':
                    x, y = batch
                    pred = model(x)
                    mask = x[:, 0:1, :, :]
                    loss = criterion(pred * mask, y * mask)
                else:
                    x_branch, x_trunk, y, mask = batch
                    batch_size = x_branch.shape[0]

                    pred_list = []
                    for i in range(batch_size):
                        pred_i = model(x_branch[i:i+1], x_trunk[i])
                        pred_list.append(pred_i)
                    pred = torch.cat(pred_list, dim=0)

                    loss = criterion(pred * mask, y * mask)

                test_loss_acc += loss.item()

                pred_masked = (pred * mask).reshape(pred.shape[0], -1)
                y_masked = (y * mask).reshape(y.shape[0], -1)
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
            torch.save(model.state_dict(), save_path)
            saved_flag = " [BEST SAVED]"
        else:
            saved_flag = ""

        pbar.set_postfix({
            'train': f'{avg_train_loss:.4g}',
            'test': f'{avg_test_loss:.4g}',
            'best': f'{best_test_loss:.4g}',
            'rel2': f'{test_rel2:.4g}',
            'best_rel2': f'{best_test_rel2:.4g}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })
        if saved_flag and (epoch + 1) % 50 == 0:
            print(f"  [BEST SAVED @ epoch {epoch+1}]")

    total_wall_time = time.time() - train_start_time
    print(f"Training Complete! Best Test Loss: {best_test_loss:.4g} | Best Test Rel L2: {best_test_rel2:.4g} | "
          f"Wall Time: {total_wall_time:.1f}s. Model saved to {save_path}")
    print(f"Per-epoch rel2/walltime log saved to {rel2_log_path}")

    np.save(os.path.join(args.save_dir, 'history.npy'), history)

    fig_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train', alpha=0.8)
    plt.plot(history['test_loss'], label='Test', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.model.upper()} Loss Curve [fast loader]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {fig_path}")

if __name__ == "__main__":
    main()
