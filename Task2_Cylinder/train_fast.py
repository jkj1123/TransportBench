"""
Overhead-fixed training script for Task II (Cylinder).

Identical to train.py in every respect (args, model configs, optimizer,
scheduler, logging, checkpointing) EXCEPT the data path: it uses
CylinderDatasetFast + FastBatchIterator (data_loader_fast.py) instead of
CylinderDataset + torch.utils.data.DataLoader, which removes a ~47ms/epoch
fixed cost (DataLoader's per-epoch default_collate re-stacking the whole
dataset from scratch) that was independent of model size and dominated wall
time for small hyperdeeponet configs. train.py / data_loader.py are untouched.
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
from model_ae import Autoencoder
from model_pt import PointTransformer
from model_mscale_deeponet import MscaleDeepONet
from model_hyperdeeponet import HyperDeepONet
from model_L_hyperdeeponet import HyperDeepONet as L_HyperDeepONet
from model_ac_hyperdeeponet import HyperDeepONet as AC_HyperDeepONet
from model_hyper_mscale_deeponet import HyperMscaleDeepONet
from model_c_hyperdeeponet import c_HyperDeepONet
from model_cf_hyperdeeponet import c_HyperDeepONet as CF_HyperDeepONet
from model_fusion_deeponet import Fusion_DeepONet
from model_residual_fusion_deeponet import Residual_Fusion_DeepONet
from data_loader_fast import CylinderDatasetFast, FastBatchIterator

def get_args():
    parser = argparse.ArgumentParser(description="TransportBench - Task II: Cylinder Flow (overhead-fixed loader)")
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt', 'mscale_deeponet', 'hyperdeeponet', 'L_hyperdeeponet', 'c_hyperdeeponet', 'cf_hyperdeeponet', 'ac_hyperdeeponet', 'hyper_mscale_deeponet', 'fusion_deeponet', 'residual_fusion_deeponet'],
                        help='Choose the baseline model')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='./data/cylinder_full_2400.pt', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--lr_decay_step', type=int, default=0,
                        help='Decay LR every N epochs (one scheduler.step() per epoch, independent of '
                             '--batch_size / iterations-per-epoch). 0 = disabled (paper default: no scheduler).')
    parser.add_argument('--lr_decay_gamma', type=float, default=1.0,
                        help='Multiplicative LR decay factor applied every --lr_decay_step epochs.')
    parser.add_argument('--run_tag', type=str, default='',
                        help='Optional suffix so this run writes to output/<model>_<run_tag>/ instead of '
                             'output/<model>/, to avoid clobbering/mixing with a prior run of the same model.')
    parser.add_argument('--hidden_dim', type=int, default=0,
                        help='Override hidden_dim for hyperdeeponet (0 = use model default, 78 / ~1M params). '
                             'Lets a param-count sweep pick hidden_dim without touching the model file.')
    parser.add_argument('--c_num_basis', type=int, default=0,
                        help='c_hyperdeeponet: override num_basis (0 = default 128).')
    parser.add_argument('--c_chunk_in', type=int, default=0,
                        help='c_hyperdeeponet: override chunk_in (0 = default 2525).')
    parser.add_argument('--c_chunk_out', type=int, default=0,
                        help='c_hyperdeeponet: override chunk_out (0 = default 384).')
    parser.add_argument('--c_num_chunks', type=int, default=0,
                        help='c_hyperdeeponet: fix num_chunks directly, independent of hidden_dim '
                             '(0 = off, num_chunks derived from chunk_out as usual). When set, '
                             'chunk_out is auto-derived to cover param_size and --c_chunk_out is ignored.')
    parser.add_argument('--cf_num_basis', type=int, default=0,
                        help='cf_hyperdeeponet: override num_basis (0 = default 128).')
    parser.add_argument('--cf_chunk_in', type=int, default=0,
                        help='cf_hyperdeeponet: override chunk_in (0 = default 2525).')
    parser.add_argument('--cf_chunk_out', type=int, default=0,
                        help='cf_hyperdeeponet: override chunk_out (0 = default 384).')
    parser.add_argument('--rel2_log_interval', type=int, default=50,
                        help='Log test rel-L2 error to rel2_history.csv every N epochs.')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fix random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Starting Task II Training [fast loader] | Model: {args.model.upper()} | Device: {device}")

    # Save to output/<model>/ (or output/<model>_<run_tag>/ if --run_tag given), aligned with Task1 structure
    model_dir_name = f"{args.model}_{args.run_tag}" if args.run_tag else args.model
    args.save_dir = os.path.join('output', model_dir_name)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"best_model.pth")

    # Test Rel2 Error logging, every REL2_LOG_INTERVAL epochs
    REL2_LOG_INTERVAL = args.rel2_log_interval
    rel2_log_path = os.path.join(args.save_dir, 'rel2_history.csv')
    if not os.path.exists(rel2_log_path):
        with open(rel2_log_path, 'w') as f:
            f.write('epoch,test_rel2_error,timestamp\n')

    # Per-run timestamp/metadata log (start time now; end time + elapsed appended after training)
    run_start_time = time.time()
    run_info_path = os.path.join(args.save_dir, 'run_info.txt')
    with open(run_info_path, 'w') as f:
        f.write(f"model={args.model}\n")
        f.write(f"hidden_dim_override={args.hidden_dim}\n")
        f.write(f"data_path={args.data_path}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"lr_decay_step={args.lr_decay_step}\n")
        f.write(f"lr_decay_gamma={args.lr_decay_gamma}\n")
        f.write(f"run_tag={args.run_tag}\n")
        f.write(f"start_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_start_time))}\n")

    # Determine data loading mode: grid-based vs coordinate-based
    data_mode = 'grid' if args.model in ['fno', 'unet', 'vit', 'ae'] else 'deeponet'

    dataset = CylinderDatasetFast(args.data_path, mode=data_mode, device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)

    if data_mode == 'grid':
        train_tensors = (dataset.x_input_all[train_idx], dataset.target_all[train_idx])
        test_tensors = (dataset.x_input_all[test_idx], dataset.target_all[test_idx])
    else:
        train_tensors = (dataset.branch_all[train_idx], dataset.target_all[train_idx])
        test_tensors = (dataset.branch_all[test_idx], dataset.target_all[test_idx])

    train_loader = FastBatchIterator(train_tensors, batch_size=args.batch_size, shuffle=True, device=device)
    test_loader = FastBatchIterator(test_tensors, batch_size=args.batch_size, shuffle=False, device=device)

    # Initialize model
    if args.model == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=32, in_channels=4, out_channels=4)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=4, out_channels=4, base_dim=19)
    elif args.model == 'vit':
        model = VisionTransformer(img_size=(128, 192), patch_size=8, in_chans=4, out_chans=4, embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = Autoencoder(in_channels=4, out_channels=4, base_width=36)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=280, num_outputs=4, depth=5)
    elif args.model == 'pt':
        model = PointTransformer(in_dim=4, out_dim=4, embed_dim=144, depth=4)
    elif args.model == 'mscale_deeponet':
        model = MscaleDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=192, num_outputs=4,
                               scales=[1, 2, 4, 8, 16], depth=4, activation='GELU')
    elif args.model == 'hyperdeeponet':
        hd_hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 78
        model = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=hd_hidden_dim, num_outputs=4,
                              trunk_depth=3, branch_depth=3, activation='GELU')
    elif args.model == 'L_hyperdeeponet':
        lh_hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 30
        model = L_HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=lh_hidden_dim, num_outputs=4,
                                trunk_depth=3, branch_depth=3, activation='GELU')
    elif args.model == 'ac_hyperdeeponet':
        ac_hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 34
        model = AC_HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=ac_hidden_dim, num_outputs=4,
                                 trunk_depth=3, branch_depth=3, activation='GELU')
    elif args.model == 'c_hyperdeeponet':
        c_hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 160
        c_num_basis = args.c_num_basis if args.c_num_basis > 0 else 128
        c_chunk_in = args.c_chunk_in if args.c_chunk_in > 0 else 2525
        c_chunk_out = args.c_chunk_out if args.c_chunk_out > 0 else 384
        c_num_chunks = args.c_num_chunks if args.c_num_chunks > 0 else None
        model = c_HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=c_hidden_dim, num_basis=c_num_basis,
                                num_outputs=4, trunk_depth=3, branch_depth=3, activation='GELU',
                                chunk_in=c_chunk_in, chunk_out=c_chunk_out, num_chunks=c_num_chunks)
    elif args.model == 'cf_hyperdeeponet':
        cf_hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 160
        cf_num_basis = args.cf_num_basis if args.cf_num_basis > 0 else 128
        cf_chunk_in = args.cf_chunk_in if args.cf_chunk_in > 0 else 2525
        cf_chunk_out = args.cf_chunk_out if args.cf_chunk_out > 0 else 384
        model = CF_HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=cf_hidden_dim, num_basis=cf_num_basis,
                                 num_outputs=4, trunk_depth=3, branch_depth=3, activation='GELU',
                                 chunk_in=cf_chunk_in, chunk_out=cf_chunk_out)
    elif args.model == 'hyper_mscale_deeponet':
        model = HyperMscaleDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=68, num_outputs=4,
                                    depth=4, activation='GELU')
    elif args.model == 'fusion_deeponet':
        fd_hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 278
        model = Fusion_DeepONet(branch_dim=2, trunk_dim=2, hidden_dim=fd_hidden_dim, num_outputs=4,
                                depth=5, activation='GELU')
    elif args.model == 'residual_fusion_deeponet':
        model = Residual_Fusion_DeepONet(branch_dim=2, trunk_dim=2, hidden_dim=278, num_outputs=4,
                                         depth=5, activation='GELU')

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.2f} M")
    with open(run_info_path, 'a') as f:
        f.write(f"num_params={num_params}\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    if args.lr_decay_step > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    history = {'train_loss': [], 'test_loss': []}

    print("Training Started...")
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        compute_rel2 = (epoch + 1) % REL2_LOG_INTERVAL == 0
        rel2_sum, rel2_count = 0.0, 0
        # --- Train Phase ---
        model.train()
        train_loss_acc = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            if data_mode == 'grid':
                x, y = batch
                pred = model(x)
            else:
                x_branch, y = batch
                x_branch = x_branch[:, :2]  # (Kn, Ma)
                x_trunk = dataset.trunk_shared  # all samples share the same grid
                pred = model(x_branch, x_trunk)

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()

        # lr decay is epoch-based (one scheduler.step() per epoch, regardless of
        # how many mini-batches/iterations that epoch contains), so --lr_decay_step
        # always means "every N epochs" no matter what --batch_size is.
        if scheduler is not None:
            scheduler.step()

        avg_train_loss = train_loss_acc / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Test Phase ---
        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if data_mode == 'grid':
                    x, y = batch
                    pred = model(x)
                else:
                    x_branch, y = batch
                    x_branch = x_branch[:, :2]
                    x_trunk = dataset.trunk_shared
                    pred = model(x_branch, x_trunk)

                loss = criterion(pred, y)
                test_loss_acc += loss.item()

                # Relative L2 Error (per-sample), only on log epochs
                if compute_rel2:
                    pred_flat = pred.reshape(pred.shape[0], -1)
                    y_flat = y.reshape(y.shape[0], -1)
                    per_sample_rel2 = torch.norm(pred_flat - y_flat, p=2, dim=1) / \
                                       (torch.norm(y_flat, p=2, dim=1) + 1e-8)
                    rel2_sum += per_sample_rel2.sum().item()
                    rel2_count += pred_flat.shape[0]

        avg_test_loss = test_loss_acc / len(test_loader)
        history['test_loss'].append(avg_test_loss)

        if compute_rel2:
            test_rel2 = rel2_sum / rel2_count
            now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            with open(rel2_log_path, 'a') as f:
                f.write(f'{epoch + 1},{test_rel2:.6g},{now_str}\n')

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            saved_flag = True
        else:
            saved_flag = False

        postfix = {
            'train': f'{avg_train_loss:.4g}',
            'test': f'{avg_test_loss:.4g}',
            'best': f'{best_test_loss:.4g}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        }
        if compute_rel2:
            postfix['rel2'] = f'{test_rel2:.4g}'
        pbar.set_postfix(postfix)
        if saved_flag and (epoch + 1) % 50 == 0:
            tqdm.write(f"  [BEST SAVED @ epoch {epoch+1}]")

    print(f"Training Complete! Best Test Loss: {best_test_loss:.4g}. Model saved to {save_path}")

    run_end_time = time.time()
    with open(run_info_path, 'a') as f:
        f.write(f"end_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_end_time))}\n")
        f.write(f"elapsed_sec={run_end_time - run_start_time:.1f}\n")
        f.write(f"best_test_loss={best_test_loss:.6g}\n")

    # Save loss history
    np.save(os.path.join(args.save_dir, 'history.npy'), history)

    # Save loss curve plot
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
