import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from model_deeponet import BoltzmannDeepONet
from model_fno import FNO
from model_unet import UNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder
from model_pt import PointTransformer
from model_mscale_deeponet import MscaleDeepONet
from model_hyperdeeponet import HyperDeepONet
from data_loader import CavityDataset

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Task III: Cavity Flow")
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt','mscale_deeponet', 'hyperdeeponet'], help='Choose model')
    parser.add_argument('--data_dir', type=str, default='./data/cavity', help='Path to .npz data')
    parser.add_argument('--pt_path', type=str, default='./cavity_dataset.pt',
                        help='Compact .pt dataset file to load directly (built from .npz if missing)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model_{}.pth', help='Path to weights')
    return parser.parse_args()

def main():

    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prevent OOM for Point Transformer
    if args.model == 'pt': device = 'cpu'

    print(f"📊 Starting Evaluation | Model: {args.model.upper()} | Device: {device}")

    dataset = CavityDataset(data_dir=args.data_dir, mode='test', model_type='fno', pt_path=args.pt_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    if args.model == 'fno':
        model = FNO(modes1=12, modes2=12, width=32, in_channels=3, out_channels=10)
    elif args.model == 'unet':
        model = UNet(n_channels=3, n_classes=10)
    elif args.model == 'vit':
        model = VisionTransformer(img_size=50, patch_size=5, in_chans=3, out_chans=10, embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = AutoEncoder(in_channels=3, out_channels=10, base_dim=32)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=230, num_outputs=10, depth=5)
    elif args.model == 'pt':
        model = PointTransformer(in_channels=3, out_channels=10, embed_dim=120, depth=4)
    elif args.model == 'mscale_deeponet':
        model = MscaleDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=175, num_outputs=10,
                               scales=[1, 2, 4, 8, 16], depth=4, activation='GELU')
    elif args.model == 'hyperdeeponet':
        model = HyperDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=77, num_outputs=10,
                              trunk_depth=3, branch_depth=3, activation='GELU')

    model = model.to(device)

    ckpt_path = args.checkpoint.format(args.model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    total_mae, total_l2_error = 0.0, 0.0
    criterion_mae = nn.L1Loss(reduction='sum')

    plot_gt, plot_pred = None, None

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            if args.model in['unet', 'vit']:
                x_in = x.permute(0, 3, 1, 2)
                pred = model(x_in)
                if args.model == 'unet': pred = pred.permute(0, 2, 3, 1)
            elif args.model in ['deeponet', 'mscale_deeponet']:
                B = x.shape[0]
                x_branch = x[:, 0, 0, 0:1]
                x_trunk = x[0, :, :, 1:3].reshape(-1, 2)
                pred = model(x_branch, x_trunk).view(B, 50, 50, 10)
            elif args.model in ['hyperdeeponet']:
                B = x.shape[0]
                x_branch = x[:, 0, 0, 0:1]
                x0 = x[:, :, :, 1:3]
                x_trunk = x0.reshape(x0.shape[0], -1, x0.shape[-1])
                y = y.view(B, 50,50,10)
                pred = model(x_branch, x_trunk)
                pred = pred.reshape(pred.shape[0],50,50,pred.shape[-1])
            else:
                pred = model(x)

            total_mae += criterion_mae(pred, y).item()
            l2_err = torch.norm(pred - y, p=2) / (torch.norm(y, p=2) + 1e-8)
            total_l2_error += l2_err.item()

            if i == 0:
                mean = torch.tensor(dataset.target_mean, device=device).view(1, 1, 1, 10)
                std = torch.tensor(dataset.target_std, device=device).view(1, 1, 1, 10)

                pred_real = pred * std + mean
                target_real = y * std + mean

                plot_pred = pred_real[0].cpu().numpy()
                plot_gt = target_real[0].cpu().numpy()

                # Grid coordinates of the plotted sample (input ch1=x, ch2=y)
                plot_X = x[0, :, :, 1].cpu().numpy()
                plot_Y = x[0, :, :, 2].cpu().numpy()

    num_elements = len(test_loader.dataset) * np.prod(y.shape[1:])
    final_mae = total_mae / num_elements
    final_rel_l2 = total_l2_error / len(test_loader.dataset)

    print("-" * 50)
    print(f"🏆 Results for {args.model.upper()}: MAE = {final_mae:.5f} | Rel L2 = {final_rel_l2*100:.2f}%")

    # ==================== PLOTTING (Ground Truth | Prediction | Absolute Error) ====================
    # Target channel layout: [w(0-3), P_flat(4-7), q(8-9)]  ->  P0 = index 4, q0 = index 8
    X, Y = plot_X, plot_Y
    channels = [('P0', 4), ('q0', 8)]

    for name, idx in channels:
        Z_true = plot_gt[..., idx]    # (nx, ny)
        Z_pred = plot_pred[..., idx]  # (nx, ny)
        Z_error = np.abs(Z_pred - Z_true)  # absolute error

        # Ground truth & prediction share one color scale.
        # NOTE: pass an explicit `levels` array (not an int) so both panels use the
        # SAME contour boundaries -> identical colorbars. Passing vmin/vmax with an
        # integer level count does NOT fix the colorbar, since contourf picks levels
        # from each array's own min/max.
        vmin = min(Z_true.min(), Z_pred.min())
        vmax = max(Z_true.max(), Z_pred.max())
        levels = np.linspace(vmin, vmax, 51)

        # Relative L2 error for this channel (||true - pred|| / ||true||)
        err = np.linalg.norm(Z_true - Z_pred) / (np.linalg.norm(Z_true) + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Ground truth
        contour0 = axes[0].contourf(X, Y, Z_true, levels=levels, cmap="jet")
        fig.colorbar(contour0, ax=axes[0], label="value")
        axes[0].set_title(f"Ground Truth: {name}")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
        axes[0].set_aspect("equal")

        # Prediction
        contour1 = axes[1].contourf(X, Y, Z_pred, levels=levels, cmap="jet")
        fig.colorbar(contour1, ax=axes[1], label="value")
        axes[1].set_title(f"{args.model.upper()} Prediction: {name}")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
        axes[1].set_aspect("equal")

        # Absolute error
        err_max = 0.005*Z_true.max()
        #err_max = 0.0001
        err_levels = np.linspace(0, err_max if err_max > 0 else 1e-8, 51)
        contour2 = axes[2].contourf(X, Y, Z_error, levels=err_levels, cmap="jet")
        fig.colorbar(contour2, ax=axes[2], label="absolute error")
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
        axes[2].set_aspect("equal")

        fig.suptitle(f"{args.model.upper()} | Channel {name} | Rel. L2 error = {err*100:.2f}%",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_fig_path = f"evaluation_cavity_{args.model}_{name}_0720.png"
        plt.savefig(save_fig_path, dpi=300)
        plt.close(fig)
        print(f"📸 [{name}] plot saved as: {save_fig_path} | Rel. L2 error = {err*100:.2f}%")

if __name__ == "__main__":
    main()
