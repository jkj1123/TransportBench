"""
Parameterized evaluation for the epoch-budget benchmark.

Same evaluation logic as eval_0727.py (Rel L2 / MAE + P0,q0 contour plots),
but takes an explicit checkpoint path, output directory and filename tag so
many (model, epoch-budget) runs can be evaluated without overwriting each
other's images, and writes the numeric results to a small JSON file next to
the images so a driver script can collect them without scraping stdout.
"""
import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from model_deeponet import BoltzmannDeepONet
from model_mscale_deeponet import MscaleDeepONet
from model_hyperdeeponet import HyperDeepONet
from model_fusion_deeponet import Fusion_DeepONet
from model_hyper_mscale_deeponet import HyperMscaleDeepONet
from model_hyper_mscale_deeponet_gelu import HyperMscaleDeepONetGELU
from data_loader import CavityDataset


def get_args():
    parser = argparse.ArgumentParser(description="Parameterized eval for epoch-budget benchmark")
    parser.add_argument('--model', type=str, required=True,
                         choices=['deeponet', 'mscale_deeponet', 'hyperdeeponet',
                                  'fusion_deeponet', 'hyper_mscale_deeponet',
                                  'hyper_mscale_deeponet_gelu'])
    parser.add_argument('--pt_path', type=str, default='./cavity_dataset.pt')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True, help='Where to save images/json')
    parser.add_argument('--tag', type=str, required=True, help='Filename tag, e.g. deeponet_1000')
    return parser.parse_args()


def build_model(name):
    if name == 'deeponet':
        return BoltzmannDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=230, num_outputs=10, depth=5)
    elif name == 'mscale_deeponet':
        return MscaleDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=175, num_outputs=10,
                               scales=[1, 2, 4, 8, 16], depth=4, activation='GELU')
    elif name == 'hyperdeeponet':
        return HyperDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=77, num_outputs=10,
                              trunk_depth=3, branch_depth=3, activation='GELU')
    elif name == 'fusion_deeponet':
        return Fusion_DeepONet(branch_dim=1, trunk_dim=2, hidden_dim=212, num_outputs=10,
                                depth=7, activation='Tanh')
    elif name == 'hyper_mscale_deeponet':
        return HyperMscaleDeepONet(branch_dim=1, trunk_dim=2, hidden_dim=68, num_outputs=10,
                                    depth=4, activation='GELU')
    elif name == 'hyper_mscale_deeponet_gelu':
        return HyperMscaleDeepONetGELU(branch_dim=1, trunk_dim=2, hidden_dim=68, num_outputs=10,
                                    depth=4, activation='GELU')
    raise ValueError(name)


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = CavityDataset(data_dir='./data/cavity', mode='test', model_type='fno', pt_path=args.pt_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    dataset.fit_train_normalizer(train_data.indices)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    total_mae, total_l2_error = 0.0, 0.0
    criterion_mae = nn.L1Loss(reduction='sum')
    plot_gt, plot_pred, plot_X, plot_Y = None, None, None, None

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            if args.model in ['deeponet', 'mscale_deeponet', 'fusion_deeponet']:
                B = x.shape[0]
                x_branch = x[:, 0, 0, 0:1]
                x_trunk = x[0, :, :, 1:3].reshape(-1, 2)
                pred = model(x_branch, x_trunk).view(B, 50, 50, 10)
            elif args.model in ['hyperdeeponet', 'hyper_mscale_deeponet', 'hyper_mscale_deeponet_gelu']:
                B = x.shape[0]
                x_branch = x[:, 0, 0, 0:1]
                x0 = x[:, :, :, 1:3]
                x_trunk = x0.reshape(x0.shape[0], -1, x0.shape[-1])
                y = y.view(B, 50, 50, 10)
                pred = model(x_branch, x_trunk)
                pred = pred.reshape(pred.shape[0], 50, 50, pred.shape[-1])
            else:
                raise ValueError(args.model)

            total_mae += criterion_mae(pred, y).item()
            l2_err = torch.norm(pred - y, p=2) / (torch.norm(y, p=2) + 1e-8)
            total_l2_error += l2_err.item()

            if i == 0:
                tmin = dataset.target_min.to(device).view(1, 1, 1, 10)
                tmax = dataset.target_max.to(device).view(1, 1, 1, 10)
                pred_real = pred * (tmax - tmin) + tmin
                target_real = y * (tmax - tmin) + tmin
                plot_pred = pred_real[0].cpu().numpy()
                plot_gt = target_real[0].cpu().numpy()
                plot_X = x[0, :, :, 1].cpu().numpy()
                plot_Y = x[0, :, :, 2].cpu().numpy()

    num_elements = len(test_loader.dataset) * np.prod(y.shape[1:])
    final_mae = total_mae / num_elements
    final_rel_l2 = total_l2_error / len(test_loader.dataset)

    print(f"[{args.tag}] Rel L2 = {final_rel_l2*100:.3f}% | MAE = {final_mae:.5f}")

    channels = [('P0', 4), ('q0', 8)]
    channel_errors = {}
    X, Y = plot_X, plot_Y

    for name, idx in channels:
        Z_true = plot_gt[..., idx]
        Z_pred = plot_pred[..., idx]
        Z_error = np.abs(Z_pred - Z_true)

        vmin = min(Z_true.min(), Z_pred.min())
        vmax = max(Z_true.max(), Z_pred.max())
        levels = np.linspace(vmin, vmax, 51)
        err = np.linalg.norm(Z_true - Z_pred) / (np.linalg.norm(Z_true) + 1e-8)
        channel_errors[name] = float(err)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        c0 = axes[0].contourf(X, Y, Z_true, levels=levels, cmap="jet")
        fig.colorbar(c0, ax=axes[0], label="value")
        axes[0].set_title(f"Ground Truth: {name}")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_aspect("equal")

        c1 = axes[1].contourf(X, Y, Z_pred, levels=levels, cmap="jet")
        fig.colorbar(c1, ax=axes[1], label="value")
        axes[1].set_title(f"{args.model} Prediction: {name}")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].set_aspect("equal")

        err_max = 0.005 * Z_true.max()
        err_levels = np.linspace(0, err_max if err_max > 0 else 1e-8, 51)
        c2 = axes[2].contourf(X, Y, Z_error, levels=err_levels, cmap="jet")
        fig.colorbar(c2, ax=axes[2], label="absolute error")
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel("x"); axes[2].set_ylabel("y"); axes[2].set_aspect("equal")

        fig.suptitle(f"{args.model} | {args.tag} | Channel {name} | Rel. L2 = {err*100:.2f}%",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(args.out_dir, f"{args.tag}_{name}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

    result = {
        'model': args.model,
        'tag': args.tag,
        'rel_l2_overall': final_rel_l2,
        'mae_overall': final_mae,
        'rel_l2_P0': channel_errors['P0'],
        'rel_l2_q0': channel_errors['q0'],
    }
    with open(os.path.join(args.out_dir, f"{args.tag}_result.json"), 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
