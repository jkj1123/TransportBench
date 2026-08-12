import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import GaussianNormalizer
from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder2d
from model_pt import PointTransformer
from model_mscale_deeponet import MscaleDeepONet
from model_hyperdeeponet import HyperDeepONet
from model_c_hyperdeeponet import c_HyperDeepONet
from model_fusion_deeponet import Fusion_DeepONet
from model_hyper_mscale_deeponet import HyperMscaleDeepONet

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation for Task 4: Double Cone Flow")
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt', 'hyperdeeponet', 'mscale_deeponet',
                                 'c_hyperdeeponet', 'fusion_deeponet', 'hyper_mscale_deeponet'],
                        help='Choose the model to evaluate')
    parser.add_argument('--no_fourier', action='store_true',
                        help='Model does not use Fourier encoding')
    parser.add_argument('--data_path', type=str, default='./data/double_cone_dataset_with_physics.pt', 
                        help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model_{}.pth', 
                        help='Path format to checkpoint')
    parser.add_argument('--sample_idx', type=int, default=50, 
                        help='Global sample index to visualize (Default: 50 for Benchmark Case)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save')
    return parser.parse_args()

def main():
    args = get_args()
    use_fourier = not args.no_fourier
    fourier_suffix = "_fourier" if use_fourier else "_nofourier"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Locate checkpoint
    ckpt_path = args.checkpoint.format(args.model + fourier_suffix) 
    if not os.path.exists(ckpt_path):
        ckpt_path_fallback = args.checkpoint.format(args.model)
        if os.path.exists(ckpt_path_fallback):
            ckpt_path = ckpt_path_fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found for {args.model}! Checked: {ckpt_path}")
            
    print(f"Loading checkpoint from: {ckpt_path}")

    # Load data and statistics
    print(f"Loading data from {args.data_path}...")
    data_full = torch.load(args.data_path, weights_only=False)
    x_data = data_full['x'].float()
    y_data = data_full['y'].float()
    
    y_data_log = y_data.clone()
    y_data_log[:, 3, :, :] = torch.log10(y_data[:, 3, :, :] + 1e-6)
    
    x_mean = torch.mean(x_data, dim=(0, 2, 3), keepdim=True).to(device)
    x_std = torch.std(x_data, dim=(0, 2, 3), keepdim=True).to(device)
    y_mean = torch.mean(y_data_log, dim=(0, 2, 3), keepdim=True).to(device)
    y_std = torch.std(y_data_log, dim=(0, 2, 3), keepdim=True).to(device)
    
    x_norm = GaussianNormalizer(mean=x_mean, std=x_std)
    y_norm = GaussianNormalizer(mean=y_mean, std=y_std)

    # Load model and weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', {})
    checkpoint_fourier = cfg.get('use_fourier', use_fourier)
    
    if args.model == 'fno':
        model = FNO2d(modes1=8, modes2=64, width=64, in_channels=5, out_channels=4, use_fourier=checkpoint_fourier)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=5, out_channels=4, features=64, use_fourier=checkpoint_fourier)
    elif args.model == 'vit':
        model = VisionTransformer(in_channels=5, out_channels=4, embed_dim=512, depth=10, use_fourier=checkpoint_fourier)
    elif args.model == 'ae':
        model = AutoEncoder2d(in_channels=5, out_channels=4, features=128, use_fourier=checkpoint_fourier)
    elif args.model == 'deeponet':
        model = DeepONet2d(in_channels=5, out_channels=4, basis_size=256, use_fourier=checkpoint_fourier)
    elif args.model == 'pt':
        model = PointTransformer(in_channels=5, out_channels=4, latent_dim=512, num_latents=1024, depth=10, use_fourier=checkpoint_fourier)
    elif args.model == 'mscale_deeponet':
        model = MscaleDeepONet(branch_dim=3, trunk_dim=2, hidden_dim=195, num_outputs=4, scales=[1, 2, 4, 8, 16], depth=4, activation='GELU')
    elif args.model == 'hyperdeeponet':
        model = HyperDeepONet(branch_dim=3, trunk_dim=2, hidden_dim=78, num_outputs=4, trunk_depth=3, branch_depth=3, activation='GELU')
    elif args.model == 'c_hyperdeeponet':
        model = c_HyperDeepONet(branch_dim=3, trunk_dim=2, hidden_dim=77, num_outputs=4, trunk_depth=3, branch_depth=3, activation='GELU', num_chunk=2)
    elif args.model == 'fusion_deeponet':
        model = Fusion_DeepONet(branch_dim=3, trunk_dim=2, hidden_dim=77, num_outputs=4, depth=5, activation='Tanh')
    elif args.model == 'hyper_mscale_deeponet':
        model = HyperMscaleDeepONet(branch_dim=3, trunk_dim=2, hidden_dim=68, num_outputs=4, depth=4, activation='GELU')

    model = model.to(device)
    model.load_state_dict(checkpoint.get('model_state', checkpoint))
    model.eval()

    # ===== Test-set Relative L2 Error (reproduces train/test split from data_loader) =====
    n_total = x_data.shape[0]
    if n_total == 51:
        n_train = 45
    elif n_total == 32:
        n_train = 28
    else:
        n_train = n_total - max(1, int(n_total * 0.1))
    g_cpu = torch.Generator(); g_cpu.manual_seed(42)
    test_idx = torch.randperm(n_total, generator=g_cpu)[n_train:].tolist()

    # Computed like the README "Pressure Field Rel L2": PHYSICAL space,
    # pressure decoded back from log10 via pow10. The 'p' channel is the README metric.
    ch_names = ['rho', 'u', 'v', 'p']
    rel_l2 = torch.zeros(len(ch_names))
    with torch.no_grad():
        for i in test_idx:
            xin = x_data[i].unsqueeze(0).to(device)
            gt = y_data[i].unsqueeze(0).to(device)                  # physical GT (pressure in Pa)
            pred_log = y_norm.decode(model(x_norm.encode(xin)))
            pred = pred_log.clone()
            pred[:, 3, :, :] = torch.pow(10, pred_log[:, 3, :, :])   # log10 -> physical pressure
            for c in range(len(ch_names)):
                num = torch.linalg.norm(pred[0, c] - gt[0, c])
                den = torch.linalg.norm(gt[0, c]) + 1e-8
                rel_l2[c] += (num / den).item()
    rel_l2 /= len(test_idx)

    print(f"\n===== Test-set Relative L2 Error (physical space, {len(test_idx)} samples, idx={sorted(test_idx)}) =====")
    for c, name in enumerate(ch_names):
        tag = "   <-- README Pressure Field Rel L2" if name == 'p' else ""
        print(f"  {name:>3s}: {rel_l2[c].item():.4%}{tag}")
    print(f"  per-channel mean: {rel_l2.mean().item():.4%}\n")
    # ===== end test-set evaluation =====

    # Extract benchmark sample
    actual_idx = args.sample_idx
    print(f"🎯 Successfully extracted Benchmark Case (Global Idx {actual_idx}).")

    x_input = x_data[actual_idx].unsqueeze(0).to(device)
    y_true_phys = y_data[actual_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        x_encoded = x_norm.encode(x_input)
        pred_encoded = model(x_encoded)
        y_pred_log = y_norm.decode(pred_encoded)

    y_pred_phys = y_pred_log.clone()
    y_pred_phys[:, 3, :, :] = torch.pow(10, y_pred_log[:, 3, :, :])
    error = torch.abs(y_true_phys - y_pred_phys)

    # Visualization
    def to_np(t): return t.squeeze(0).cpu().numpy()
    
    x_np, y_true, y_pred, err = to_np(x_input), to_np(y_true_phys), to_np(y_pred_phys), to_np(error)
    Grid_X, Grid_Y = x_np[0], x_np[1]
    
    fourier_text = "With Fourier" if checkpoint_fourier else "No Fourier"
    title_str = f"{args.model.upper()} | Benchmark Case (Global Idx: {actual_idx}) | {fourier_text}"
    plot_configs = [{'name': 'Velocity u (m/s)', 'idx': 1, 'cmap': 'jet'},
                    {'name': 'Pressure p (Pa)', 'idx': 3, 'cmap': 'magma'}]

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3)
    plt.suptitle(title_str, fontsize=16)

    for row_idx, cfg in enumerate(plot_configs):
        var_idx, cmap = cfg['idx'], cfg['cmap']
        gt, pred, e = y_true[var_idx], y_pred[var_idx], err[var_idx]
        
        l2_err = np.linalg.norm(e) / (np.linalg.norm(gt) + 1e-8)
        vmin, vmax = min(gt.min(), pred.min()), max(np.percentile(gt, 99), np.percentile(pred, 99))

        ax1 = fig.add_subplot(gs[row_idx, 0])
        im1 = ax1.pcolormesh(Grid_X, Grid_Y, gt, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        ax1.set_title(f"GT {cfg['name']}"); ax1.axis('equal'); ax1.axis('off'); plt.colorbar(im1, ax=ax1)

        ax2 = fig.add_subplot(gs[row_idx, 1])
        im2 = ax2.pcolormesh(Grid_X, Grid_Y, pred, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        ax2.set_title(f"Pred {cfg['name']}"); ax2.axis('equal'); ax2.axis('off'); plt.colorbar(im2, ax=ax2)

        ax3 = fig.add_subplot(gs[row_idx, 2])
        im3 = ax3.pcolormesh(Grid_X, Grid_Y, e, cmap='inferno', shading='gouraud')
        ax3.set_title(f"Error (Rel L2={l2_err:.1%})"); ax3.axis('equal'); ax3.axis('off'); plt.colorbar(im3, ax=ax3)

    # Extract wall curves
    wall_idx = 2
    wall_x = Grid_X[wall_idx, :]
    ax_wall = fig.add_subplot(gs[2, :])
    ax_wall.plot(wall_x, y_true[3][wall_idx, :], 'k-', lw=2.5, label='CFD Ground Truth')
    ax_wall.plot(wall_x, y_pred[3][wall_idx, :], 'r--', lw=2.5, label=f'{args.model.upper()} Pred ({fourier_text})')
    ax_wall.set_title(f"Near-Wall Pressure Distribution (Benchmark Case)")
    ax_wall.set_xlabel("X (m)"); ax_wall.set_ylabel("Pressure (Pa)"); ax_wall.legend(); ax_wall.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_img = os.path.join(args.output_dir, f"inference_{args.model}{fourier_suffix}_benchmark.png")
    plt.savefig(output_img, dpi=150)
    print(f"Saved visualization to: {output_img}")

if __name__ == "__main__":
    main()