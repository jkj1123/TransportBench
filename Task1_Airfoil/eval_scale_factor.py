"""
Evaluation script for Task I (Airfoil) with scale-factor reporting.

Identical to eval.py (same metrics + visualizations), plus: for
HyperMscaleDeepONet, the learned per-sample scale factor
(scale = exp(log_scale), the last entry the branch net outputs) is
collected across the test set and its mean/std are printed.
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# Import models and data loader
from model_deeponet import BoltzmannDeepONet
from model_fno import FNO2d
from model_unet import FluidUNet
from model_vit import VisionTransformer
from model_ae import AutoEncoder
from model_pt import PointTransformerONet
from model_mscale_deeponet import MscaleDeepONet
from model_hyperdeeponet import HyperDeepONet
from model_hyper_mscale_deeponet import HyperMscaleDeepONet
try:
    from model_hyper_mscale5_deeponet import HyperMscale5DeepONet
except ImportError:
    HyperMscale5DeepONet = None
try:
    from model_hyper_mscale5_learnable_scale_deeponet import HyperMscale5LearnableScaleDeepONet
except ImportError:
    HyperMscale5LearnableScaleDeepONet = None
try:
    from model_c_hyperdeeponet import c_HyperDeepONet
except ImportError:
    c_HyperDeepONet = None
try:
    from model_fusion_deeponet import Fusion_DeepONet
except ImportError:
    Fusion_DeepONet = None
try:
    from model_residual_fusion_deeponet import Residual_Fusion_DeepONet
except ImportError:
    Residual_Fusion_DeepONet = None
from data_loader import AirfoilDataset

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation Script for Task I: Airfoil Flow (+ scale factor stats)")
    parser.add_argument('--model', type=str, required=True,
                        choices=['deeponet', 'fno', 'unet', 'vit', 'ae', 'pt', 'mscale_deeponet', 'hyperdeeponet', 'c_hyperdeeponet', 'hyper_mscale_deeponet', 'hyper_mscale5_deeponet', 'hyper_mscale5_learnable_scale_deeponet', 'fusion_deeponet', 'residual_fusion_deeponet'],
                        help='Choose the baseline model to evaluate')
    parser.add_argument('--data_path', type=str, default='data/airfoil_unified_128x128.pt', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to weights (default: output/<model>/best_model.pth)')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save visualizations (default: output/<model>)')
    return parser.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Evaluation | Model: {args.model.upper()} | Device: {device}")

    # Create output directory
    if args.output_dir is None:
        args.output_dir = os.path.join('output', args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load test dataset
    data_mode = 'fno' if args.model in ['fno', 'unet', 'vit', 'ae'] else 'deeponet'
    dataset = AirfoilDataset(args.data_path, mode=data_mode)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # Must ensure the same manual_seed to maintain consistency between train and test splits
    _, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 2. Initialize model
    if args.model == 'fno':
        model = FNO2d(modes1=12, modes2=12, width=28, in_channels=3, out_channels=4)
    elif args.model == 'unet':
        model = FluidUNet(in_channels=3, out_channels=4, base_dim=20)
    elif args.model == 'vit':
        model = VisionTransformer(embed_dim=144, depth=4)
    elif args.model == 'ae':
        model = AutoEncoder(in_channels=3, out_channels=4, base_dim=24)
    elif args.model == 'deeponet':
        model = BoltzmannDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=256, num_outputs=4)
    elif args.model == 'pt':
        model = PointTransformerONet(hidden_dim=256, num_outputs=4)
    elif args.model == 'mscale_deeponet':
        model = MscaleDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=181, num_outputs=4,
                               scales=[1, 2, 4, 8, 16], depth=4, activation='GELU')
    elif args.model == 'hyperdeeponet':
        model = HyperDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=76, num_outputs=4,
                              trunk_depth=3, branch_depth=3, activation='GELU')
    elif args.model == 'c_hyperdeeponet':
        # 999,582 params (~1.00M budget), trunk [2,256,256,256,256,4], branch hidden 94, K=779
        model = c_HyperDeepONet(branch_dim=674, trunk_dim=2, trunk_hidden_dim=256, branch_hidden_dim=94,
                                num_outputs=4, trunk_depth=4, branch_depth=3, activation='GELU',
                                chunk_in=1024, chunk_out=256)
    elif args.model == 'hyper_mscale_deeponet':
        model = HyperMscaleDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=67, num_outputs=4,
                                    depth=4, activation='GELU')
    elif args.model == 'hyper_mscale5_deeponet':
        # 1,034,289 params (~1.03M budget), 5 hyper-generated trunk branches + fusion
        model = HyperMscale5DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=36, num_outputs=4,
                                     depth=4, n_scales=5, activation='GELU')
    elif args.model == 'hyper_mscale5_learnable_scale_deeponet':
        # 1,034,109 params (~1.03M budget); same 5-branch hyper trunk, but the
        # per-branch scale is a plain nn.Parameter (not branch-net output)
        model = HyperMscale5LearnableScaleDeepONet(branch_dim=674, trunk_dim=2, hidden_dim=36, num_outputs=4,
                                                   depth=4, n_scales=5, activation='GELU')
    elif args.model == 'fusion_deeponet':
        # 1,007,000 params (~1.01M budget), branch [674,253,...,253,1012], trunk [2,253,...,253,253]
        model = Fusion_DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=253, num_outputs=4,
                                depth=5, activation='GELU')
    elif args.model == 'residual_fusion_deeponet':
        # Same 1,007,000 params; branch-to-trunk gate uses 1+skip (residual)
        model = Residual_Fusion_DeepONet(branch_dim=674, trunk_dim=2, hidden_dim=253, num_outputs=4,
                                         depth=5, activation='GELU')

    model = model.to(device)

    # 3. Load model weights
    if args.checkpoint is None:
        args.checkpoint = os.path.join('output', args.model, 'best_model.pth')
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {ckpt_path}")

    # 3b. Capture the scale factor(s).
    # HyperMscaleDeepONet's branch net emits [trunk weights/biases | log_scale]
    # (1 trailing scalar); HyperMscale5DeepONet emits [... | 5 log-scales]
    # (one per trunk branch) — these vary per sample, so we hook the branch
    # net. HyperMscale5LearnableScaleDeepONet instead stores `log_scale` as a
    # plain nn.Parameter (NOT branch-net output), so it's identical for every
    # sample — no hook needed, just read it once.
    scale_records = []
    hook_handle = None
    global_log_scale = None
    n_scales = getattr(model, 'n_scales', 1)
    if isinstance(getattr(model, 'log_scale', None), nn.Parameter):
        global_log_scale = model.log_scale.detach().cpu()
    elif hasattr(model, 'branch_net') and hasattr(model, '_trunk_dims'):
        def _capture_scale(module, inputs, output):
            log_scale = output[:, -n_scales:]  # [B, n_scales]
            scale_records.append(torch.exp(log_scale).detach().cpu())
        hook_handle = model.branch_net.register_forward_hook(_capture_scale)
    else:
        print(f"[Note] '{args.model}' has no branch-net scale factor to report; skipping scale stats.")

    # 4. Calculate evaluation metrics and collect samples for visualization
    total_mae = 0.0
    total_mse = 0.0
    total_l2_error = 0.0
    criterion_mae = nn.L1Loss(reduction='sum')
    criterion_mse = nn.MSELoss(reduction='sum')

    # Store multiple samples for visualization
    samples_to_plot = []
    var_names = ['rho', 'u', 'v', 'T']

    # Get the actual geometry masks from dataset for coordinate-based models
    if data_mode != 'fno':
        # Load the full dataset to access geometry masks
        full_dataset = AirfoilDataset(args.data_path, mode='fno')  # Use 'fno' mode to get masks
        test_indices = test_loader.dataset.indices  # Get test set indices

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if data_mode == 'fno':
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
                mask = x[:, 0:1, :, :]
            else:
                x_branch, x_trunk, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                batch_size = x_branch.shape[0]

                # Process each sample individually for coordinate-based models
                pred_list = []
                for j in range(batch_size):
                    pred_j = model(x_branch[j:j+1], x_trunk[j])
                    pred_list.append(pred_j)
                pred = torch.cat(pred_list, dim=0)

                # Get the actual geometry mask for this sample
                actual_idx = test_indices[i]
                mask_2d = full_dataset.geo_mask[actual_idx].to(device)  # [1, 128, 128]
                # Reshape mask to match pred shape: [1, 16384, 1]
                mask = mask_2d.view(1, -1, 1)  # [1, 16384, 1]

            # Apply mask
            pred_masked = pred * mask
            y_masked = y * mask

            # Calculate Metrics
            total_mae += criterion_mae(pred_masked, y_masked).item()
            total_mse += criterion_mse(pred_masked, y_masked).item()

            # Relative L2 Error
            l2_err = torch.norm(pred_masked - y_masked, p=2) / (torch.norm(y_masked, p=2) + 1e-8)
            total_l2_error += l2_err.item()

            # Collect samples for visualization
            if i < args.num_samples:
                if data_mode == 'fno':
                    # Shape: [1, 4, 128, 128]
                    gt_sample = y_masked[0].cpu().numpy()  # [4, 128, 128]
                    pred_sample = pred_masked[0].cpu().numpy()  # [4, 128, 128]
                    mask_sample = mask[0, 0].cpu().numpy()  # [128, 128]
                else:
                    # Shape: [1, 16384, 4] -> reshape to [4, 128, 128]
                    gt_sample = y_masked[0].view(128, 128, 4).permute(2, 0, 1).cpu().numpy()
                    pred_sample = pred_masked[0].view(128, 128, 4).permute(2, 0, 1).cpu().numpy()
                    # Get the actual 2D mask - mask_2d shape is [1, 128, 128]
                    mask_sample = mask_2d[0].cpu().numpy()  # [128, 128]

                samples_to_plot.append({
                    'gt': gt_sample,
                    'pred': pred_sample,
                    'mask': mask_sample,
                    'index': i
                })

    if hook_handle is not None:
        hook_handle.remove()

    num_samples = len(test_loader.dataset)
    num_elements = num_samples * np.prod(y_masked.shape[1:])  # Total number of elements

    final_mae = total_mae / num_elements
    final_mse = total_mse / num_elements
    final_rel_l2 = total_l2_error / num_samples

    print("-" * 50)
    print(f"Final Results for {args.model.upper()}:")
    print(f"Mean Absolute Error (MAE) : {final_mae:.4g}")
    print(f"Mean Squared Error (MSE)  : {final_mse:.4g}")
    print(f"Relative L2 Error         : {final_rel_l2:.4g}")
    print("-" * 50)

    # 4b. Scale factor statistics
    if global_log_scale is not None:
        # Plain learnable scale: one fixed value per branch, identical for every sample.
        # mean/min/max all collapse to the same value and std=0 here (by construction,
        # since it doesn't vary across the test set) — printed in the same
        # mean/std/min/max format as the per-sample (branch-net-conditioned) case
        # above for direct comparison.
        scales = torch.exp(global_log_scale)  # [n_scales]
        print(f"Scale Factors — {n_scales} trunk branches, plain learnable parameter "
              f"(identical for every test sample, no branch-net conditioning):")
        for k in range(n_scales):
            v = scales[k].item()
            print(f"  Branch {k}: mean={v:.6f}  std=0.000000  min={v:.6f}  max={v:.6f}")
        print(f"  All {n_scales} branches pooled: mean={scales.mean().item():.6f}  "
              f"std={scales.std().item():.6f}  min={scales.min().item():.6f}  max={scales.max().item():.6f}")
        print("-" * 50)
    elif scale_records:
        scale_all = torch.cat(scale_records, dim=0)  # [N_test, n_scales]
        n_test = scale_all.shape[0]
        if n_scales == 1:
            s = scale_all.squeeze(-1)  # [N_test]
            print(f"Scale Factor over {n_test} test samples (varies sample to sample):")
            print(f"  Mean : {s.mean().item():.6f}")
            print(f"  Std  : {s.std().item():.6f}")
            print(f"  Min  : {s.min().item():.6f}")
            print(f"  Max  : {s.max().item():.6f}")
        else:
            print(f"Scale Factors over {n_test} test samples ({n_scales} trunk branches, "
                  f"each sample has its own value per branch):")
            for k in range(n_scales):
                s_k = scale_all[:, k]
                print(f"  Branch {k}: mean={s_k.mean().item():.6f}  std={s_k.std().item():.6f}  "
                      f"min={s_k.min().item():.6f}  max={s_k.max().item():.6f}")
            s_flat = scale_all.reshape(-1)
            print(f"  All {n_scales} branches pooled: mean={s_flat.mean().item():.6f}  "
                  f"std={s_flat.std().item():.6f}  min={s_flat.min().item():.6f}  max={s_flat.max().item():.6f}")
        print("-" * 50)

    # 5. Generate visualizations for multiple samples and variables
    print(f"\nGenerating visualizations for {len(samples_to_plot)} samples...")

    for sample_data in samples_to_plot:
        sample_idx = sample_data['index']
        gt = sample_data['gt']  # [4, 128, 128]
        pred = sample_data['pred']  # [4, 128, 128]
        mask = sample_data['mask']  # [128, 128]

        # Create a figure for all 4 variables
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))

        for var_idx, var_name in enumerate(var_names):
            gt_var = gt[var_idx]
            pred_var = pred[var_idx]
            err_var = np.abs(gt_var - pred_var)

            # Apply mask for visualization
            gt_var_masked = np.ma.masked_where(mask < 0.5, gt_var)
            pred_var_masked = np.ma.masked_where(mask < 0.5, pred_var)
            err_var_masked = np.ma.masked_where(mask < 0.5, err_var)

            # Determine common color scale for GT and Pred
            vmin = min(gt_var_masked.min(), pred_var_masked.min())
            vmax = max(gt_var_masked.max(), pred_var_masked.max())

            # Ground Truth
            im0 = axes[var_idx, 0].imshow(gt_var_masked, cmap='jet', vmin=vmin, vmax=vmax)
            axes[var_idx, 0].set_title(f"GT - {var_name}", fontsize=12)
            axes[var_idx, 0].axis('off')
            plt.colorbar(im0, ax=axes[var_idx, 0], fraction=0.046, pad=0.04)

            # Prediction
            im1 = axes[var_idx, 1].imshow(pred_var_masked, cmap='jet', vmin=vmin, vmax=vmax)
            axes[var_idx, 1].set_title(f"Pred - {var_name}", fontsize=12)
            axes[var_idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[var_idx, 1], fraction=0.046, pad=0.04)

            # Error
            im2 = axes[var_idx, 2].imshow(err_var_masked, cmap='hot')
            axes[var_idx, 2].set_title(f"Error - {var_name}", fontsize=12)
            axes[var_idx, 2].axis('off')
            plt.colorbar(im2, ax=axes[var_idx, 2], fraction=0.046, pad=0.04)

        plt.suptitle(f"{args.model.upper()} - Sample {sample_idx}", fontsize=16, y=0.995)
        plt.tight_layout()

        save_fig_path = os.path.join(args.output_dir, f"sample_{sample_idx}.png")
        plt.savefig(save_fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_fig_path}")

    print(f"\nAll visualizations saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
