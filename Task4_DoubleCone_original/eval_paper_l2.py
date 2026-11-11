"""
Paper Eq.(17) relative-L2 evaluator for Task IV -- NOT part of upstream.

U(x,t) in the paper is the full vector of macroscopic variables (Eq.3/Eq.9),
so Eq.17's ||.||_2 is a vector norm taken ACROSS CHANNELS at each point, and
the outer sum pools over every point (here: every test-set sample x every
grid cell, no geometry mask exists for this task -- confirmed: double_cone.pt
carries no mask field, unlike the airfoil/cylinder tasks).

This is computed in raw physical units (after the log10 inverse-transform on
the pressure channel), exactly as specified. That has a real consequence
worth knowing before trusting a single number: rho, u, v, p differ by orders
of magnitude in this dataset (rho ~1e1, v ~1e5, u/p ~1e6-1e7 in combined
L2 norm over the test set), so a *combined* vector norm is overwhelmingly
dominated by whichever channel has the largest raw magnitude -- here,
pressure and u -- while rho's contribution is negligible regardless of how
good or bad the rho prediction is. This script reports:
  1. the single pooled Eq.17 number (what Table 7 reports), and
  2. the per-channel breakdown + each channel's share of the denominator norm,
so the dominance is visible rather than hidden behind one aggregate figure.
"""
import argparse
import torch

from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_unet import FluidUNet


def get_args():
    p = argparse.ArgumentParser(description="Paper Eq.17 pooled relative-L2 for Task4 models")
    p.add_argument('--model', type=str, required=True, choices=['deeponet', 'fno', 'unet'])
    p.add_argument('--data_path', type=str, default='./double_cone.pt')
    p.add_argument('--checkpoint', type=str, default='./checkpoints_paper_protocol/best_model_{}_nofourier.pth')
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


def build_model(name, use_fourier):
    if name == 'deeponet':
        return DeepONet2d(in_channels=5, out_channels=4, basis_size=256, use_fourier=use_fourier)
    elif name == 'fno':
        return FNO2d(modes1=8, modes2=64, width=90, in_channels=5, out_channels=4, use_fourier=use_fourier)
    elif name == 'unet':
        return FluidUNet(in_channels=5, out_channels=4, features=64, use_fourier=use_fourier)


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.checkpoint.format(args.model)
    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"Loading data from {args.data_path}...")

    data_full = torch.load(args.data_path, weights_only=False)
    x_data = data_full['x'].float()
    y_data = data_full['y'].float()  # physical space GT, pressure NOT log yet

    y_data_log = y_data.clone()
    y_data_log[:, 3, :, :] = torch.log10(y_data[:, 3, :, :] + 1e-6)

    x_mean = torch.mean(x_data, dim=(0, 2, 3), keepdim=True).to(device)
    x_std = torch.std(x_data, dim=(0, 2, 3), keepdim=True).to(device)
    y_mean = torch.mean(y_data_log, dim=(0, 2, 3), keepdim=True).to(device)
    y_std = torch.std(y_data_log, dim=(0, 2, 3), keepdim=True).to(device)

    class GN:
        def __init__(self, mean, std):
            self.mean, self.std, self.eps = mean, std, 1e-6
        def encode(self, x): return (x - self.mean) / (self.std + self.eps)
        def decode(self, x): return x * (self.std + self.eps) + self.mean

    x_norm = GN(x_mean, x_std)
    y_norm = GN(y_mean, y_std)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', {})
    use_fourier = cfg.get('use_fourier', False)

    model = build_model(args.model, use_fourier).to(device)
    model.load_state_dict(checkpoint.get('model_state', checkpoint))
    model.eval()

    # same train/test split (seed=42) as data_loader.py
    n_total = x_data.shape[0]
    n_train = 45 if n_total == 51 else (28 if n_total == 32 else n_total - max(1, int(n_total * 0.1)))
    g_cpu = torch.Generator(); g_cpu.manual_seed(42)
    test_idx = sorted(torch.randperm(n_total, generator=g_cpu)[n_train:].tolist())

    ch_names = ['rho', 'u', 'v', 'p']

    # pooled across every (sample, spatial-point) pair -- no mask exists for this task
    err_sq_sum = 0.0             # sum ||pred - gt||_2^2 over channels+points+samples
    gt_sq_sum = 0.0
    err_sq_per_ch = torch.zeros(4)   # per-channel sum of squared error (for the breakdown)
    gt_sq_per_ch = torch.zeros(4)    # per-channel sum of squared GT magnitude

    with torch.no_grad():
        for i in test_idx:
            xin = x_data[i].unsqueeze(0).to(device)
            gt_phys = y_data[i].unsqueeze(0).to(device)

            x_enc = x_norm.encode(xin)
            out_enc = model(x_enc)
            pred_log = y_norm.decode(out_enc)
            pred_phys = pred_log.clone()
            pred_phys[:, 3, :, :] = torch.pow(10, pred_log[:, 3, :, :])

            diff = pred_phys - gt_phys  # [1,4,17,384]

            err_sq_sum += diff.pow(2).sum().item()
            gt_sq_sum += gt_phys.pow(2).sum().item()

            err_sq_per_ch += diff.pow(2).sum(dim=(0, 2, 3)).cpu()
            gt_sq_per_ch += gt_phys.pow(2).sum(dim=(0, 2, 3)).cpu()

    pooled_rel_l2 = (err_sq_sum ** 0.5) / (gt_sq_sum ** 0.5 + 1e-12)
    per_ch_rel_l2 = (err_sq_per_ch.sqrt()) / (gt_sq_per_ch.sqrt() + 1e-12)
    gt_norm_share = gt_sq_per_ch / gt_sq_per_ch.sum()

    print(f"\n===== {args.model.upper()} | Test set ({len(test_idx)} samples, idx={test_idx}) =====")
    print(f"[paper Eq.17] pooled physical-space Relative L2 (all channels+points+samples combined): {pooled_rel_l2:.5f}")
    print("\n-- per-channel breakdown (for transparency; NOT what Eq.17 reports on its own) --")
    for c, name in enumerate(ch_names):
        print(f"  {name:>3s}: rel_L2={per_ch_rel_l2[c].item():.4%}   "
              f"(share of pooled denominator norm^2: {gt_norm_share[c].item():.2%})")
    print(f"  unweighted per-channel mean: {per_ch_rel_l2.mean().item():.4%}  (a different, non-Eq.17 metric)")


if __name__ == "__main__":
    main()
