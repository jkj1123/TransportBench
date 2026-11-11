"""
Train-set vs. test-set metrics for the paper-protocol checkpoints -- NOT part
of upstream.

Computes, with the trained model frozen (eval mode, no grad) and using the
SAME 45/6 seed=42 split as training:
  - unweighted MSE / MAE in the normalized (encoded) learning space, matching
    how train_paper_protocol.py reports its own "Test MSE/MAE" line (but here
    computed once, cleanly, on a fixed final checkpoint rather than read off
    per-epoch training-time logs).
  - the paper's Eq.17 pooled physical-space relative L2 (all channels+points
    combined, same pooling as eval_paper_l2.py), plus the per-channel
    breakdown for transparency.

Run once per split per model; nothing here mutates the checkpoints.
"""
import argparse
import torch
import torch.nn as nn

from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_unet import FluidUNet


def get_args():
    p = argparse.ArgumentParser(description="Train vs test MSE/MAE/rel-L2 for Task4 paper-protocol checkpoints")
    p.add_argument('--model', type=str, required=True, choices=['deeponet', 'fno', 'unet'])
    p.add_argument('--split', type=str, required=True, choices=['train', 'test'])
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

    # same train/test split (seed=42) as data_loader.py / train_paper_protocol.py
    n_total = x_data.shape[0]
    n_train = 45 if n_total == 51 else (28 if n_total == 32 else n_total - max(1, int(n_total * 0.1)))
    g_cpu = torch.Generator(); g_cpu.manual_seed(42)
    perm = torch.randperm(n_total, generator=g_cpu)
    idx = sorted(perm[:n_train].tolist()) if args.split == 'train' else sorted(perm[n_train:].tolist())

    ch_names = ['rho', 'u', 'v', 'p']
    mae_fn = nn.L1Loss(reduction='none')
    mse_fn = nn.MSELoss(reduction='none')

    enc_mae_sum, enc_mse_sum = 0.0, 0.0
    err_sq_sum, gt_sq_sum = 0.0, 0.0
    err_sq_per_ch = torch.zeros(4)
    gt_sq_per_ch = torch.zeros(4)

    with torch.no_grad():
        for i in idx:
            xin = x_data[i].unsqueeze(0).to(device)
            gt_log = y_data_log[i].unsqueeze(0).to(device)
            gt_phys = y_data[i].unsqueeze(0).to(device)

            x_enc = x_norm.encode(xin)
            y_enc_gt = y_norm.encode(gt_log)
            out_enc = model(x_enc)

            enc_mae_sum += mae_fn(out_enc, y_enc_gt).mean().item()
            enc_mse_sum += mse_fn(out_enc, y_enc_gt).mean().item()

            pred_log = y_norm.decode(out_enc)
            pred_phys = pred_log.clone()
            pred_phys[:, 3, :, :] = torch.pow(10, pred_log[:, 3, :, :])

            diff = pred_phys - gt_phys
            err_sq_sum += diff.pow(2).sum().item()
            gt_sq_sum += gt_phys.pow(2).sum().item()
            err_sq_per_ch += diff.pow(2).sum(dim=(0, 2, 3)).cpu()
            gt_sq_per_ch += gt_phys.pow(2).sum(dim=(0, 2, 3)).cpu()

    n = len(idx)
    enc_mae = enc_mae_sum / n
    enc_mse = enc_mse_sum / n
    pooled_rel_l2 = (err_sq_sum ** 0.5) / (gt_sq_sum ** 0.5 + 1e-12)
    per_ch_rel_l2 = (err_sq_per_ch.sqrt()) / (gt_sq_per_ch.sqrt() + 1e-12)
    gt_norm_share = gt_sq_per_ch / gt_sq_per_ch.sum()

    print(f"\n===== {args.model.upper()} | split={args.split} ({n} samples, idx={idx}) =====")
    print(f"Encoded-space (normalized) unweighted MAE: {enc_mae:.5f}  MSE: {enc_mse:.5f}")
    print(f"[paper Eq.17] pooled physical-space Relative L2: {pooled_rel_l2:.5f}")
    print("-- per-channel breakdown --")
    for c, name in enumerate(ch_names):
        print(f"  {name:>3s}: rel_L2={per_ch_rel_l2[c].item():.4%}  (denom norm^2 share: {gt_norm_share[c].item():.2%})")
    print(f"  unweighted per-channel mean: {per_ch_rel_l2.mean().item():.4%}")


if __name__ == "__main__":
    main()
