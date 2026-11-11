import argparse
import torch
import torch.nn as nn

from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_unet import FluidUNet


def get_args():
    p = argparse.ArgumentParser(description="Normalized-space MAE/MSE + physical-space L2 for Task4 models")
    p.add_argument('--model', type=str, required=True, choices=['deeponet', 'fno', 'unet'])
    p.add_argument('--data_path', type=str, default='./double_cone.pt')
    p.add_argument('--checkpoint', type=str, default='./checkpoints_2500ep_nofourier/best_model_{}_nofourier.pth')
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


def build_model(name, use_fourier):
    if name == 'deeponet':
        return DeepONet2d(in_channels=5, out_channels=4, basis_size=256, use_fourier=use_fourier)
    elif name == 'fno':
        return FNO2d(modes1=8, modes2=64, width=64, in_channels=5, out_channels=4, use_fourier=use_fourier)
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
    y_data = data_full['y'].float()  # physical space GT (pressure NOT log yet)

    # Same log10 transform on pressure channel as data_loader.py / train.py
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

    # Same train/test split (seed=42) as data_loader.py / eval.py
    n_total = x_data.shape[0]
    if n_total == 51:
        n_train = 45
    elif n_total == 32:
        n_train = 28
    else:
        n_train = n_total - max(1, int(n_total * 0.1))
    g_cpu = torch.Generator(); g_cpu.manual_seed(42)
    test_idx = torch.randperm(n_total, generator=g_cpu)[n_train:].tolist()

    ch_names = ['rho', 'u', 'v', 'p']
    mae_fn = nn.L1Loss(reduction='none')
    mse_fn = nn.MSELoss(reduction='none')

    mae_norm = torch.zeros(len(ch_names))
    mse_norm = torch.zeros(len(ch_names))
    rel_l2_phys = torch.zeros(len(ch_names))

    with torch.no_grad():
        for i in test_idx:
            xin = x_data[i].unsqueeze(0).to(device)
            gt_log = y_data_log[i].unsqueeze(0).to(device)     # normalized-space target (log10 pressure, un-normalized units)
            gt_phys = y_data[i].unsqueeze(0).to(device)         # physical-space target

            x_enc = x_norm.encode(xin)
            y_enc_gt = y_norm.encode(gt_log)
            out_enc = model(x_enc)

            # --- normalized (encoded) learning space MAE/MSE ---
            mae_norm += mae_fn(out_enc, y_enc_gt).mean(dim=(0, 2, 3)).cpu()
            mse_norm += mse_fn(out_enc, y_enc_gt).mean(dim=(0, 2, 3)).cpu()

            # --- physical space relative L2 (same as eval.py) ---
            pred_log = y_norm.decode(out_enc)
            pred_phys = pred_log.clone()
            pred_phys[:, 3, :, :] = torch.pow(10, pred_log[:, 3, :, :])
            for c in range(len(ch_names)):
                num = torch.linalg.norm(pred_phys[0, c] - gt_phys[0, c])
                den = torch.linalg.norm(gt_phys[0, c]) + 1e-8
                rel_l2_phys[c] += (num / den).item()

    n = len(test_idx)
    mae_norm /= n
    mse_norm /= n
    rel_l2_phys /= n

    print(f"\n===== {args.model.upper()} (nofourier={not use_fourier is False}) | Test set ({n} samples, idx={sorted(test_idx)}) =====")
    print("-- Normalized (encoded) learning-space error --")
    for c, name in enumerate(ch_names):
        print(f"  {name:>3s}: MAE={mae_norm[c].item():.5f}  MSE={mse_norm[c].item():.5f}")
    print(f"  mean: MAE={mae_norm.mean().item():.5f}  MSE={mse_norm.mean().item():.5f}")

    print("-- Physical-space Relative L2 Error --")
    for c, name in enumerate(ch_names):
        tag = "   <-- README Pressure Field Rel L2" if name == 'p' else ""
        print(f"  {name:>3s}: {rel_l2_phys[c].item():.4%}{tag}")
    print(f"  per-channel mean: {rel_l2_phys.mean().item():.4%}")


if __name__ == "__main__":
    main()
