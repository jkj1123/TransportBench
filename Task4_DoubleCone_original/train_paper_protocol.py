"""
Paper-protocol training script for Task IV (Double-Cone) -- NOT part of upstream.

Reproduces the settings described in the TransportBench paper (Sec. 5.2/5.3),
which differ from the public repo's train.py in three ways:
  1. Optimizer: Adam (paper) instead of AdamW (repo).
  2. Training/eval loss: masked MSE (paper Eq.15), not L1/MAE (repo).
  3. Parameter budget: ~33M for every model (paper Sec 5.2). DeepONet (27.8M)
     and U-Net (29.7M) are already close with their repo-default configs;
     FNO's repo default (width=64) is only ~16.8M, so width is raised to 90
     here (~33.2M) to match the stated budget. Nothing else about FNO changes.

Everything else -- OneCycleLR warmup (pct_start=0.4), the epoch-800 spatial
curriculum that upweights the wall/near-wall rows and the pressure channel,
the epoch>=1000 checkpoint-saving guard, batch size 8, lr 5e-4, 2500 epochs,
45/6 train-test split -- is unchanged from train.py / the paper's stated
protocol (Adam optimizer, 2500 epochs, curriculum + OneCycleLR "applied
consistently to all baseline models in Task IV").
"""
import torch
import torch.nn as nn
import os
import argparse
from data_loader import get_dataloader_and_stats

from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_unet import FluidUNet


def get_args():
    parser = argparse.ArgumentParser(description="Paper-protocol training (Adam + masked MSE, ~33M params)")
    parser.add_argument('--model', type=str, required=True, choices=['deeponet', 'fno', 'unet'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--no_fourier', action='store_true', help='Disable Fourier Encoding')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_paper_protocol')
    return parser.parse_args()


def build_model(model_name, use_fourier):
    if model_name == 'deeponet':
        return DeepONet2d(in_channels=5, out_channels=4, basis_size=256, use_fourier=use_fourier)
    elif model_name == 'fno':
        # width=90 -> ~33.2M params, matching the paper's ~33M budget for Task IV
        # (repo default width=64 is only ~16.8M).
        return FNO2d(modes1=8, modes2=64, width=90, in_channels=5, out_channels=4, use_fourier=use_fourier)
    elif model_name == 'unet':
        return FluidUNet(in_channels=5, out_channels=4, features=64, use_fourier=use_fourier)


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fourier = not args.no_fourier
    fourier_suffix = "_fourier" if use_fourier else "_nofourier"

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'best_model_{args.model}{fourier_suffix}.pth')
    log_file = os.path.join(args.save_dir, f'train_{args.model}{fourier_suffix}.log')
    metrics_csv = os.path.join(args.save_dir, f'metrics_{args.model}{fourier_suffix}.csv')

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log(f"=== [PAPER PROTOCOL] Training {args.model.upper()} | Fourier: {use_fourier} | Device: {device} ===")
    log("Optimizer: Adam | Loss: masked MSE (Eq.15) | Scheduler: OneCycleLR (pct_start=0.4)")
    log("Curriculum: uniform weights for Ep<800, then wall/near-wall/pressure upweighting")

    train_loader, test_loader, x_norm, y_norm = get_dataloader_and_stats(args.data_path, args.batch_size, device)

    model = build_model(args.model, use_fourier).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.4, anneal_strategy='cos'
    )

    mse_fn = nn.MSELoss(reduction='none')
    mae_fn = nn.L1Loss(reduction='none')  # logged alongside for comparison with Table 7's MAE column

    best_test_mse = float('inf')
    best_test_epoch = -1
    best_test_mae_at_best = None

    with open(metrics_csv, 'w') as f:
        f.write("epoch,lr,w_wall,train_mse,train_mae,test_mse,test_mae\n")

    for ep in range(args.epochs):
        model.train()
        train_mse_val = 0.0
        train_mae_val = 0.0

        if ep < 800:
            w_wall, w_near, w_p = 1.0, 1.0, 1.0
        else:
            progress = min(1.0, (ep - 800) / 1000.0)
            w_wall = 1.0 + 4.0 * progress
            w_near = 1.0 + 1.0 * progress
            w_p = 1.0 + 1.0 * progress

        for x, y in train_loader:
            x_enc = x_norm.encode(x)
            y_enc = y_norm.encode(y)

            optimizer.zero_grad()
            out_enc = model(x_enc)

            raw_mse = mse_fn(out_enc, y_enc)
            w = torch.ones_like(raw_mse)
            w[:, :, 2, :] = w_wall
            w[:, :, 1, :] = w_near
            w[:, :, 3, :] = w_near
            w[:, 3, :, :] *= w_p

            loss = (raw_mse * w).mean()  # weighted MSE -- the paper's stated training objective
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                train_mse_val += loss.item()
                raw_mae = mae_fn(out_enc, y_enc)
                train_mae_val += (raw_mae * w).mean().item()

        train_mse_val /= len(train_loader)
        train_mae_val /= len(train_loader)

        model.eval()
        test_mse_val = 0.0
        test_mae_val = 0.0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_enc_test = x_norm.encode(x_test)
                y_enc_test = y_norm.encode(y_test)
                out_enc_test = model(x_enc_test)

                raw_test_mse = mse_fn(out_enc_test, y_enc_test)
                raw_test_mae = mae_fn(out_enc_test, y_enc_test)
                test_mse_val += raw_test_mse.mean().item()
                test_mae_val += raw_test_mae.mean().item()

        test_mse_val /= len(test_loader)
        test_mae_val /= len(test_loader)

        with open(metrics_csv, 'a') as f:
            f.write(f"{ep},{scheduler.get_last_lr()[0]:.6e},{w_wall:.2f},"
                    f"{train_mse_val:.6f},{train_mae_val:.6f},{test_mse_val:.6f},{test_mae_val:.6f}\n")

        if ep % 50 == 0 or ep == args.epochs - 1:
            log(f"Ep {ep:04d} | LR: {scheduler.get_last_lr()[0]:.1e} | "
                f"Train MSE: {train_mse_val:.5f} MAE: {train_mae_val:.5f} | "
                f"Test MSE: {test_mse_val:.5f} MAE: {test_mae_val:.5f} | W_wall: {w_wall:.1f}")

        if ep >= 1000 and test_mse_val < best_test_mse:
            best_test_mse = test_mse_val
            best_test_mae_at_best = test_mae_val
            best_test_epoch = ep

            torch.save({
                'model_state': model.state_dict(),
                'config': {'use_fourier': use_fourier},
                'epoch': ep,
                'best_test_mse': best_test_mse,
                'best_test_mae': best_test_mae_at_best,
            }, save_path)

            if ep % 50 == 0:
                log(f"  >>> [SAVED] Epoch {ep:04d} | Best Test MSE: {best_test_mse:.6f} | MAE: {best_test_mae_at_best:.6f} <<<")

    mae_str = f"{best_test_mae_at_best:.6f}" if best_test_mae_at_best is not None else "N/A"
    log(f"Finished. Best Test MSE: {best_test_mse:.6f} | MAE @ best: {mae_str} discovered at Epoch {best_test_epoch}")


if __name__ == "__main__":
    main()
