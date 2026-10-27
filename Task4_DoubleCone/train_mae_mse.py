import torch
import torch.nn as nn
import time
import os
import argparse
from data_loader import get_dataloader_and_stats

from model_deeponet import DeepONet2d
from model_fno import FNO2d
from model_unet import FluidUNet

# Same as train.py, but logs BOTH MAE(L1) and MSE(L2) every logged epoch
# (weighted train / unweighted test), and records them for later reuse.

def get_args():
    parser = argparse.ArgumentParser(description="Golden Protocol Training Script w/ MAE+MSE logging")
    parser.add_argument('--model', type=str, required=True, choices=['deeponet', 'fno', 'unet'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--no_fourier', action='store_true', help='Disable Fourier Encoding')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_2500ep_nofourier_metrics')
    return parser.parse_args()


def build_model(model_name, use_fourier):
    if model_name == 'deeponet':
        return DeepONet2d(in_channels=5, out_channels=4, basis_size=256, use_fourier=use_fourier)
    elif model_name == 'fno':
        return FNO2d(modes1=8, modes2=64, width=64, in_channels=5, out_channels=4, use_fourier=use_fourier)
    elif model_name == 'unet':
        return FluidUNet(in_channels=5, out_channels=4, features=64, use_fourier=use_fourier)


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fourier = not args.no_fourier
    fourier_suffix = "_fourier" if use_fourier else "_nofourier"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, f'best_model_{args.model}{fourier_suffix}.pth')
    log_file = os.path.join(args.save_dir, f'train_{args.model}{fourier_suffix}.log')
    metrics_csv = os.path.join(args.save_dir, f'metrics_{args.model}{fourier_suffix}.csv')

    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log(f"=== Training {args.model.upper()} | Fourier: {use_fourier} | Device: {device} ===")
    log("Strategy: TIME-DILATED GOLDEN PROTOCOL (Peak LR @ 40%, Weights start @ Ep800)")
    log("Logging BOTH MAE(L1) and MSE(L2), normalized/encoded learning space, weighted-train / unweighted-test")

    train_loader, test_loader, x_norm, y_norm, _vac_unused = get_dataloader_and_stats(args.data_path, args.batch_size, device)  # NOTE: legacy script, not updated for vacuum masking/MSE loss -- see UPSTREAM_CHANGES_2026-10-25.md

    model = build_model(args.model, use_fourier).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters: {total_params:,}")
    log(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.4, anneal_strategy='cos'
    )

    mae_fn = nn.L1Loss(reduction='none')   # training loss (weighted) uses this
    mse_fn = nn.MSELoss(reduction='none')  # logged in parallel, not used for optimization

    best_test_mae = float('inf')
    best_test_epoch = -1
    best_test_mse_at_best = None

    with open(metrics_csv, 'w') as f:
        f.write("epoch,lr,w_wall,train_mae,train_mse,test_mae,test_mse\n")

    for ep in range(args.epochs):
        model.train()
        train_mae_val = 0.0
        train_mse_val = 0.0

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

            raw_mae = mae_fn(out_enc, y_enc)
            w = torch.ones_like(raw_mae)
            w[:, :, 2, :] = w_wall
            w[:, :, 1, :] = w_near
            w[:, :, 3, :] = w_near
            w[:, 3, :, :] *= w_p

            loss = (raw_mae * w).mean()   # optimize on weighted MAE, same as train.py
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                train_mae_val += loss.item()
                raw_mse = mse_fn(out_enc, y_enc)
                train_mse_val += (raw_mse * w).mean().item()

        train_mae_val /= len(train_loader)
        train_mse_val /= len(train_loader)

        model.eval()
        test_mae_val = 0.0
        test_mse_val = 0.0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_enc_test = x_norm.encode(x_test)
                y_enc_test = y_norm.encode(y_test)
                out_enc_test = model(x_enc_test)

                raw_test_mae = mae_fn(out_enc_test, y_enc_test)
                raw_test_mse = mse_fn(out_enc_test, y_enc_test)
                test_mae_val += raw_test_mae.mean().item()
                test_mse_val += raw_test_mse.mean().item()

        test_mae_val /= len(test_loader)
        test_mse_val /= len(test_loader)

        with open(metrics_csv, 'a') as f:
            f.write(f"{ep},{scheduler.get_last_lr()[0]:.6e},{w_wall:.2f},"
                    f"{train_mae_val:.6f},{train_mse_val:.6f},{test_mae_val:.6f},{test_mse_val:.6f}\n")

        if ep % 50 == 0 or ep == args.epochs - 1:
            log(f"Ep {ep:08d} | LR: {scheduler.get_last_lr()[0]:.1e} | "
                f"Train MAE: {train_mae_val:.5f} MSE: {train_mse_val:.5f} | "
                f"Test MAE: {test_mae_val:.5f} MSE: {test_mse_val:.5f} | W_wall: {w_wall:.1f}")

        if ep >= 1000 and test_mae_val < best_test_mae:
            best_test_mae = test_mae_val
            best_test_mse_at_best = test_mse_val
            best_test_epoch = ep

            torch.save({
                'model_state': model.state_dict(),
                'config': {'use_fourier': use_fourier},
                'epoch': ep,
                'best_test_mae': best_test_mae,
                'best_test_mse': best_test_mse_at_best,
            }, save_path)

            if ep % 50 == 0:
                log(f"  >>> [SAVED] Epoch {ep:04d} | Best Test MAE: {best_test_mae:.6f} | Test MSE: {best_test_mse_at_best:.6f} <<<")

    log(f"Finished. Best Test MAE: {best_test_mae:.6f} | Test MSE @ best: {best_test_mse_at_best:.6f} discovered at Epoch {best_test_epoch}")


if __name__ == "__main__":
    main()
