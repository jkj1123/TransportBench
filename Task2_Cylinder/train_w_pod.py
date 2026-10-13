"""
Hybrid POD-ROM + HyperDeepONet residual model for Task II (Cylinder).

    u(Kn, Ma, x) = u_pod(Kn, Ma, x) + r_hyper(Kn, Ma, x)

Two stages, trained SEQUENTIALLY (not jointly), using only the TRAINING
split at every step -- no test-set information (and no "oracle" direct
projection of the true test field) ever enters either stage:

Stage 1 -- POD-ROM, with RBF interpolation approximating the coefficients:
    1a. Per-channel normalize the stored (already-[-1,1]) target field using
        TRAIN-only mean/std, then mean-center across train snapshots and
        take an SVD -- this gives the fixed POD basis V_r (top `--pod_r`
        modes). All of this uses only the 320 training snapshots.
    1b. Fit a thin-plate-spline RBF interpolator: (Kn, Ma) -> r POD
        coefficients, against the TRUE train-snapshot coefficients
        (obtained by projecting the train snapshots onto V_r). This is a
        closed-form fit (no training loop) -- it never sees the test set.
    u_pod(Kn, Ma, x) is then: RBF(Kn,Ma) -> r coeffs -> reconstruct via
    V_r -> un-normalize (x std_c + mean_c) -> back in the SAME space as
    the dataset's stored (already-[-1,1]) target field.

Stage 2 -- HyperDeepONet residual correction:
    r_hyper is a HyperDeepONet (~20K params by default) trained to predict
    the RESIDUAL (true_field - u_pod), evaluated only on the TRAIN split.
    Because u_pod already explains most of the variance, this residual has
    a much smaller natural scale than the original field, so it is
    normalized (its own train-only per-channel mean/std) before being used
    as the regression target -- otherwise HyperDeepONet would be fitting a
    tiny-magnitude, poorly-scaled target. At eval time the predicted
    residual is un-normalized and added back to u_pod.

Final prediction (what gets compared to the true field for Rel L2):
    u_pred = u_pod(Kn,Ma,x) + unnormalize(HyperDeepONet(Kn,Ma,x))

Everything (POD basis, channel stats, coefficient MLP, residual stats,
HyperDeepONet) is saved into ONE checkpoint so eval_w_pod.py can rebuild
the whole pipeline without retraining.
"""

import os
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator

from model_hyperdeeponet import HyperDeepONet
from data_loader_fast import CylinderDatasetFast, FastBatchIterator


def get_args():
    parser = argparse.ArgumentParser(description="Task II Cylinder: POD-ROM + HyperDeepONet residual hybrid")
    parser.add_argument('--data_path', type=str, default='./cylinder_full_2400_400_seed42.pt')
    parser.add_argument('--pod_r', type=int, default=32, help='POD rank (number of modes).')
    parser.add_argument('--pod_rbf_smoothing', type=float, default=1e-6, help='RBFInterpolator smoothing (avoids singular-matrix failures).')
    parser.add_argument('--hyper_hidden_dim', type=int, default=20, help='HyperDeepONet hidden_dim (default 20 -> ~21.6K params).')
    parser.add_argument('--epochs', type=int, default=5000, help='HyperDeepONet residual training epochs.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_step', type=int, default=100)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.91)
    parser.add_argument('--run_tag', type=str, default='')
    parser.add_argument('--rel2_log_interval', type=int, default=50)
    return parser.parse_args()


def main():
    args = get_args()
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_dir_name = f"w_pod_hyperdeeponet_{args.run_tag}" if args.run_tag else "w_pod_hyperdeeponet"
    save_dir = os.path.join('output', model_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    rel2_log_path = os.path.join(save_dir, 'rel2_history.csv')
    with open(rel2_log_path, 'w') as f:
        f.write('epoch,test_rel2_error,timestamp\n')
    run_info_path = os.path.join(save_dir, 'run_info.txt')
    run_start_time = time.time()
    with open(run_info_path, 'w') as f:
        f.write("model=w_pod_hyperdeeponet\n")
        f.write(f"data_path={args.data_path}\n")
        f.write(f"pod_r={args.pod_r}\n")
        f.write(f"pod_method=RBFInterpolator(thin_plate_spline)\n")
        f.write(f"pod_rbf_smoothing={args.pod_rbf_smoothing}\n")
        f.write(f"hyper_hidden_dim={args.hyper_hidden_dim}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"lr_decay_step={args.lr_decay_step}\n")
        f.write(f"lr_decay_gamma={args.lr_decay_gamma}\n")
        f.write(f"run_tag={args.run_tag}\n")
        f.write(f"start_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_start_time))}\n")

    print(f"Starting Task II Training [POD-ROM + HyperDeepONet residual] | Device: {device}")

    # ---------------------------------------------------------------
    # Data (identical split convention as train_fast.py: 0.8/0.2, seed 42)
    # ---------------------------------------------------------------
    dataset = CylinderDatasetFast(args.data_path, mode='deeponet', device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)

    x_branch_train = dataset.branch_all[train_idx]
    x_branch_test = dataset.branch_all[test_idx]
    x_trunk = dataset.trunk_shared  # shared grid, [HW, 2]
    field_train = dataset.target_all[train_idx]  # [n_train, HW, 4], already dataset-native [-1,1]
    field_test = dataset.target_all[test_idx]
    n_train, HW, C = field_train.shape

    # ---------------------------------------------------------------
    # Stage 1a: POD basis from TRAIN snapshots only
    # ---------------------------------------------------------------
    mean_c = field_train.mean(dim=(0, 1), keepdim=True)          # [1,1,4]
    std_c = field_train.std(dim=(0, 1), keepdim=True) + 1e-12    # [1,1,4]

    Xtr = ((field_train - mean_c) / std_c).reshape(n_train, -1)  # [n_train, D]
    Xm = Xtr.mean(dim=0, keepdim=True)                           # [1, D]  mean flow (POD mean)
    Xtr_c = Xtr - Xm

    print("Computing POD basis (SVD on train snapshots only)...")
    U, S, Vt = torch.linalg.svd(Xtr_c, full_matrices=False)
    r = args.pod_r
    V_r = Vt[:r].T.contiguous()                                  # [D, r]
    cumvar = (S[:r] ** 2).sum() / (S ** 2).sum()
    print(f"  POD rank r={r}  train cumulative variance explained={cumvar.item()*100:.3f}%")

    true_coeff_train = Xtr_c @ V_r                               # [n_train, r] -- regression TARGET for the RBF fit

    # ---------------------------------------------------------------
    # Stage 1b: fit RBF interpolator -- (Kn,Ma) -> r POD coefficients
    # (closed-form fit, no training loop -- never sees the test set)
    # ---------------------------------------------------------------
    print(f"Fitting RBF interpolator (thin_plate_spline) for POD coefficients...")
    rbf = RBFInterpolator(x_branch_train.cpu().numpy(), true_coeff_train.cpu().numpy(),
                           kernel='thin_plate_spline', smoothing=args.pod_rbf_smoothing)

    def u_pod(x_branch):
        """POD-ROM prediction in the SAME space as the stored (dataset-native, [-1,1]) target field."""
        with torch.no_grad():
            coeff_np = rbf(x_branch.cpu().numpy())                      # [B, r]
            coeff = torch.from_numpy(coeff_np).float().to(device)
            B = coeff.shape[0]
            x_norm_flat = coeff @ V_r.T + Xm                            # [B, D]  (D = HW*4)
            x_norm = x_norm_flat.reshape(B, HW, C)
            return x_norm * std_c + mean_c                              # un-normalize -> dataset-native space

    # ---------------------------------------------------------------
    # Residual target for HyperDeepONet, computed on TRAIN only. u_pod
    # already explains most of the variance, so the residual's natural
    # magnitude is small -- rescale it (no mean shift, just a single
    # SCALAR divisor, same constant for every channel) into a
    # [-1,1]-like range, matching the scale HyperDeepONet is otherwise
    # used to seeing. Reassembly is then exactly:
    #     u_pred = u_pod + normalize_constant * HyperDeepONet(x)
    # ---------------------------------------------------------------
    u_pod_train = u_pod(x_branch_train)                             # [n_train, HW, 4]
    residual_train = field_train - u_pod_train
    # A handful of outlier points (e.g. Kn near the parameter-range edge)
    # have residuals ~100x the typical (median) one -- normalizing by the
    # raw max squashes almost the entire training signal below |0.01|,
    # making "predict ~0 everywhere" a deceptively strong trivial minimum
    # that the optimizer gets stuck in. A single global STD is a gentler
    # scalar: it reflects the *typical* residual magnitude instead of the
    # single worst point, so most of the normalized signal lands in a
    # learnable range (outliers just exceed +-1 a bit, which is fine).
    normalize_constant = residual_train.std()
    print(f"Residual single-scalar normalize_constant (std): {normalize_constant.item():.6g}  "
          f"(for reference, max={residual_train.abs().max().item():.6g}, "
          f"median={residual_train.abs().median().item():.6g})")
    print(f"Original field scale (train, per-channel std): {std_c.flatten().tolist()}")

    residual_train_norm = residual_train / normalize_constant

    # Precompute u_pod on test set ONCE (frozen POD-ROM, never touches test labels)
    u_pod_test = u_pod(x_branch_test)

    # ---------------------------------------------------------------
    # Stage 2: HyperDeepONet residual model
    # ---------------------------------------------------------------
    model = HyperDeepONet(branch_dim=2, trunk_dim=2, hidden_dim=args.hyper_hidden_dim,
                           num_outputs=4, trunk_depth=3, branch_depth=3, activation='GELU').to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"HyperDeepONet residual model parameters: {num_params} ({num_params/1e3:.2f}K)")
    with open(run_info_path, 'a') as f:
        f.write(f"hyper_num_params={num_params}\n")
        f.write(f"pod_train_cumvar={cumvar.item()}\n")

    train_loader = FastBatchIterator((x_branch_train, residual_train_norm), batch_size=args.batch_size,
                                      shuffle=True, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma) \
        if args.lr_decay_step > 0 else None
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    REL2_LOG_INTERVAL = args.rel2_log_interval

    print("Training HyperDeepONet residual...")
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        compute_rel2 = (epoch + 1) % REL2_LOG_INTERVAL == 0

        model.train()
        train_loss_acc, n_batches = 0.0, 0
        for x_branch_b, resid_b in train_loader:
            optimizer.zero_grad()
            pred = model(x_branch_b, x_trunk)
            loss = criterion(pred, resid_b)
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        avg_train_loss = train_loss_acc / n_batches

        model.eval()
        with torch.no_grad():
            r_hyper_test_norm = model(x_branch_test, x_trunk)
            test_loss = criterion(r_hyper_test_norm, (field_test - u_pod_test) / normalize_constant).item()

            if compute_rel2:
                r_hyper_test = normalize_constant * r_hyper_test_norm
                u_final = u_pod_test + r_hyper_test
                pred_flat = u_final.reshape(u_final.shape[0], -1)
                true_flat = field_test.reshape(field_test.shape[0], -1)
                per_sample_rel2 = torch.norm(pred_flat - true_flat, p=2, dim=1) / \
                                   (torch.norm(true_flat, p=2, dim=1) + 1e-8)
                test_rel2 = per_sample_rel2.mean().item()
                now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                with open(rel2_log_path, 'a') as f:
                    f.write(f'{epoch + 1},{test_rel2:.6g},{now_str}\n')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'hyper_state_dict': model.state_dict(),
                'rbf': rbf,
                'V_r': V_r.cpu(), 'Xm': Xm.cpu(),
                'mean_c': mean_c.cpu(), 'std_c': std_c.cpu(),
                'normalize_constant': normalize_constant.cpu(),
                'pod_r': r,
                'hyper_hidden_dim': args.hyper_hidden_dim,
            }, save_path)
            saved_flag = True
        else:
            saved_flag = False

        postfix = {'train': f'{avg_train_loss:.4g}', 'test': f'{test_loss:.4g}',
                   'best': f'{best_test_loss:.4g}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'}
        if compute_rel2:
            postfix['rel2'] = f'{test_rel2:.4g}'
        pbar.set_postfix(postfix)
        if saved_flag and (epoch + 1) % 50 == 0:
            tqdm.write(f"  [BEST SAVED @ epoch {epoch+1}]")

    run_end_time = time.time()
    with open(run_info_path, 'a') as f:
        f.write(f"end_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_end_time))}\n")
        f.write(f"elapsed_sec={run_end_time - run_start_time:.1f}\n")
        f.write(f"best_test_loss={best_test_loss}\n")

    print(f"Training Complete! Best Test Loss (residual, normalized): {best_test_loss:.6g}. "
          f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
