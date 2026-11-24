"""
7-family x 5-param-tier experiment matrix (trunk_depth=5 / depth=5
everywhere, matching Fusion_DeepONet's layer count), each run standalone:
searches hidden_dim (and s for LRNR variants) to hit the target param
count, trains for --epochs with the standard recipe (Adam, StepLR,
masked MSE, same data pipeline as train_fast.py/train_lrnr_ortho.py), and
records best rel2, peak GPU memory, and wall-clock time to run_info.txt.

Families: fusion, lrnr_lb_adapt, lrnr_nn_adapt, hyper_adapt,
          lrnr_lb_plain, lrnr_nn_plain, hyper_plain
"""
import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'low_rank_models'))
from model_fusion_deeponet import Fusion_DeepONet
from model_LRNR import LRNR
from model_LRNR_linearbranch import LRNR_LinearBranch
from model_LRNR_adaptact import LRNR_AdaptAct
from model_hyperdeeponet import HyperDeepONet
from model_hyperdeeponet_adaptact import HyperDeepONet_AdaptAct
from model_deeponet import BoltzmannDeepONet
from model_deeponet_adaptact import BoltzmannDeepONet_AdaptAct
from data_loader_fast import CylinderDatasetFast, FastBatchIterator

FAMILIES = ['fusion', 'lrnr_lb_adapt', 'lrnr_nn_adapt', 'hyper_adapt',
            'lrnr_lb_plain', 'lrnr_nn_plain', 'hyper_plain',
            'deeponet_plain', 'deeponet_adapt']
CFG = dict(branch_dim=2, trunk_dim=2, num_outputs=4, branch_depth=3)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def build_model(family, target, h_max=300, s_list=(8, 12, 16, 24, 32, 48, 64), trunk_depth=5, fixed_hidden_dim=0):
    if fixed_hidden_dim > 0:
        h = fixed_hidden_dim
        if family in ('deeponet_plain', 'deeponet_adapt'):
            cls = BoltzmannDeepONet_AdaptAct if family.endswith('adapt') else BoltzmannDeepONet
            m = cls(branch_dim=2, trunk_dim=2, hidden_dim=h, num_outputs=4, depth=5, activation='GELU')
        elif family in ('hyper_adapt', 'hyper_plain'):
            cls = HyperDeepONet_AdaptAct if family.endswith('adapt') else HyperDeepONet
            m = cls(**CFG, hidden_dim=h, trunk_depth=trunk_depth, activation='GELU')
        else:
            raise ValueError(f"fixed_hidden_dim not supported for family={family}")
        n = count_params(m)
        return m, n, f"hidden_dim={h}"

    if family in ('deeponet_plain', 'deeponet_adapt'):
        adapt = family.endswith('adapt')
        cls = BoltzmannDeepONet_AdaptAct if adapt else BoltzmannDeepONet
        best = None
        for h in range(2, h_max):
            m = cls(branch_dim=2, trunk_dim=2, hidden_dim=h, num_outputs=4, depth=5, activation='GELU')
            n = count_params(m)
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (h, n, m)
        return best[2], best[1], f"hidden_dim={best[0]}"

    if family == 'fusion':
        best = None
        for h in range(4, h_max):
            m = Fusion_DeepONet(branch_dim=2, trunk_dim=2, num_outputs=4, hidden_dim=h, depth=5, activation='GELU')
            n = count_params(m)
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (h, n, m)
        return best[2], best[1], f"hidden_dim={best[0]}"

    if family in ('lrnr_lb_adapt', 'lrnr_nn_adapt', 'lrnr_lb_plain', 'lrnr_nn_plain'):
        linear_branch = '_lb_' in family
        adapt = family.endswith('adapt')
        best = None
        for h in range(8, h_max):
            for s in s_list:
                if s > h:
                    continue
                if adapt:
                    m = LRNR_AdaptAct(**CFG, hidden_dim=h, trunk_depth=5, s=s, linear_branch=linear_branch)
                else:
                    cls = LRNR_LinearBranch if linear_branch else LRNR
                    m = cls(**CFG, hidden_dim=h, trunk_depth=5, s=s)
                n = count_params(m)
                if best is None or abs(n - target) < abs(best[2] - target):
                    best = (h, s, n, m)
        return best[3], best[2], f"hidden_dim={best[0]},s={best[1]}"

    # hyper_adapt / hyper_plain
    adapt = family.endswith('adapt')
    best = None
    for h in range(4, h_max):
        cls = HyperDeepONet_AdaptAct if adapt else HyperDeepONet
        m = cls(**CFG, hidden_dim=h, trunk_depth=trunk_depth, activation='GELU')
        n = count_params(m)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (h, n, m)
    return best[2], best[1], f"hidden_dim={best[0]}"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True, choices=FAMILIES)
    p.add_argument('--target_params', type=int, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_decay_step', type=int, default=100)
    p.add_argument('--lr_decay_gamma', type=float, default=0.91)
    p.add_argument('--data_path', type=str, default='./cylinder_full_2400_400_seed42.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rel2_log_interval', type=int, default=50)
    p.add_argument('--run_tag', type=str, default='')
    p.add_argument('--trunk_depth', type=int, default=5, help='hyper_adapt/hyper_plain trunk_depth (ignored by other families)')
    p.add_argument('--hidden_dim', type=int, default=0, help='if >0, use this hidden_dim directly instead of searching for target_params')
    return p.parse_args()


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tag = args.run_tag or f"{args.family}_{args.target_params}"
    save_dir = os.path.join('output', f"matrix_{tag}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    rel2_log_path = os.path.join(save_dir, 'rel2_history.csv')
    with open(rel2_log_path, 'w') as f:
        f.write('epoch,test_rel2_error,timestamp\n')
    run_info_path = os.path.join(save_dir, 'run_info.txt')

    print(f"=== family={args.family} target_params={args.target_params} ===")
    model, n_params, cfg_str = build_model(args.family, args.target_params, trunk_depth=args.trunk_depth,
                                            fixed_hidden_dim=args.hidden_dim)
    print(f"Config: {cfg_str} -> {n_params} params")
    model = model.to(device)

    run_start_time = time.time()
    with open(run_info_path, 'w') as f:
        f.write(f"family={args.family}\n")
        f.write(f"target_params={args.target_params}\n")
        f.write(f"config={cfg_str}\n")
        f.write(f"num_params={n_params}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"start_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_start_time))}\n")

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    dataset = CylinderDatasetFast(args.data_path, mode='deeponet', device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)
    dataset.fit_train_normalizer(train_idx)
    fluid = dataset.fluid_mask

    train_tensors = (dataset.branch_all[train_idx], dataset.target_all[train_idx])
    test_tensors = (dataset.branch_all[test_idx], dataset.target_all[test_idx])
    train_loader = FastBatchIterator(train_tensors, batch_size=args.batch_size, shuffle=True, device=device)
    test_loader = FastBatchIterator(test_tensors, batch_size=args.batch_size, shuffle=False, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma) \
        if args.lr_decay_step > 0 else None
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    best_rel2 = float('inf')
    REL2_LOG_INTERVAL = args.rel2_log_interval

    pbar = tqdm(range(args.epochs), desc=f"{args.family}_{args.target_params}")
    for epoch in pbar:
        compute_rel2 = (epoch + 1) % REL2_LOG_INTERVAL == 0
        rel2_sum, rel2_count = 0.0, 0

        model.train()
        train_loss_acc = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            x_branch, y = batch
            x_branch = x_branch[:, :2]
            x_trunk = dataset.trunk_shared
            pred = model(x_branch, x_trunk)
            loss = criterion(pred[:, fluid], y[:, fluid])
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()

        if scheduler is not None:
            scheduler.step()
        avg_train_loss = train_loss_acc / len(train_loader)

        model.eval()
        test_loss_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x_branch, y = batch
                x_branch = x_branch[:, :2]
                x_trunk = dataset.trunk_shared
                pred = model(x_branch, x_trunk)
                loss = criterion(pred[:, fluid], y[:, fluid])
                test_loss_acc += loss.item()

                if compute_rel2:
                    pred_f, y_f = pred[:, fluid], y[:, fluid]
                    pred_flat = pred_f.reshape(pred_f.shape[0], -1)
                    y_flat = y_f.reshape(y_f.shape[0], -1)
                    per_sample_rel2 = torch.norm(pred_flat - y_flat, p=2, dim=1) / \
                                       (torch.norm(y_flat, p=2, dim=1) + 1e-8)
                    rel2_sum += per_sample_rel2.sum().item()
                    rel2_count += pred_flat.shape[0]

        avg_test_loss = test_loss_acc / len(test_loader)

        if compute_rel2:
            test_rel2 = rel2_sum / rel2_count
            if test_rel2 < best_rel2:
                best_rel2 = test_rel2
            now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            with open(rel2_log_path, 'a') as f:
                f.write(f'{epoch + 1},{test_rel2:.6g},{now_str}\n')

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)

        postfix = {'train': f'{avg_train_loss:.4g}', 'test': f'{avg_test_loss:.4g}', 'best_rel2': f'{best_rel2:.4g}'}
        pbar.set_postfix(postfix)

    elapsed_sec = time.time() - run_start_time
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device == 'cuda' else 0.0

    with open(run_info_path, 'a') as f:
        f.write(f"end_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"elapsed_sec={elapsed_sec:.1f}\n")
        f.write(f"best_test_loss={best_test_loss:.6g}\n")
        f.write(f"best_rel2={best_rel2:.6g}\n")
        f.write(f"peak_memory_mb={peak_mem_mb:.1f}\n")

    print(f"DONE family={args.family} target={args.target_params} params={n_params} "
          f"best_rel2={best_rel2*100:.3f}% elapsed={elapsed_sec/60:.1f}min peak_mem={peak_mem_mb:.0f}MB")


if __name__ == "__main__":
    main()
