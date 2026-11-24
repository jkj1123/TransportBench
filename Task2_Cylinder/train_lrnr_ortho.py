"""
Dedicated training script for LRNR (low_rank_models/model_LRNR.py) +
orthogonality regularization on its U_l/V_l low-rank factors:

    L_total = L_data + lambda_ortho * R_ortho

Shares the exact same data pipeline, optimizer (Adam, weight_decay=1e-4),
scheduler (StepLR), criterion (masked MSE), and train/test loop structure
as train_fast.py -- only the per-batch loss (and what gets logged) differs.
lambda_ortho=0.0 (default) reproduces plain LRNR exactly; see
`python train_lrnr_ortho.py --selftest` for the unit tests proving this and
the gradient-flow/shape/NaN checks for the regularizer.

No change to the model file's forward()/architecture/init, and no change
to train_fast.py.
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
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'low_rank_models'))
from model_LRNR import LRNR
from model_LRNR_linearbranch import LRNR_LinearBranch
from model_LRNR_adaptact import LRNR_AdaptAct
from data_loader_fast import CylinderDatasetFast, FastBatchIterator


def lrnr_orthogonality_penalty(model):
    """R_ortho = sum_l [ ||U_l^T U_l - I||_F^2 / numel(U_l) + ||V_l^T V_l - I||_F^2 / numel(V_l) ]

    Verified against model_LRNR.py (LRNR._trunk_forward) before writing this:
      - U_l is stored DIRECTLY as `trunk_net.hidden_0_net[2*layer_idx].weight`,
        shape [dims[i+1], s] == [d_out, s] (the model's own comment: "U_l: [dims[i+1], s]").
      - The matrix stored at `trunk_net.hidden_1_net[2*layer_idx].weight`, shape
        [s, dims[i]] == [s, d_in], is V_l^T itself, NOT V_l (comment: "V_l^T: [s, dims[i]]").
        So V_l = that stored matrix transposed, shape [d_in, s].
      Both Gram matrices are therefore computed as:
        U_l^T U_l   = weight1.T @ weight1            (weight1 = U_l,  [d_out,s] -> [s,s])
        V_l^T V_l   = weight2   @ weight2.T           (weight2 = V_l^T,[s,d_in] -> [s,s], since
                                                        V_l^T V_l = (V_l^T)(V_l^T)^T when V_l^T
                                                        is the matrix already in hand)
      i.e. gram_V is NOT weight2.T @ weight2 -- using that instead would silently compute
      V_l V_l^T ([d_in,d_in]) instead of V_l^T V_l ([s,s]), the wrong Gram matrix and wrong
      shape for all but square layers. This is the "stored transposed" case the task asked
      to check for.

    Only U_l/V_l (hidden_0_net/hidden_1_net) are touched -- branch_net (the hypernetwork),
    bias_basis_net (B_l), and s_w_l/s_b_l are not regularized.

    No torch.no_grad()/.detach() anywhere: every tensor here is a plain autograd-tracked
    view/matmul of the model's own nn.Parameter weights, so grad flows back into U_l/V_l
    exactly as for any other term in the loss.

    Returns a 0-dim tensor on the same device/dtype as the model's own weights.
    """
    penalty = None
    for layer_idx in range(model.num_hidden_layers):
        U_l = model.trunk_net.hidden_0_net[2 * layer_idx].weight   # [d_out, s]  (this IS U_l)
        Vt_l = model.trunk_net.hidden_1_net[2 * layer_idx].weight  # [s, d_in]   (this IS V_l^T)

        s = U_l.shape[1]
        I_s = torch.eye(s, device=U_l.device, dtype=U_l.dtype)

        gram_U = U_l.t() @ U_l       # U_l^T U_l,  [s, s]
        gram_V = Vt_l @ Vt_l.t()     # V_l^T V_l,  [s, s]  (Vt_l already is V_l^T)

        term = ((gram_U - I_s) ** 2).sum() / U_l.numel() \
             + ((gram_V - I_s) ** 2).sum() / Vt_l.numel()
        penalty = term if penalty is None else penalty + term
    return penalty


def get_args():
    parser = argparse.ArgumentParser(description="Task II: LRNR+orthogonality ablation")
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_path', type=str, default='./cylinder_full_2400_400_seed42.pt')
    parser.add_argument('--run_tag', type=str, default='')
    parser.add_argument('--hidden_dim', type=int, default=0, help='0 = default 46')
    parser.add_argument('--lr_decay_step', type=int, default=0)
    parser.add_argument('--lr_decay_gamma', type=float, default=1.0)
    parser.add_argument('--rel2_log_interval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lrnr_s', type=int, default=12)
    parser.add_argument('--lrnr_s_w_offset', type=float, default=0.0)
    parser.add_argument('--lambda_ortho', type=float, default=0.0,
                        help='Weight on R_ortho. 0.0 (default) reproduces plain LRNR exactly.')
    parser.add_argument('--linear_branch', action='store_true',
                        help='Replace the branch MLP with a single nn.Linear '
                             '(no activation, no hidden layers). See model_LRNR_linearbranch.py.')
    parser.add_argument('--adapt_act', action='store_true',
                        help='Use Fusion_DeepONet-style adaptive activation '
                             '(per-layer a,c,F1,a1,c1) on the trunk instead of plain GELU/Tanh. '
                             'Combinable with --linear_branch. See model_LRNR_adaptact.py.')
    parser.add_argument('--selftest', action='store_true',
                        help='Run the lightweight unit tests in __main__ and exit (no training).')
    return parser.parse_args()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Starting Task II Training | Model: LRNR_ORTHO (lambda_ortho={args.lambda_ortho}) | Device: {device}")

    model_dir_name = f"lrnr_ortho_{args.run_tag}" if args.run_tag else "lrnr_ortho"
    save_dir = os.path.join('output', model_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    rel2_log_path = os.path.join(save_dir, 'rel2_history.csv')
    with open(rel2_log_path, 'w') as f:
        f.write('epoch,test_rel2_error,timestamp\n')
    loss_log_path = os.path.join(save_dir, 'loss_breakdown.csv')
    with open(loss_log_path, 'w') as f:
        f.write('epoch,train_data_loss,train_ortho_penalty,train_total_loss\n')

    run_start_time = time.time()
    run_info_path = os.path.join(save_dir, 'run_info.txt')
    with open(run_info_path, 'w') as f:
        f.write("model=lrnr_ortho\n")
        f.write(f"lambda_ortho={args.lambda_ortho}\n")
        f.write(f"hidden_dim_override={args.hidden_dim}\n")
        f.write(f"lrnr_s={args.lrnr_s}\n")
        f.write(f"linear_branch={args.linear_branch}\n")
        f.write(f"adapt_act={args.adapt_act}\n")
        f.write(f"data_path={args.data_path}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"lr_decay_step={args.lr_decay_step}\n")
        f.write(f"lr_decay_gamma={args.lr_decay_gamma}\n")
        f.write(f"run_tag={args.run_tag}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"start_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_start_time))}\n")

    dataset = CylinderDatasetFast(args.data_path, mode='deeponet', device=device)
    train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)
    dataset.fit_train_normalizer(train_idx)
    fluid = dataset.fluid_mask

    train_tensors = (dataset.branch_all[train_idx], dataset.target_all[train_idx])
    test_tensors = (dataset.branch_all[test_idx], dataset.target_all[test_idx])
    train_loader = FastBatchIterator(train_tensors, batch_size=args.batch_size, shuffle=True, device=device)
    test_loader = FastBatchIterator(test_tensors, batch_size=args.batch_size, shuffle=False, device=device)

    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else 46
    if args.adapt_act:
        model = LRNR_AdaptAct(branch_dim=2, trunk_dim=2, hidden_dim=hidden_dim, num_outputs=4,
                              trunk_depth=3, branch_depth=3, activation='GELU', s=args.lrnr_s,
                              s_w_offset=args.lrnr_s_w_offset, linear_branch=args.linear_branch).to(device)
    else:
        lrnr_cls = LRNR_LinearBranch if args.linear_branch else LRNR
        model = lrnr_cls(branch_dim=2, trunk_dim=2, hidden_dim=hidden_dim, num_outputs=4,
                         trunk_depth=3, branch_depth=3, activation='GELU', s=args.lrnr_s,
                         s_w_offset=args.lrnr_s_w_offset).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.2f} M")
    with open(run_info_path, 'a') as f:
        f.write(f"num_params={num_params}\n")

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    if args.lr_decay_step > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    history = {'train_loss': [], 'test_loss': []}
    REL2_LOG_INTERVAL = args.rel2_log_interval

    print("Training Started...")
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        compute_rel2 = (epoch + 1) % REL2_LOG_INTERVAL == 0
        rel2_sum, rel2_count = 0.0, 0

        model.train()
        train_loss_acc, data_loss_acc, reg_acc = 0.0, 0.0, 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            x_branch, y = batch
            x_branch = x_branch[:, :2]
            x_trunk = dataset.trunk_shared

            pred = model(x_branch, x_trunk)
            loss_data = criterion(pred[:, fluid], y[:, fluid])
            reg_penalty = lrnr_orthogonality_penalty(model)
            loss = loss_data + args.lambda_ortho * reg_penalty

            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()
            data_loss_acc += loss_data.item()
            reg_acc += reg_penalty.item()

        if scheduler is not None:
            scheduler.step()

        n_batches = len(train_loader)
        avg_train_loss = train_loss_acc / n_batches
        avg_data_loss = data_loss_acc / n_batches
        avg_reg = reg_acc / n_batches
        history['train_loss'].append(avg_train_loss)
        with open(loss_log_path, 'a') as f:
            f.write(f'{epoch + 1},{avg_data_loss:.6g},{avg_reg:.6g},{avg_train_loss:.6g}\n')

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
        history['test_loss'].append(avg_test_loss)

        if compute_rel2:
            test_rel2 = rel2_sum / rel2_count
            now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            with open(rel2_log_path, 'a') as f:
                f.write(f'{epoch + 1},{test_rel2:.6g},{now_str}\n')

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)

        postfix = {
            'data': f'{avg_data_loss:.4g}',
            'ortho': f'{avg_reg:.4g}',
            'total': f'{avg_train_loss:.4g}',
            'test': f'{avg_test_loss:.4g}',
            'best': f'{best_test_loss:.4g}',
        }
        if compute_rel2:
            postfix['rel2'] = f'{test_rel2:.4g}'
        pbar.set_postfix(postfix)

    print(f"Training Complete! Best Test Loss: {best_test_loss:.4g}. Model saved to {save_path}")

    run_end_time = time.time()
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device == 'cuda' else 0.0
    with open(run_info_path, 'a') as f:
        f.write(f"end_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_end_time))}\n")
        f.write(f"elapsed_sec={run_end_time - run_start_time:.1f}\n")
        f.write(f"peak_memory_mb={peak_mem_mb:.1f}\n")
        f.write(f"best_test_loss={best_test_loss:.6g}\n")

    np.save(os.path.join(save_dir, 'history.npy'), history)
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train (data+ortho)', alpha=0.8)
    plt.plot(history['test_loss'], label='Test (data only)', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'lrnr_ortho (lambda_ortho={args.lambda_ortho})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150)


def _selftest():
    """Lightweight unit tests, no CFD data required."""
    torch.manual_seed(0)

    print("=" * 60)
    print("lrnr_ortho")
    print("=" * 60)
    cfg = dict(branch_dim=2, trunk_dim=2, hidden_dim=46, num_outputs=4,
               trunk_depth=3, branch_depth=3, activation='GELU', s=12)

    print("1) lambda_ortho=0 -> total loss == data loss exactly")
    model = LRNR(**cfg)
    xb = torch.randn(5, 2)
    xt = torch.randn(50, 2)
    y = torch.randn(5, 50, 4)
    pred = model(xb, xt)
    loss_data = nn.functional.mse_loss(pred, y)
    ortho = lrnr_orthogonality_penalty(model)
    loss_total = loss_data + 0.0 * ortho
    print(f"   loss_data={loss_data.item():.8f}  loss_total(lambda=0)={loss_total.item():.8f}  "
          f"equal={loss_data.item() == loss_total.item()}")
    assert loss_data.item() == loss_total.item()

    print("2) lambda_ortho>0 -> ortho penalty computed, finite, nonzero")
    ortho_val = ortho.item()
    print(f"   R_ortho={ortho_val:.6f}  finite={np.isfinite(ortho_val)}")
    assert np.isfinite(ortho_val) and ortho_val > 0

    print("3) U_l/V_l receive gradient from the ortho penalty alone (data loss excluded)")
    model.zero_grad()
    ortho2 = lrnr_orthogonality_penalty(model)
    ortho2.backward()
    for layer_idx in range(model.num_hidden_layers):
        U_l = model.trunk_net.hidden_0_net[2 * layer_idx].weight
        Vt_l = model.trunk_net.hidden_1_net[2 * layer_idx].weight
        assert U_l.grad is not None and U_l.grad.abs().sum().item() > 0
        assert Vt_l.grad is not None and Vt_l.grad.abs().sum().item() > 0
    branch_grad_zero = all(p.grad is None for p in model.branch_net.parameters())
    bias_basis_grad_zero = all(p.grad is None for p in model.trunk_net.bias_basis_net.parameters())
    print(f"   U_l/V_l all nonzero grad: True;  branch_net untouched: {branch_grad_zero};  "
          f"bias_basis_net untouched: {bias_basis_grad_zero}")
    assert branch_grad_zero and bias_basis_grad_zero

    print("4) shape sanity: 0-dim tensor")
    assert ortho.shape == ()

    print("5) non-orthogonal (scaled) U_l -> larger penalty")
    model2 = LRNR(**cfg)
    with torch.no_grad():
        for layer_idx in range(model2.num_hidden_layers):
            model2.trunk_net.hidden_0_net[2 * layer_idx].weight.mul_(10.0)
    ortho_small = lrnr_orthogonality_penalty(model).item()
    ortho_big = lrnr_orthogonality_penalty(model2).item()
    print(f"   R_ortho(normal)={ortho_small:.4f}  R_ortho(x10)={ortho_big:.4f}")
    assert ortho_big > ortho_small

    print("6) no NaN/Inf across 20 random re-inits")
    for _ in range(20):
        m = LRNR(**cfg)
        assert np.isfinite(lrnr_orthogonality_penalty(m).item())
    print("   OK")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    args = get_args()
    if args.selftest:
        _selftest()
    else:
        main(args)
