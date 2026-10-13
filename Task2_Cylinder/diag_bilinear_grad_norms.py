"""
Diagnostic: log per-epoch gradient L2 norms for Bilinear_DeepONet's U/V/C
low-rank factors (plus branch_net/trunk_net for reference), to check whether
the stuck-at-high-rel2 plateau seen in the normal training run is caused by
vanishing gradients in the bilinear interaction specifically.

Same data/optimizer/schedule as the real run (cylinder_full_2400_400_seed42.pt,
hidden_dim=29 -> ~20K params, Adam lr=1e-3, StepLR 100/0.91), just fewer
epochs (this is a diagnostic, not a full training run) and with gradient-norm
logging added.
"""
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model_bilinear_deeponet import Bilinear_DeepONet
from data_loader_fast import CylinderDatasetFast, FastBatchIterator

EPOCHS = 300
HIDDEN_DIM = 29
OUT_DIR = 'experiment_logs_bilinear'
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, 'grad_norms.csv')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

dataset = CylinderDatasetFast('./cylinder_full_2400_400_seed42.pt', mode='deeponet', device=device)
train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
train_idx = train_idx.to(device)

train_tensors = (dataset.branch_all[train_idx], dataset.target_all[train_idx])
train_loader = FastBatchIterator(train_tensors, batch_size=16, shuffle=True, device=device)

model = Bilinear_DeepONet(branch_dim=2, trunk_dim=2, hidden_dim=HIDDEN_DIM, num_outputs=4,
                           depth=5, activation='GELU').to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {n_params}")

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.91)
criterion = nn.MSELoss()


def group_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'grad_U', 'grad_V', 'grad_C',
                      'grad_branch_net', 'grad_trunk_net', 'grad_ab_cb_scalars'])

    pbar = tqdm(range(EPOCHS), desc="Diag")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        g_U = g_V = g_C = g_branch = g_trunk = g_scalars = 0.0
        n_batches = 0
        for x_branch, y in train_loader:
            x_branch = x_branch[:, :2]
            x_trunk = dataset.trunk_shared
            optimizer.zero_grad()
            pred = model(x_branch, x_trunk)
            loss = criterion(pred, y)
            loss.backward()

            g_U += group_grad_norm([model.U])
            g_V += group_grad_norm([model.V])
            g_C += group_grad_norm([model.C])
            g_branch += group_grad_norm(model.branch_net.parameters())
            g_trunk += group_grad_norm(model.trunk_net.parameters())
            g_scalars += group_grad_norm([model.ab, model.cb, model.a1b, model.F1b, model.c1b])

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        row = [epoch + 1, avg_loss, g_U / n_batches, g_V / n_batches, g_C / n_batches,
               g_branch / n_batches, g_trunk / n_batches, g_scalars / n_batches]
        writer.writerow(row)
        f.flush()

        pbar.set_postfix({'loss': f'{avg_loss:.4g}', 'gU': f'{g_U/n_batches:.3g}',
                           'gV': f'{g_V/n_batches:.3g}', 'gC': f'{g_C/n_batches:.3g}'})

print(f"Saved gradient-norm log to {CSV_PATH}")
