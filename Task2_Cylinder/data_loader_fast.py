"""
Overhead-fixed data loader for Task II (Cylinder).

Why this exists
----------------
`CylinderDataset` (data_loader.py) reshapes/permutes each sample fresh inside
`__getitem__`, and DataLoader's default_collate then re-stacks 192+48 individual
(24576, 2)/(24576, 4) tensors into one batch tensor -- EVERY epoch, even though
the split never changes across the 50,000 epochs of a run. Profiling showed this
costs ~47ms/epoch just to *fetch* the data, independent of model size (measured
2026-09-05, see Task2_Cylinder chat history) -- for small hyperdeeponet configs
that dominates wall time and makes it a bad proxy for the model's real compute.

Fix: reshape/permute the WHOLE dataset once in __init__ (one vectorized op,
not a python loop over samples), keep it resident on `device`, and hand out
batches via plain tensor indexing (no Dataset.__getitem__, no collate).

This file is a parallel, opt-in replacement -- `data_loader.py`/`train.py` are
untouched. Only `train_fast.py` uses this.
"""
import torch


class CylinderDatasetFast:
    def __init__(self, pt_path, mode='deeponet', device='cpu'):
        self.mode = mode.lower()
        print(f"[fast] Loading cylinder data from {pt_path}...")
        data_dict = torch.load(pt_path)

        input_params = data_dict["input_params"]  # [N, 2] -> Kn, Ma
        flow_label   = data_dict["flow_label"]     # [N, 128, 192, 4]
        mask         = data_dict["mask"]           # [N, 1, 128, 192]
        grid_coords  = data_dict["grid_coords"]    # [N, 128, 192, 2]
        self.stats   = data_dict["flow_stats"]

        self.n_samples = flow_label.shape[0]
        self.H, self.W = flow_label.shape[1], flow_label.shape[2]
        print(f"[fast] Data loaded. Samples: {self.n_samples}, Grid: {self.H}x{self.W}")

        if self.mode == 'deeponet':
            # Branch input: (Kn, Ma) per sample -- already [N, 2], nothing to precompute.
            self.branch_all = input_params
            # All samples share the same query grid (train.py already only ever used
            # sample 0's grid at each step) -- compute it ONCE instead of once per sample.
            self.trunk_shared = grid_coords[0].reshape(-1, 2)          # [24576, 2]
            # One vectorized reshape of the whole tensor, not a per-sample python loop.
            self.target_all = flow_label.reshape(self.n_samples, -1, 4)  # [N, 24576, 4]

        elif self.mode == 'grid':
            kn_ma_all = input_params.view(self.n_samples, 2, 1, 1).expand(-1, -1, self.H, self.W)
            coords_all = grid_coords.permute(0, 3, 1, 2)                # [N, 2, H, W]
            self.x_input_all = torch.cat([kn_ma_all, coords_all], dim=1).contiguous()  # [N, 4, H, W]
            self.target_all = flow_label.permute(0, 3, 1, 2).contiguous()              # [N, 4, H, W]

        elif self.mode == 'point':
            kn_ma_all = input_params.view(self.n_samples, 2, 1, 1).expand(-1, -1, self.H, self.W)
            coords_all = grid_coords.permute(0, 3, 1, 2)
            x_input_all = torch.cat([kn_ma_all, coords_all], dim=1)
            self.x_input_all = x_input_all.permute(0, 2, 3, 1).reshape(self.n_samples, -1, 4).contiguous()
            self.target_all = flow_label.reshape(self.n_samples, -1, 4).contiguous()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if device != 'cpu':
            self._to(device)

    def _to(self, device):
        if self.mode == 'deeponet':
            self.branch_all = self.branch_all.to(device)
            self.trunk_shared = self.trunk_shared.to(device)
            self.target_all = self.target_all.to(device)
        else:
            self.x_input_all = self.x_input_all.to(device)
            self.target_all = self.target_all.to(device)

    def split(self, train_ratio=0.8, seed=42):
        """Reproduces torch.utils.data.random_split(..., generator=Generator().manual_seed(seed))
        index-for-index, so results are directly comparable to the original train.py."""
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(self.n_samples, generator=g)
        n_train = int(train_ratio * self.n_samples)
        return perm[:n_train], perm[n_train:]


class FastBatchIterator:
    """Yields minibatches by index-slicing pre-cached, already-on-device tensors.
    No Dataset.__getitem__, no default_collate -- each batch is one fancy-index op."""
    def __init__(self, tensors, batch_size, shuffle, device):
        self.tensors = tensors
        self.n = tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.device) if self.shuffle \
              else torch.arange(self.n, device=self.device)
        for start in range(0, self.n, self.batch_size):
            b = idx[start:start + self.batch_size]
            yield tuple(t[b] for t in self.tensors)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size
