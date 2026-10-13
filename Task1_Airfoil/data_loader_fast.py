"""
Overhead-fixed data loader for Task I (Airfoil).

Same rationale as Task2_Cylinder/data_loader_fast.py: `AirfoilDataset.__getitem__`
reshapes/permutes each sample fresh, and DataLoader's default_collate re-stacks the
whole split from scratch every epoch even though the split never changes across a
run. Here we reshape the WHOLE dataset once in __init__ (vectorized, not a python
loop over samples), keep it resident on `device`, and hand out batches via plain
tensor indexing.

This file is a parallel, opt-in replacement -- data_loader.py/train.py are untouched.
Only train_fast.py uses this.

Note: the per-sample forward loop in train.py (`for i in range(batch_size):
pred_i = model(x_branch[i:i+1], x_trunk[i])`) is a *separate* source of overhead
(one full model call per sample instead of a single batched call) tied to how
each baseline model's trunk handles a per-sample-varying query grid. train_fast.py
deliberately keeps that loop as-is -- only the data-fetch path is fixed here -- so
results stay directly comparable to train.py without re-verifying every model's
batched-trunk behavior.
"""
import torch


class AirfoilDatasetFast:
    def __init__(self, pt_path, mode='deeponet', device='cpu'):
        self.mode = mode.lower()
        print(f"[fast] Loading data from {pt_path}...")
        data_dict = torch.load(pt_path)

        geo_points  = data_dict["geometry_points"]  # [N, 337, 2]
        geo_mask    = data_dict["geometry_mask"]    # [N, 1, 128, 128]
        flow_label  = data_dict["flow_label"]       # [N, 128, 128, 4]
        grid_coords = data_dict["grid_coords"]      # [N, 128, 128, 2]
        self.stats  = data_dict["stats"]

        self.n_samples = flow_label.shape[0]
        print(f"[fast] Data loaded. Mode: {self.mode}, Samples: {self.n_samples}")

        # Precompute ONCE (vectorized ops over the whole tensor), instead of
        # redoing these per-sample inside __getitem__ on every epoch.
        self.geo_mask_flat = geo_mask.reshape(self.n_samples, -1, 1)  # [N, 16384, 1]

        if self.mode == 'deeponet':
            self.branch_all = geo_points.reshape(self.n_samples, -1)          # [N, 674]
            self.trunk_all  = grid_coords.reshape(self.n_samples, -1, 2)      # [N, 16384, 2]
            self.target_all = flow_label.reshape(self.n_samples, -1, 4)       # [N, 16384, 4]
        elif self.mode == 'fno':
            coords_all = grid_coords.permute(0, 3, 1, 2)                      # [N, 2, H, W]
            self.x_input_all = torch.cat([geo_mask, coords_all], dim=1).contiguous()  # [N, 3, H, W]
            self.target_all = flow_label.permute(0, 3, 1, 2).contiguous()     # [N, 4, H, W]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if device != 'cpu':
            self._to(device)

    def _to(self, device):
        self.geo_mask_flat = self.geo_mask_flat.to(device)
        if self.mode == 'deeponet':
            self.branch_all = self.branch_all.to(device)
            self.trunk_all = self.trunk_all.to(device)
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
    No Dataset.__getitem__, no default_collate, no per-epoch mask-lookup loop."""
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
