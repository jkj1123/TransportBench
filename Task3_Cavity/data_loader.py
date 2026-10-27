import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset

class CavityDataset(Dataset):
    def __init__(self, data_dir='../../data/cavity', mode='train', model_type='fno',
                 pt_path=None, use_cache=True):
        """
        FNO Dataset Loader
        Input:  (Batch, 50, 50, 3)  -> [Kn, x, y]
        Output: (Batch, 50, 50, 10) -> [w, P, q]

        pt_path   : path of a compact .pt file holding the processed dataset.
                    If given and it exists (and use_cache), the dataset is loaded
                    straight from it and the raw .npz files are not touched.
                    If given but missing, the dataset is built from .npz and then
                    saved to this path.
        use_cache : set False to force rebuilding from .npz even if the .pt exists.
        """
        self.model_type = model_type.lower()
        self.pt_path = pt_path

        # Fast path: load the compact .pt directly.
        if pt_path is not None and use_cache and os.path.exists(pt_path):
            self._load_pt(pt_path)
            return

        self._build_from_npz(data_dir, model_type)

        # Persist the processed dataset as a single small .pt file.
        if pt_path is not None:
            self.save_pt(pt_path)

    def _build_from_npz(self, data_dir, model_type):
        search_path = os.path.join(os.path.expanduser(data_dir), '*.npz')
        file_list = sorted(glob.glob(search_path))

        if len(file_list) == 0:
            raise FileNotFoundError(f"in {search_path} couldn't find .npz file！")

        print(f"[{model_type.upper()}] loading {len(file_list)} data files...")

        input_list = []
        target_list = []

        temp_data = np.load(file_list[0], allow_pickle=True)
        w_shape = temp_data['w'].shape
        self.nx = w_shape[0] # 50
        self.ny = w_shape[1] # 50

        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.grid_coords = np.stack([X, Y], axis=-1) # (50, 50, 2)

        for fname in file_list:
            data = np.load(fname, allow_pickle=True)

            base_name = os.path.basename(fname)
            try:
                kn_str = base_name.split('Kn')[1].replace('.npz', '')
                kn_value = float(kn_str)
            except:
                continue

            w = data['w']   # (50, 50, 4)
            q = data['q']   # (50, 50, 2)
            P = data['P']   # (50, 50, 2, 2)
            P_flat = P.reshape(self.nx, self.ny, -1)

            target_sample = np.concatenate([w, P_flat, q], axis=-1)

            kn_map = np.full((self.nx, self.ny, 1), kn_value, dtype=np.float32)

            input_image = np.concatenate([kn_map, self.grid_coords], axis=-1)

            input_list.append(input_image)
            target_list.append(target_sample)

        self.inputs = np.stack(input_list, axis=0)      # (99, 50, 50, 3)
        self.targets = np.stack(target_list, axis=0)    # (99, 50, 50, 10)

        kn_min, kn_max = 0.02, 1.0
        self.inputs[..., 0] = (self.inputs[..., 0] - kn_min) / (kn_max - kn_min)

        self.inputs_tensor = torch.tensor(self.inputs, dtype=torch.float32)
        # Targets stay RAW (physical units) here -- fit_train_normalizer() must be
        # called (after the train/test split) before targets_tensor is usable for
        # training/eval. This avoids baking in full-dataset (train+test) statistics.
        self.targets_tensor = torch.tensor(self.targets, dtype=torch.float32)
        self.target_min = None
        self.target_max = None

        print(f"done!: Input {self.inputs_tensor.shape}, Target {self.targets_tensor.shape} (raw, un-normalized)")

    def save_pt(self, pt_path):
        """Save the processed input/output tensors and stats to one compact .pt file."""
        save_dir = os.path.dirname(pt_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        payload = {
            'inputs': self.inputs_tensor,   # (N, 50, 50, 3) -> [Kn, x, y]
            'targets': self.targets_tensor, # (N, 50, 50, 10) -> [w, P, q] (RAW, physical units)
            'nx': self.nx,
            'ny': self.ny,
        }
        torch.save(payload, pt_path)
        size_mb = os.path.getsize(pt_path) / 1e6
        print(f"saved dataset to {pt_path} ({size_mb:.1f} MB)")

    def _load_pt(self, pt_path):
        """Load the dataset directly from a compact .pt file."""
        print(f"loading dataset from {pt_path} ...")
        payload = torch.load(pt_path, weights_only=False)

        self.inputs_tensor = payload['inputs']
        self.targets_tensor = payload['targets']
        self.nx = int(payload['nx'])
        self.ny = int(payload['ny'])
        self.target_min = None
        self.target_max = None

        if 'target_mean' in payload:
            # Old-format cache: targets were baked in as a full-dataset
            # (train+test) z-score. Un-bake back to raw physical units so
            # fit_train_normalizer() can re-encode with train-only stats.
            mean = payload['target_mean']
            std = payload['target_std']
            self.targets_tensor = self.targets_tensor * std + mean
            print("Old-format cache detected: un-baked full-dataset z-score back to raw units.")

        print(f"done!: Input {self.inputs_tensor.shape}, Target {self.targets_tensor.shape} (raw, un-normalized)")

    def fit_train_normalizer(self, train_indices):
        """Compute per-channel min/max on train split only (targets_tensor must
        be RAW/un-normalized at this point), normalize targets to [0,1] in place."""
        train_indices = torch.as_tensor(train_indices)
        train = self.targets_tensor.index_select(0, train_indices)
        self.target_min = train.amin(dim=(0, 1, 2))
        self.target_max = train.amax(dim=(0, 1, 2))
        rng = self.target_max - self.target_min
        rng = torch.where(rng < 1e-12, torch.ones_like(rng), rng)
        self.targets_tensor = (self.targets_tensor - self.target_min) / rng
        print(f"Train-only min-max applied: {len(train_indices)} train samples.")

    def denormalize(self, tensor, channel_idx=None):
        """Restore [0,1] train-only normalized tensor(s) back to physical values."""
        if self.target_min is None:
            raise RuntimeError("fit_train_normalizer() must be called before denormalize()")
        if channel_idx is None:
            return tensor * (self.target_max - self.target_min) + self.target_min
        return tensor * (self.target_max[channel_idx] - self.target_min[channel_idx]) + self.target_min[channel_idx]

    def __len__(self):
        return len(self.inputs_tensor)

    def __getitem__(self, idx):
        return self.inputs_tensor[idx], self.targets_tensor[idx]


def get_args():
    parser = argparse.ArgumentParser(description="Build a compact .pt dataset from cavity .npz files")
    parser.add_argument('--data_dir', type=str, default='../../data/cavity', help='Directory containing .npz data')
    parser.add_argument('--pt_path', type=str, default='./cavity_dataset.pt', help='Output .pt path')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ds = CavityDataset(data_dir=args.data_dir, pt_path=args.pt_path, use_cache=False)
    x, y = ds[0]
    print(f"Test Shape: Input {x.shape}, Target {y.shape}")
