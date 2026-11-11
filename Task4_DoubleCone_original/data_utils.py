"""
Compatibility shim -- NOT part of the upstream CFDML/TransportBench repo.

train.py does `from data_utils import get_dataloader_and_stats`, but the
upstream repo only ships `data_loader.py` (which defines the same function --
see its own docstring: "Compatible with original data_utils.py logic"). This
one-line re-export lets the unmodified upstream train.py run as-is, without
touching any of the original files.
"""
from data_loader import get_dataloader_and_stats, GaussianNormalizer  # noqa: F401
