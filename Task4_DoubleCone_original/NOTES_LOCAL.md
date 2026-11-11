# Local notes (not part of upstream)

This folder is a byte-for-byte copy of `Task4_DoubleCone/` from
https://github.com/CFDML/TransportBench (cloned 2026-08-31), plus three
additions needed to actually run it, none of which modify an original file:

- `double_cone.pt` -- the dataset. Not tracked in the git repo (weights/data
  are hosted externally per the repo's own README); copied over from the
  existing `Task4_DoubleCone/double_cone.pt` in this workspace.
- `data_utils.py` -- a one-line compatibility shim. Upstream's `train.py`
  imports `data_utils.get_dataloader_and_stats`, but the repo only ships
  `data_loader.py` (this looks like an upstream packaging bug). The shim just
  re-exports from `data_loader.py` so `train.py` runs unmodified.
- `eval_extra_metrics.py` -- copied from the sibling `Task4_DoubleCone/`
  folder (the same script used for its 2500-epoch/no-fourier retrain). It
  computes physical-space relative-L2 error averaged over the full 6-sample
  test set (upstream's own `eval.py` only evaluates one visualization sample,
  idx 50). Depends only on the model_*.py files, which are unmodified here.

Everything else (`README.md`, `.gitignore`, `train.py`, `eval.py`,
`data_loader.py`, `model_*.py`) is untouched upstream content.
