# Upstream changes (ZweigGit/TransportBench) not yet in this local checkout

Reviewed on 2026-10-25 by diffing the local `TransportBench/` tree against
`https://github.com/ZweigGit/TransportBench` (branch `main`). Three
independent kinds of change were found, spread across Task1-4. This file is
the reference for applying them locally.

## 1. Train-only normalization (Task1, Task2, Task3, Task4)

**Root commit: `64f10d4d` (2026-08-31) "Unify all tasks to train-only
min-max normalization [0,1]"**

- **Task1/Task2**: data is stored pre-baked to `[-1,1]` using **global**
  (all-sample, train+test) min/max. A new `Dataset.fit_train_normalizer(train_indices)`
  method un-does that bake-in and re-encodes to `[0,1]` using **train-only**
  min/max, called once from `train.py`/`eval.py` right after the
  `random_split`. Identical function body in both tasks.
- **Task3**: local `CavityDataset` z-score-normalizes targets at load time
  using **all 99 samples** (`np.mean(self.targets, ...)`) — this is test-set
  leakage. Upstream removes that step entirely and adds the same
  `fit_train_normalizer(train_indices)` idea, but as **train-only min-max**
  (not z-score), called after the split.
- **Task4**: local `get_dataloader_and_stats` computes `GaussianNormalizer`
  stats (`x_mean/x_std/y_mean/y_std`) from **all samples including test**
  (`torch.mean(x_data, ...)`) — same leakage bug. Upstream computes
  `MinMaxNormalizer` stats from `x_data[train_idx]`/`y_data[train_idx]` only.

**⚠️ Important side effect for Task1/Task2 specifically**: the
`[-1,1] -> train-only [0,1]` step is `v2 = a*z + b` with a non-zero shift
`b`. Rel-L2 is invariant to pure scaling but **not to an additive shift**:
the error norm scales by `a` (~0.5), but the shift inflates `||true||` by
less than `a` (measured: `a≈0.50`, but `||true||` only shrinks to `≈0.87`x
of its old value) — so reported rel-L2 drops by roughly **half**, with zero
change in actual model quality. Verified numerically this session: our own
Residual-Fusion-DeepONet result (2.637%, 1M params, Task1) projects to
≈1.51% under this transform, matching an externally-reported 1.77% for the
same config. **Task3/Task4 do not have this artifact** (no pre-existing
`[-1,1]` bake-in to un-shift from) — their normalization change is a
straight leakage fix, not a metric-shift.

## 2. Masking (Task2, Task4)

- **Task2** (`6fc08a00`, `bcd21486`, 2026-10-11/12): adds
  `dataset.fluid_mask` (`[H*W]` bool, cylinder-interior cells excluded).
  Applied to **train loss AND eval metrics**, first for DeepONet-family
  models only, then extended to grid models (FNO/UNet/ViT/AE) too. Verified
  this session with real checkpoints: DeepONet -0.69pp, HyperDeepONet
  -0.24pp, MscaleDeepONet -0.26pp; Fusion_DeepONet ~unchanged (it already
  predicts ~0 in the interior on its own).
- **Task1**: already masks with `geo_mask` everywhere (no change to rel-L2);
  only the eval.py MAE/MSE **denominator** changed to a fluid-point count
  instead of the full grid size. Airfoil solid fraction is ~0.19% of the
  grid, so this shift is negligible (~0.2%).
- **Task4** (`48a3bfb7` merge, `c17d7006` fix, 2026-09-29): adds
  `vacuum_mask(y_data)` — `[H,W]` bool, True where pressure is exactly 0 in
  every sample (body-interior wedge + outflow-boundary mesh filler cells).
  Replaces an earlier ad-hoc per-batch threshold (`y[:,3] > -5.5` in
  log-pressure space) with one canonical mask shared by train loss, eval
  metrics, and visualization.

## 3. Task4 loss function and curriculum-weight normalization

- **Loss type**: `L1Loss` → briefly `MSELoss` for `wavelet_deeponet` only
  (`92e884ea`-adjacent work) → reverted to `L1Loss` for everyone
  (`73ac1a9f`, "L1's constant gradient let rare cells drift decades in
  log-p space") → finally **`MSELoss` for every model** (`92e884ea`,
  2026-10-05). Current upstream state: `loss_fn = nn.MSELoss(reduction='none')`.
- **Curriculum-weight normalization bug fix**: the grid-model loss applies
  a spatial weight matrix `w` (`w_wall` on the wall row, `w_near` on
  near-wall rows, `w_p` extra weight on the pressure channel, `0` on vacuum
  cells). Before `92e884ea`, this was normalized as `(raw_loss * w).mean()`
  — dividing by the **fixed element count**, not by `w.sum()`. Coordinate
  models (DeepONet family) used a *different* branch entirely
  (`(loss*valid_w).sum() / (valid_n*4)`, no `w_wall`/`w_near`/`w_p` at all —
  the old code says so explicitly: *"grid-position curriculum weights don't
  apply in flat space"*). **So coordinate-based models never received the
  wall/near-wall curriculum weighting at all**, while grid models did, and
  even grid models' effective wall/near-wall emphasis drifted with however
  many vacuum cells happened to be zeroed out (since `.mean()`'s divisor
  doesn't track `w`'s actual mass).
  `92e884ea` fixes both problems at once: every coordinate model is now
  wrapped in a `GridAdapter` (presents a `[B,5,H,W]` interface) so **all**
  models — coordinate and grid alike — go through the identical
  `loss = (raw_loss * w).sum() / w.sum()` path. This is a true weighted
  mean, so the wall-vs-bulk emphasis ratio is now (a) actually equal to
  `w_wall`/`w_near` as configured, regardless of vacuum-cell count, and
  (b) identical across every model in the zoo.

## Two bugs found in upstream while porting (fixed locally, not matching upstream exactly)

1. **Task2 `fluid_mask` direction is inverted from its own stated intent.**
   Upstream: `self.fluid_mask = (self.mask[:, 0] >= 0.5).all(dim=0)`, commented
   "True where every sample marks fluid". Verified directly against
   `cylinder_full_2400.pt`: `mask > 0.5` is only 1.53% of pixels and is
   **exactly 0 with zero cross-sample variance** there (the signature of the
   non-physical cylinder-interior filler); real, varying CFD values live in
   `mask <= 0.5` (98.47%). So upstream's condition selects the *cylinder
   interior*, not the fluid region. **Fixed locally to `<= 0.5`.**
2. **Task4 `vacuum_mask()` is called after the log10 pressure transform, so
   it always finds zero vacuum cells.** `vacuum_mask` checks
   `(y_data[:,3] == 0).all(dim=0)`, but upstream's `get_dataloader_and_stats`
   calls it on `y_data` *after* `y_data[:,3] = log10(y_data[:,3] + 1e-6)` is
   applied in place — and `log10(0 + 1e-6) = -6`, never `0`. Verified: on
   `double_cone.pt`, the RAW-pressure mask finds 313/6528 cells (4.8%); the
   log-transformed-pressure mask finds 0. **Fixed locally by computing
   `vacuum_mask(y_data)` before the log10 transform line**, in both
   `data_loader.py` and (duplicated split logic) `eval.py`/`eval_idx3.py`.

## What was actually applied locally (2026-10-25)

| File | Change |
|---|---|
| `Task1_Airfoil/data_loader_fast.py` | added `fit_train_normalizer()` + `denormalize()` to `AirfoilDatasetFast` |
| `Task1_Airfoil/data_loader.py` | added `fit_train_normalizer()`, `denormalize()` updated to train-only |
| `Task1_Airfoil/train_fast.py` | calls `fit_train_normalizer()` after split |
| `Task1_Airfoil/train.py` | calls `fit_train_normalizer()` after split |
| `Task1_Airfoil/eval.py` | calls `fit_train_normalizer()`; MAE/MSE denominator now fluid-point-only |
| `Task2_Cylinder/data_loader_fast.py` | added `fluid_mask` (direction fixed, see above) + `fit_train_normalizer()` + `denormalize()` |
| `Task2_Cylinder/data_loader.py` | same additions as the fast loader |
| `Task2_Cylinder/train_fast.py` | fluid-masks loss/rel2 for both `grid` and `deeponet` modes; calls `fit_train_normalizer()` |
| `Task2_Cylinder/train.py` | same as train_fast.py |
| `Task2_Cylinder/eval.py` | fluid-masks MAE/MSE/rel-L2; MAE/MSE denominator fluid-point-only |
| `Task3_Cavity/data_loader.py` | removed full-dataset z-score at load time; `targets_tensor` now raw until `fit_train_normalizer()`; old-format `.pt` cache auto-detected and un-baked |
| `Task3_Cavity/train.py` | calls `fit_train_normalizer()` after split |
| `Task4_DoubleCone/data_loader.py` | `GaussianNormalizer`→`MinMaxNormalizer`, train-only stats, added `vacuum_mask()` (log10-order bug fixed), `get_split_indices()` factored out |
| `Task4_DoubleCone/train.py` | loss `L1`→`MSE`; `w[:,:,vac]=0`; `(raw_loss*w).mean()` → `(raw_loss*w).sum()/w.sum()`; vacuum-masked test loss |
| `Task4_DoubleCone/eval.py`, `eval_idx3.py` | `MinMaxNormalizer` + train-only stats (previously recomputed **full-dataset** Gaussian stats independently — its own leakage bug, now fixed); vacuum-masked rel-L2 |
| `Task4_DoubleCone/train_1.py`, `train_2.py`, `train_mae_mse.py` | **compatibility shim only** (fixed 4→5 tuple unpack so they don't crash) — still use `L1Loss`, no vacuum masking, no curriculum-weight-sum fix. Use `train.py` instead; these are not fully updated. |

**Not touched**: Task4's coordinate/grid `GridAdapter` unification (section 3
of this doc) was **not ported** — it doesn't apply locally. Local Task4
models (`model_hyperdeeponet.py` etc.) already take a single grid tensor
`forward(self, x)` directly (branch/trunk mechanics handled inside the model
class), so local `train.py` only ever had ONE loss code path to begin with.
The upstream "coordinate models never got wall-weighting" bug never existed
in this codebase's architecture.

## ⚠️ Critical: existing checkpoints are now invalid

Every one of these normalization changes alters what the model's output is
trained to match. **A checkpoint trained before this change, evaluated with
the updated eval script, will show garbage numbers** (verified: a real
Task1 HyperDeepONet checkpoint showed rel-L2 = 113.9% post-change, vs ~2.7%
pre-change) — not because anything is broken, but because the old weights
were fit to the old `[-1,1]`/Gaussian target space and the eval script now
compares them against a different (train-only `[0,1]`/MinMax) target space.
**Every model must be retrained from scratch** under the new code before its
numbers mean anything again.

## Net effect on reported numbers if applied

| Task | Mechanism | Expected effect |
|---|---|---|
| 1 | train-only `[0,1]` renorm | rel-L2 drops ~40-50% (metric artifact, not real) |
| 1 | eval MAE/MSE denominator | negligible (~0.2%) |
| 2 | train-only `[0,1]` renorm | same metric-shift as Task1 |
| 2 | fluid masking | real improvement, model-dependent (0-13%) |
| 3 | leakage fix (z-score→train-only min-max) | real fix, direction unpredictable without rerun |
| 4 | leakage fix (Gaussian→train-only MinMax) | real fix |
| 4 | vacuum masking | real fix (removes non-physical filler cells from loss/eval) |
| 4 | L1→MSE | changes convergence behavior, not directly comparable to old L1 numbers |
| 4 | curriculum-weight unification | coordinate models now get wall-weighting they never had; grid models' wall emphasis becomes consistent/controlled |
