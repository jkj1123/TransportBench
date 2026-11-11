# Task1_Airfoil: mean-subtracted residual learning — why it fails to train

Summary of a diagnostic session investigating why neural models (DeepONet, LNLR)
trained on Task1_Airfoil's mean-subtracted residual field collapse to predicting
an input-independent constant (rel2 ≈ 1.0, i.e. no better than "predict zero
residual everywhere") when trained on the full 80-sample train split.

## Setup

- Target: `residual = flow_field - train_mean_field` (train-only [0,1]
  renormalized field minus the fluid-weighted mean over the 80 train samples).
- Per-channel `[-1, 1]` min-max normalization of the residual (train-only
  stats), to avoid the heavy-tailed-target failure mode this was originally
  introduced to fix (see `train_residual_deeponet.py` docstring/comments).
- Branch input: flattened airfoil surface point cloud, raw 674-dim
  (337 points × 2 coords).
- Metric: masked relative L2 error on the physical-scale residual (`rel2`).

## Result 1 — every full-dataset (80-sample) config collapses to rel2 ≈ 1.0

| Config | Model | Params | Epochs | Result |
|---|---|---|---|---|
| Original (baseline recipe) | LNLR, old arch | 100K | 5000 | rel2 ≈ 1.0 |
| New arch, s=32 | LNLR | 200K | 5000 | rel2 ≈ 1.0 |
| New arch, s=12 | LNLR | 200K | 5000 | rel2 ≈ 1.0 |
| lr=3e-5, no decay | DeepONet | 100K | 5000 | rel2 ≈ 1.0 (best 0.999967 @ ep1456) |
| Gaussian-smoothed residual (σ=1.5) | DeepONet | 50K | 5000 | rel2 best 0.990 @ ep51, worse (0.998) by ep5000 |
| PCA(14)-reduced branch (674→14, 100% variance) | DeepONet | 50K | 5000 | rel2 ≈ 1.0 (0.999988) |
| PCA(4)-reduced branch (674→4, 88% variance) | DeepONet | 50K | 5000 | rel2 ≈ 1.0 (0.999988) |
| +0.1 constant offset on entire residual field | DeepONet | 50K | 500 | rel2(raw)=8.4% **looks good but is fake** — rel2(offset subtracted back out) = 1.00007 |
| PCA(4) branch **+** 0.1 offset (combined) | DeepONet | 50K | 500 | Same: rel2(true_signal) = 1.00006 |

**None of these interventions fixed it.** Ruled out: noise in the target,
geometry input dimensionality, the specific value 0 being privileged
(weight_decay / activation symmetry).

## Result 2 — the signal is real and learnable (ruling out "no signal" hypothesis)

### Classical POD(r)+RBF ROM (closed-form, no gradient descent)
- Current repo script (`pod_rom_results/pod_rom_airfoil.py`, LOOCV over even
  ranks only): best r=2, LOOCV rel2 = 91.3%; full test rel2 mean = 95.0%
  (pointwise rel2 is dominated by per-pixel DSMC sampling noise).
- **Visually**, predicted residual clearly reconstructs the shock-front V-line
  at the correct location/sign in every case inspected (median, worst, and
  weak-shock test cases) — the raw rel2 number is misleading because it's
  swamped by background pixel noise that no smooth method can match.
- An earlier (session-referenced, not re-derivable from current repo state)
  analysis found **POD(r=7)+RBF reconstructs shock position to ~1px accuracy**,
  with a signal-strength-dependent amplitude bias (~20–25% underestimate on
  weak-shock cases, confirmed noise-robust via windowed integration). This is
  a much more favorable result than raw rel2 suggests.

### Single-sample overfit (no generalization pressure)
- `train_residual_deeponet_overfit1.py`: train+eval on exactly ONE sample's
  real (non-degenerate) residual.
  - wd=1e-4, 200 epochs: rel2 = 0.9999 (looks stuck)
  - **wd=0, 2000 epochs: rel2 = 0.9245** — genuinely still improving, just
    very slowly (not a hard local minimum).
  - Visualization confirms the predicted residual captures the real
    shock-front shape/location, not just noise — model IS learning real
    structure, just slowly (**spectral bias**: smooth/compact near-body
    correction learned first, the thin high-frequency shock-front line
    learned much more slowly — matches TransportBench paper §3.1/§6.4.2).
- Smoothed residual + single-sample overfit (500 epochs, wd=0): rel2 = 0.9436
  vs 0.9564 unsmoothed at the same epoch — smoothing helps marginally, not
  fundamentally.

## Result 3 — the collapse is about sample COUNT, not input dimension or noise

`train_residual_deeponet_nsamples.py`: train on exactly N distinct train-split
geometries simultaneously (single full-batch per epoch, no mini-batching),
2000 epochs, wd=0, hidden_dim=41 (~50K params).

| N | mean rel2 | std across samples | interpretation |
|---|---|---|---|
| 1 | 0.925 | – | learns real per-sample structure (slowly) |
| 2 | 0.961 | 0.018 | still differentiates between geometries |
| 5 | 0.979 | 0.028 | still differentiates (range 0.93–1.01) |
| 10 | 0.967 | 0.030 | still differentiates (range 0.91–1.01) |
| **20** | 0.997 | **0.003** | **collapsed** — all samples predict ~same value |
| **40** | 0.998 | **0.003** | **collapsed** |

Clear transition between N=10 and N=20: std drops ~10x, meaning the branch
net stops conditioning on geometry at all and just predicts a near-constant
output regardless of input. This reproduces the full-80-sample collapse at a
much smaller N, and does not depend on the branch input dimension (same
collapse pattern seen with 674-dim and PCA-reduced inputs) or on the specific
target value (same collapse pattern seen with the 0.1-shifted target).

**Caveat**: this N-sweep used full-batch (batch_size = N, one gradient step
per epoch), while the full-dataset experiments above used `batch_size=16`
(5 gradient steps/epoch over 80 samples). Whether the collapse is driven by
*effective batch size per gradient step* or by *total number of distinct
samples the branch must learn to discriminate between* is not yet
disambiguated — an open next step (e.g. re-run N=40/80 with batch_size=16).

## Working hypothesis

The failure is **not**:
- noise in the residual target,
- the geometry branch input's dimensionality (674 vs 14 vs 4 — all collapse),
- the residual being centered on 0 specifically (shifting the whole target
  by +0.1 just moves the collapse point to 0.1, doesn't prevent it),
- a total absence of learnable signal (POD-ROM and 1–10 sample training both
  extract real, geometry-dependent shock structure).

It **is** most consistent with: once the branch net has to discriminate
between roughly 20+ different geometries within the training dynamics,
gradient descent finds it easier to settle into an input-independent
"predict the same (near-)constant for everyone" solution than to actually
differentiate per-geometry corrections — a branch-conditioning collapse
specific to this multi-sample optimization regime, compounded by ordinary
spectral bias (the thin shock-front line is a high-frequency feature that's
slow to learn even in the best case).

## Open next steps (not yet run)

1. Re-run the N-sweep with a fixed mini-batch size (e.g. batch_size=16) as N
   grows past 20–80, to separate "distinct geometries in the dataset" from
   "samples averaged per gradient step" as the driver.
2. Try the `LNLR_UdiagS`-style fix validated on Task2_Cylinder (single fixed
   shared matrix + branch-only diagonal/FiLM-style gating, no full per-sample
   weight generation) on Task1's branch-trunk architecture directly.
3. Re-evaluate with a position/windowed-amplitude metric (like the earlier
   POD r=7 analysis) instead of raw pointwise rel2, which is dominated by
   irreducible per-pixel noise and may be underselling partially-successful
   models.
