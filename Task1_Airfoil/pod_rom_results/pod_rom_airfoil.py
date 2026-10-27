"""
Classical POD(r) + RBF reduced-order model for Task1_Airfoil's residual field,
redone under the CURRENT normalization standard (same as every
train_residual_*.py script after today's fixes):

  1. dataset.fit_train_normalizer(train_idx): train-only [0,1] renormalization
     of the base field (upstream ZweigGit-style), replacing the old baked-in
     global [-1,1] encoding.
  2. mask-weighted train-only mean field subtracted -> residual field.
  3. POD (snapshot SVD) of the train residual -> rank-r spatial basis, rank
     chosen via leave-one-out CV on the 80 train cases (no test leakage).
  4. RBF interpolation: branch parameters (674-dim airfoil surface geometry,
     PCA-reduced first -- RBF on raw 674-dim input with only 80 points is
     both unstable and not meaningful) -> r POD mode coefficients.
  5. Test-time reconstruction, fluid-masked rel L2.

This script supersedes the earlier ad-hoc (never saved to disk) Task1 POD-ROM
analysis from before fit_train_normalizer existed -- same POD(r)+RBF method,
just run on the now-current data pipeline for a fair, up-to-date baseline
against the neural train_residual_*.py experiments.

Note: unlike the neural experiments, POD+RBF is a closed-form linear method,
so the specific choice of residual target scaling (global std vs [-1,1]
min-max, as just fixed for the neural scripts) makes NO difference to the
final prediction here -- a linear rescale of the fit target doesn't change
either the SVD basis directions or the RBF interpolant's reconstruction.
What DOES matter (and is applied here) is the upstream fit_train_normalizer
step on the base field itself.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader_fast import AirfoilDatasetFast

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNEL_NAMES = ['rho', 'u', 'v', 'p']
H = W = 128
device = 'cpu'

dataset = AirfoilDatasetFast('../airfoil_unified_128x128.pt', mode='deeponet', device=device)
train_idx, test_idx = dataset.split(train_ratio=0.8, seed=42)
dataset.fit_train_normalizer(train_idx)

target_train = dataset.target_all[train_idx]
target_test = dataset.target_all[test_idx]
mask_train = dataset.geo_mask_flat[train_idx]
mask_test = dataset.geo_mask_flat[test_idx]
branch_train = dataset.branch_all[train_idx].numpy()
branch_test = dataset.branch_all[test_idx].numpy()

mask_sum = mask_train.sum(dim=0).clamp(min=1.0)
mean_field_flat = (target_train * mask_train).sum(dim=0) / mask_sum  # [16384, 4]

resid_train = ((target_train - mean_field_flat.unsqueeze(0)) * mask_train).numpy()
resid_test_true = ((target_test - mean_field_flat.unsqueeze(0)) * mask_test).numpy()

n_train, n_pts, n_ch = resid_train.shape
n_test = resid_test_true.shape[0]
print(f"train={n_train} test={n_test} grid={H}x{W} channels={n_ch} branch_dim={branch_train.shape[1]}")

X_train = resid_train.reshape(n_train, -1)
X_test = resid_test_true.reshape(n_test, -1)


def rel_l2(pred, true):
    num = np.linalg.norm((pred - true).reshape(pred.shape[0], -1), axis=1)
    den = np.linalg.norm(true.reshape(true.shape[0], -1), axis=1) + 1e-12
    return num / den


# --- branch PCA: 674-dim raw airfoil geometry -> a handful of components,
# both for RBF numerical stability and because 80 train points can't support
# a meaningful 674-dim interpolant. ---
K_BRANCH = 15
bmean = branch_train.mean(axis=0)
Ub, Sb, Vtb = np.linalg.svd(branch_train - bmean[None], full_matrices=False)
k_eff = min(K_BRANCH, len(Sb))
bmodes = Vtb[:k_eff]
branch_cumvar = (Sb[:k_eff] ** 2).sum() / (Sb ** 2).sum()
print(f"Branch PCA: {k_eff} components capture {branch_cumvar*100:.2f}% of geometry variance")
branch_feat_train = (branch_train - bmean[None]) @ bmodes.T
branch_feat_test = (branch_test - bmean[None]) @ bmodes.T

# --- POD (snapshot SVD) on full train set ---
U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
r_full = len(S)
energy = S ** 2
cum_energy = np.cumsum(energy) / energy.sum()
print(f"Singular values (first 10 of {r_full}): {S[:10]}")


def pod_rbf_predict(r, feat_fit, coeff_fit, feat_query):
    modes_r = Vt[:r]
    rbf = RBFInterpolator(feat_fit, coeff_fit, kernel='thin_plate_spline', smoothing=0.0)
    coeff_pred = rbf(feat_query)
    resid_pred = coeff_pred @ modes_r
    return resid_pred.reshape(-1, n_pts, n_ch)


# --- leave-one-out CV on the 80 train cases to pick r (no test leakage) ---
loocv_candidates = list(range(2, min(30, n_train - 2) + 1, 2))


def loocv_all_ranks():
    errs = {rr: [] for rr in loocv_candidates}
    idx_all = np.arange(n_train)
    for i in idx_all:
        fold = np.delete(idx_all, i)
        mask_fold = mask_train[fold]
        mask_sum_i = mask_fold.sum(dim=0).clamp(min=1.0)
        mean_i = (target_train[fold] * mask_fold).sum(dim=0) / mask_sum_i
        resid_fold = ((target_train[fold] - mean_i.unsqueeze(0)) * mask_fold).numpy()
        Xi = resid_fold.reshape(len(fold), -1)
        Ui, Si, Vti = np.linalg.svd(Xi, full_matrices=False)

        bmean_i = branch_train[fold].mean(axis=0)
        Ubi, Sbi, Vtbi = np.linalg.svd(branch_train[fold] - bmean_i[None], full_matrices=False)
        k_i = min(K_BRANCH, len(Sbi))
        bmodes_i = Vtbi[:k_i]
        feat_fold = (branch_train[fold] - bmean_i[None]) @ bmodes_i.T
        feat_held = (branch_train[i:i + 1] - bmean_i[None]) @ bmodes_i.T

        held_true = ((target_train[i:i + 1] - mean_i.unsqueeze(0)) * mask_train[i:i + 1]).numpy()

        for rr in loocv_candidates:
            r_eff = min(rr, len(Si))
            modes_i = Vti[:r_eff]
            coeff_i = Xi @ modes_i.T
            rbf_i = RBFInterpolator(feat_fold, coeff_i, kernel='thin_plate_spline', smoothing=0.0)
            coeff_pred_i = rbf_i(feat_held)
            pred_i = (coeff_pred_i @ modes_i).reshape(1, n_pts, n_ch)
            errs[rr].append(rel_l2(pred_i, held_true)[0])
    return {rr: float(np.mean(v)) for rr, v in errs.items()}


loocv_scores = loocv_all_ranks()
r = min(loocv_scores, key=loocv_scores.get)
print(f"LOOCV selects r={r} (LOOCV mean rel2={loocv_scores[r]*100:.3f}%), "
      f"captures {cum_energy[r-1]*100:.4f}% of full-train residual energy")

# --- final model on full train set ---
coeff_train_r = X_train @ Vt[:r].T
coeff_test_true_r = X_test @ Vt[:r].T  # oracle, projection-only diagnostic

resid_test_pred = pod_rbf_predict(r, branch_feat_train, coeff_train_r, branch_feat_test)
resid_test_proj = (coeff_test_true_r @ Vt[:r]).reshape(n_test, n_pts, n_ch)

full_test_pred = resid_test_pred + mean_field_flat.unsqueeze(0).numpy()
full_test_proj = resid_test_proj + mean_field_flat.unsqueeze(0).numpy()
full_test_true = target_test.numpy()
mask_test_np = mask_test.numpy()

rel2_overall = rel_l2(resid_test_pred, resid_test_true)
rel2_proj = rel_l2(resid_test_proj, resid_test_true)

rel2_per_channel = np.zeros((n_test, n_ch))
for c in range(n_ch):
    rel2_per_channel[:, c] = rel_l2(resid_test_pred[:, :, c:c+1], resid_test_true[:, :, c:c+1])

print(f"Test rel L2 (residual, fluid-masked): mean={rel2_overall.mean()*100:.4f}%  "
      f"median={np.median(rel2_overall)*100:.4f}%  min={rel2_overall.min()*100:.4f}%  "
      f"max={rel2_overall.max()*100:.4f}%")
print(f"Projection-only (oracle coeffs) mean rel2={rel2_proj.mean()*100:.4f}%")

with open(os.path.join(OUT_DIR, 'metrics.txt'), 'w') as f:
    f.write(f"Task1_Airfoil RBF POD-ROM, redone under the current fit_train_normalizer "
            f"pipeline (train-only [0,1] base-field normalization).\n")
    f.write(f"r={r} (leave-one-out CV on {n_train} train cases, LOOCV mean rel2="
            f"{loocv_scores[r]*100:.3f}%), captures {cum_energy[r-1]*100:.4f}% of "
            f"full-train residual energy.\n")
    f.write(f"Branch PCA: {k_eff} components capture {branch_cumvar*100:.2f}% of the "
            f"674-dim airfoil geometry variance (RBF fit on these, not raw 674-dim).\n")
    f.write(f"train={n_train}  test={n_test}  grid={H}x{W}  channels={n_ch}\n\n")
    f.write("Test rel L2 (residual, fluid-masked, full RBF POD-ROM):\n")
    f.write(f"  mean={rel2_overall.mean()*100:.4f}%  median={np.median(rel2_overall)*100:.4f}%  "
            f"std={rel2_overall.std()*100:.4f}%  min={rel2_overall.min()*100:.4f}%  max={rel2_overall.max()*100:.4f}%\n\n")
    f.write("Test rel L2 (residual, POD projection only, oracle coefficients -- upper bound on RBF):\n")
    f.write(f"  mean={rel2_proj.mean()*100:.4f}%  median={np.median(rel2_proj)*100:.4f}%\n\n")
    f.write("Leave-one-out CV rank sweep (train-only, no test leakage):\n")
    for rr in loocv_candidates:
        marker = '  <- selected' if rr == r else ''
        f.write(f"  r={rr:3d}: LOOCV mean rel2={loocv_scores[rr]*100:7.3f}%{marker}\n")
    f.write("\nPer-channel RBF POD-ROM residual rel L2 (mean over test cases):\n")
    for c in range(n_ch):
        f.write(f"  {CHANNEL_NAMES[c]:>5s}: {rel2_per_channel[:, c].mean()*100:.4f}%\n")
    f.write("\nNote: unlike the neural train_residual_*.py scripts, this closed-form "
            "POD+RBF method is mathematically invariant to the residual target's linear "
            "scaling (global std vs [-1,1] min-max) -- the only pipeline change that "
            "actually affects these numbers vs. the old (pre-10/26) POD-ROM analysis is "
            "fit_train_normalizer on the base field.\n")

np.savez(os.path.join(OUT_DIR, 'pod_rom_predictions.npz'),
         resid_test_pred=resid_test_pred, resid_test_true=resid_test_true,
         full_test_pred=full_test_pred, full_test_true=full_test_true, mask_test=mask_test_np,
         rel2_overall=rel2_overall, rel2_per_channel=rel2_per_channel, rel2_projection_only=rel2_proj,
         modes=Vt[:r], mean_field_flat=mean_field_flat.numpy(), S=S)

# --- plots ---
plt.figure(figsize=(6, 4))
plt.semilogy(np.arange(1, r_full + 1), S, 'o-', ms=3)
plt.axvline(r, color='r', ls='--', label=f'selected r={r}')
plt.xlabel('POD mode index'); plt.ylabel('singular value')
plt.title('Task1_Airfoil train-residual POD spectrum (current normalization)')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'singular_value_spectrum.png'), dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
rs_sorted = sorted(loocv_scores)
plt.plot(rs_sorted, [loocv_scores[rr] * 100 for rr in rs_sorted], 'o-')
plt.axvline(r, color='r', ls='--', label=f'selected r={r}')
plt.xlabel('POD rank r'); plt.ylabel('LOOCV mean rel L2 (%)')
plt.title('Task1_Airfoil: leave-one-out CV rank selection (train only)')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'loocv_rank_selection.png'), dpi=150)
plt.close()

order_err = np.argsort(rel2_overall)
show_idx = {'best': order_err[0], 'median': order_err[len(order_err) // 2], 'worst': order_err[-1]}
for tag, i in show_idx.items():
    m2d = mask_test_np[i].reshape(H, W)
    fig, axes = plt.subplots(n_ch, 3, figsize=(12, 2.6 * n_ch))
    for c in range(n_ch):
        true_c = np.where(m2d[..., 0] == 1, resid_test_true[i, :, c].reshape(H, W), np.nan)
        pred_c = np.where(m2d[..., 0] == 1, resid_test_pred[i, :, c].reshape(H, W), np.nan)
        err_c = pred_c - true_c
        vmax = np.nanmax(np.abs(true_c))
        emax = np.nanmax(np.abs(err_c))
        for ax, data, title, cmap, vlim in [
            (axes[c, 0], true_c, f'{CHANNEL_NAMES[c]} true residual', 'coolwarm', (-vmax, vmax)),
            (axes[c, 1], pred_c, f'{CHANNEL_NAMES[c]} pred (RBF POD-ROM)', 'coolwarm', (-vmax, vmax)),
            (axes[c, 2], err_c, f'{CHANNEL_NAMES[c]} error', 'coolwarm', (-emax, emax)),
        ]:
            im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1], origin='lower')
            ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Task1_Airfoil POD-ROM, {tag} case (test idx={i}), rel2={rel2_overall[i]*100:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'airfoil_pod_rom_{tag}_case{i}.png'), dpi=140)
    plt.close()

print(f"Saved results to {OUT_DIR}")
