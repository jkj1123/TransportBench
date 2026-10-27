"""
Standalone runner for the hyper_mscale_deeponet_gelu ablation (trunk B-spline
activation swapped for GELU). Mirrors run_epoch_benchmark.py's directory
conventions and results.csv schema exactly, so this new model's rows sit
alongside the main 5-model x 6-epoch benchmark for direct comparison.

Written as a separate script (instead of editing run_epoch_benchmark.py's
MODELS list) so it doesn't disturb the main driver, which may still be
mid-run on its own long-running combo.
"""
import os
import sys
import csv
import json
import time
import subprocess

MODEL = 'hyper_mscale_deeponet_gelu'
EPOCH_BUDGETS = [1000, 2500, 10000, 25000, 50000, 100000]

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'epoch_benchmark')
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints')
IMG_DIR = os.path.join(OUT_DIR, 'images')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
RESULTS_CSV = os.path.join(OUT_DIR, 'results.csv')

PYTHON = sys.executable


def already_done(epochs):
    tag = f"{MODEL}_{epochs}"
    return os.path.exists(os.path.join(IMG_DIR, f"{tag}_result.json"))


def run_one(epochs):
    tag = f"{MODEL}_{epochs}"
    ckpt_dir = os.path.join(CKPT_DIR, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"best_model_{MODEL}.pth")

    t0 = time.time()
    print(f"\n=== [{tag}] training ({epochs} epochs) ===", flush=True)
    train_log_path = os.path.join(LOG_DIR, f"{tag}_train.log")
    with open(train_log_path, 'w') as logf:
        proc = subprocess.run(
            [PYTHON, 'train.py', '--model', MODEL, '--epochs', str(epochs), '--save_dir', ckpt_dir],
            cwd=BASE, stdout=logf, stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        print(f"[{tag}] TRAIN FAILED (see {train_log_path})", flush=True)
        return None
    train_time = time.time() - t0

    print(f"=== [{tag}] evaluating ===", flush=True)
    eval_log_path = os.path.join(LOG_DIR, f"{tag}_eval.log")
    with open(eval_log_path, 'w') as logf:
        proc = subprocess.run(
            [PYTHON, 'eval_epoch.py', '--model', MODEL, '--checkpoint', ckpt_path,
             '--out_dir', IMG_DIR, '--tag', tag],
            cwd=BASE, stdout=logf, stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        print(f"[{tag}] EVAL FAILED (see {eval_log_path})", flush=True)
        return None

    with open(os.path.join(IMG_DIR, f"{tag}_result.json")) as f:
        result = json.load(f)
    result['train_time_sec'] = round(train_time, 1)
    result['epochs'] = epochs

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'epochs', 'rel_l2_overall',
                                                'rel_l2_P0', 'rel_l2_q0', 'mae_overall',
                                                'train_time_sec'])
        if write_header:
            writer.writeheader()
        writer.writerow({k: result[k] for k in writer.fieldnames})

    print(f"[{tag}] done in {train_time:.1f}s | Rel L2 = {result['rel_l2_overall']*100:.3f}%", flush=True)
    return result


def main():
    for epochs in EPOCH_BUDGETS:
        tag = f"{MODEL}_{epochs}"
        if already_done(epochs):
            print(f"{tag}: already done, skipping", flush=True)
            continue
        run_one(epochs)
    print("\nGELU ablation complete.", flush=True)


if __name__ == "__main__":
    main()
