"""
Sequential epoch-budget benchmark driver for Task III (Cavity Flow).

For each (model, epoch_budget) pair, runs a fresh `train.py` job (own cosine
LR schedule sized to that epoch budget), then evaluates the resulting
best-checkpoint with `eval_epoch.py` (Rel L2 error + P0/q0 plots).

Runs strictly ONE subprocess at a time (no parallelism) to keep memory/CPU
footprint low on a shared server, and appends to results.csv after every
single run so partial progress is visible even if interrupted.
"""
import os
import sys
import csv
import json
import time
import subprocess

MODELS = ['deeponet', 'hyperdeeponet', 'mscale_deeponet', 'hyper_mscale_deeponet', 'fusion_deeponet']  # fastest -> slowest
EPOCH_BUDGETS = [1000, 2500, 10000, 25000, 50000, 100000]

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'epoch_benchmark')
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints')
IMG_DIR = os.path.join(OUT_DIR, 'images')
LOG_DIR = os.path.join(OUT_DIR, 'logs')
RESULTS_CSV = os.path.join(OUT_DIR, 'results.csv')

PYTHON = sys.executable


def already_done(model, epochs):
    tag = f"{model}_{epochs}"
    result_json = os.path.join(IMG_DIR, f"{tag}_result.json")
    return os.path.exists(result_json)


def run_one(model, epochs):
    tag = f"{model}_{epochs}"
    ckpt_dir = os.path.join(CKPT_DIR, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"best_model_{model}.pth")

    t0 = time.time()
    print(f"\n=== [{tag}] training ({epochs} epochs) ===", flush=True)
    train_log_path = os.path.join(LOG_DIR, f"{tag}_train.log")
    with open(train_log_path, 'w') as logf:
        proc = subprocess.run(
            [PYTHON, 'train.py', '--model', model, '--epochs', str(epochs),
             '--save_dir', ckpt_dir],
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
            [PYTHON, 'eval_epoch.py', '--model', model, '--checkpoint', ckpt_path,
             '--out_dir', IMG_DIR, '--tag', tag],
            cwd=BASE, stdout=logf, stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        print(f"[{tag}] EVAL FAILED (see {eval_log_path})", flush=True)
        return None

    result_json = os.path.join(IMG_DIR, f"{tag}_result.json")
    with open(result_json) as f:
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
    os.makedirs(OUT_DIR, exist_ok=True)
    total = len(MODELS) * len(EPOCH_BUDGETS)
    done = 0
    for model in MODELS:
        for epochs in EPOCH_BUDGETS:
            done += 1
            tag = f"{model}_{epochs}"
            if already_done(model, epochs):
                print(f"[{done}/{total}] {tag}: already done, skipping", flush=True)
                continue
            print(f"[{done}/{total}] {tag}", flush=True)
            run_one(model, epochs)

    print("\nAll runs complete.", flush=True)


if __name__ == "__main__":
    main()
