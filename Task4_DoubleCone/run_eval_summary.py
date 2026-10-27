"""
Run eval.py (benchmark case = global idx 50, fixed --sample_idx default) and
eval_idx3.py (benchmark case = global idx 3, auto-selected as the
highest-temperature TEST-set sample) for every model checkpoint found in
./checkpoints/, and collect everything into one output folder:

  eval_summary/
    idx50_images/   <- eval.py's inference_<model>_fourier_benchmark.png
    idx3_images/    <- eval_idx3.py's inference_<model>_fourier_benchmark.png
    logs/           <- raw stdout of every run, for reference
    errors_idx50.csv
    errors_idx3.csv

Models are auto-discovered from checkpoints/best_model_{model}_fourier.pth.
"""
import os
import re
import csv
import glob
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE, 'checkpoints')
DATA_PATH = os.path.join(BASE, 'double_cone.pt')

OUT_DIR = os.path.join(BASE, 'eval_summary')
IDX50_IMG_DIR = os.path.join(OUT_DIR, 'idx50_images')
IDX3_IMG_DIR = os.path.join(OUT_DIR, 'idx3_images')
LOG_DIR = os.path.join(OUT_DIR, 'logs')

PYTHON = sys.executable

CHANNEL_RE = re.compile(r'^\s*(rho|u|v|p):\s*([\d.]+)%')
MEAN_RE = re.compile(r'per-channel mean:\s*([\d.]+)%')
ENC_L1_RE = re.compile(r'encoded L1 \(Test-Pure\)\s*:\s*([\d.]+)')
PAPER_L2_RE = re.compile(r'physical Relative L2 \(all-variable aggregate\)\s*:\s*([\d.]+)')


def discover_models():
    models = []
    for p in sorted(glob.glob(os.path.join(CKPT_DIR, 'best_model_*_fourier.pth'))):
        name = os.path.basename(p)[len('best_model_'):-len('_fourier.pth')]
        models.append(name)
    return models


def parse_channel_errors(stdout):
    rows = {}
    for line in stdout.splitlines():
        m = CHANNEL_RE.match(line)
        if m:
            rows[m.group(1)] = float(m.group(2))
        m = MEAN_RE.search(line)
        if m:
            rows['per_channel_mean'] = float(m.group(1))
    return rows


def run_script(script, model, img_dir):
    log_path = os.path.join(LOG_DIR, f"{script.replace('.py','')}_{model}.log")
    proc = subprocess.run(
        [PYTHON, script, '--model', model, '--data_path', DATA_PATH, '--output_dir', img_dir],
        cwd=BASE, capture_output=True, text=True,
    )
    stdout = proc.stdout + '\n' + proc.stderr
    with open(log_path, 'w') as f:
        f.write(stdout)
    return proc.returncode, stdout


def main():
    os.makedirs(IDX50_IMG_DIR, exist_ok=True)
    os.makedirs(IDX3_IMG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    models = discover_models()
    print(f"Discovered models in checkpoints/: {models}")

    rows_idx50, rows_idx3 = [], []

    for model in models:
        print(f"\n=== {model} : eval.py (idx 50) ===", flush=True)
        rc, out = run_script('eval.py', model, IDX50_IMG_DIR)
        if rc != 0:
            print(f"  FAILED (see logs/eval.py_{model}.log)")
        else:
            row = {'model': model, **parse_channel_errors(out)}
            rows_idx50.append(row)
            print(f"  {row}")

        print(f"=== {model} : eval_idx3.py (idx 3, auto-selected) ===", flush=True)
        rc, out = run_script('eval_idx3.py', model, IDX3_IMG_DIR)
        if rc != 0:
            print(f"  FAILED (see logs/eval_idx3.py_{model}.log)")
        else:
            row = {'model': model, **parse_channel_errors(out)}
            m = ENC_L1_RE.search(out)
            if m:
                row['encoded_L1_test_pure'] = float(m.group(1))
            m = PAPER_L2_RE.search(out)
            if m:
                row['paper_eq17_rel_l2'] = float(m.group(1))
            rows_idx3.append(row)
            print(f"  {row}")

    fields50 = ['model', 'rho', 'u', 'v', 'p', 'per_channel_mean']
    with open(os.path.join(OUT_DIR, 'errors_idx50.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields50)
        w.writeheader()
        for r in rows_idx50:
            w.writerow({k: r.get(k, '') for k in fields50})

    fields3 = ['model', 'rho', 'u', 'v', 'p', 'per_channel_mean', 'encoded_L1_test_pure', 'paper_eq17_rel_l2']
    with open(os.path.join(OUT_DIR, 'errors_idx3.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields3)
        w.writeheader()
        for r in rows_idx3:
            w.writerow({k: r.get(k, '') for k in fields3})

    print(f"\nDone. Summary written to {OUT_DIR}/errors_idx50.csv and errors_idx3.csv")


if __name__ == "__main__":
    main()
