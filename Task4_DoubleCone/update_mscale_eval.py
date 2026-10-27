"""
Waits for the mscale_deeponet retraining process (given PID) to exit, then
re-runs eval.py and eval_idx3.py for mscale_deeponet only and patches its
row in eval_summary/errors_idx50.csv / errors_idx3.csv + its two images,
leaving every other model's results untouched.
"""
import os
import re
import csv
import sys
import time
import subprocess

TRAIN_PID = int(sys.argv[1])
MODEL = 'mscale_deeponet'

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'double_cone.pt')
OUT_DIR = os.path.join(BASE, 'eval_summary')
IDX50_IMG_DIR = os.path.join(OUT_DIR, 'idx50_images')
IDX3_IMG_DIR = os.path.join(OUT_DIR, 'idx3_images')
LOG_DIR = os.path.join(OUT_DIR, 'logs')

CHANNEL_RE = re.compile(r'^\s*(rho|u|v|p):\s*([\d.]+)%')
MEAN_RE = re.compile(r'per-channel mean:\s*([\d.]+)%')
ENC_L1_RE = re.compile(r'encoded L1 \(Test-Pure\)\s*:\s*([\d.]+)')
PAPER_L2_RE = re.compile(r'physical Relative L2 \(all-variable aggregate\)\s*:\s*([\d.]+)')


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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


def run_script(script, img_dir):
    log_path = os.path.join(LOG_DIR, f"{script.replace('.py','')}_{MODEL}.log")
    proc = subprocess.run(
        [sys.executable, script, '--model', MODEL, '--data_path', DATA_PATH, '--output_dir', img_dir],
        cwd=BASE, capture_output=True, text=True,
    )
    stdout = proc.stdout + '\n' + proc.stderr
    with open(log_path, 'w') as f:
        f.write(stdout)
    return proc.returncode, stdout


def patch_csv(path, fields, new_row):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['model'] != MODEL]
    rows.append({k: new_row.get(k, '') for k in fields})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print(f"Waiting for training PID {TRAIN_PID} to finish...", flush=True)
    while pid_alive(TRAIN_PID):
        time.sleep(30)
    print("Training finished. Re-evaluating mscale_deeponet...", flush=True)

    rc, out = run_script('eval.py', IDX50_IMG_DIR)
    if rc != 0:
        print(f"eval.py FAILED, see logs/eval.py_{MODEL}.log")
    else:
        row50 = {'model': MODEL, **parse_channel_errors(out)}
        patch_csv(os.path.join(OUT_DIR, 'errors_idx50.csv'),
                  ['model', 'rho', 'u', 'v', 'p', 'per_channel_mean'], row50)
        print(f"errors_idx50.csv updated: {row50}", flush=True)

    rc, out = run_script('eval_idx3.py', IDX3_IMG_DIR)
    if rc != 0:
        print(f"eval_idx3.py FAILED, see logs/eval_idx3.py_{MODEL}.log")
    else:
        row3 = {'model': MODEL, **parse_channel_errors(out)}
        m = ENC_L1_RE.search(out)
        if m:
            row3['encoded_L1_test_pure'] = float(m.group(1))
        m = PAPER_L2_RE.search(out)
        if m:
            row3['paper_eq17_rel_l2'] = float(m.group(1))
        patch_csv(os.path.join(OUT_DIR, 'errors_idx3.csv'),
                  ['model', 'rho', 'u', 'v', 'p', 'per_channel_mean', 'encoded_L1_test_pure', 'paper_eq17_rel_l2'], row3)
        print(f"errors_idx3.csv updated: {row3}", flush=True)

    print("mscale_deeponet eval_summary update complete.", flush=True)


if __name__ == "__main__":
    main()
