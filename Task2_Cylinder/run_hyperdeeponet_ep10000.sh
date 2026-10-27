#!/bin/bash
# HyperDeepONet, 50K/100K/200K x {no-decay, decay_step=200, decay_step=100},
# EPOCHS=10000 (2x the original 5000-epoch sweep), to compare epoch budget.
# 9 jobs total, queued across 4 idle GPUs via flock-based per-GPU locks so
# a new job starts the instant any GPU frees (no fixed "waves").
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
DATA=./cylinder_full_2400_400_seed42.pt
LOGDIR=gpu_hyperdeeponet_ep10000_logs
TRAIN_SCRIPT=train_fast.py
LOCKDIR="$LOGDIR/locks"
mkdir -p "$LOCKDIR"
for i in 0 1 2 3; do : > "$LOCKDIR/gpu$i.lock"; done

run_job () {
  local tag=$1 h=$2 cond=$3 step=$4 gamma=$5
  local gpu=-1
  while true; do
    for i in 0 1 2 3; do
      exec {fd}>"$LOCKDIR/gpu$i.lock"
      if flock -n "$fd"; then
        gpu=$i
        break 2
      fi
      exec {fd}>&-
    done
    sleep 5
  done
  echo "[$(date)] starting hyperdeeponet_${tag}_${cond}_ep10000 on gpu$gpu" >> "$LOGDIR/chain.log"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PY" "$TRAIN_SCRIPT" --model hyperdeeponet --data_path "$DATA" \
      --batch_size 16 --epochs 10000 --rel2_log_interval 5 --lr 1e-3 \
      --lr_decay_step "$step" --lr_decay_gamma "$gamma" \
      --hidden_dim "$h" --run_tag "cyl400_param${tag}_${cond}_ep10000" \
      > "$LOGDIR/run_hyperdeeponet_${tag}_${cond}_ep10000.log" 2>&1
  local rc=$?
  echo "[$(date)] finished hyperdeeponet_${tag}_${cond}_ep10000 on gpu$gpu (exit=$rc)" >> "$LOGDIR/chain.log"
  exec {fd}>&-
}

echo "[$(date)] starting 9 ep10000 jobs (50k/100k/200k x nodecay/decay200/decay100)" >> "$LOGDIR/chain.log"

run_job 50k  27 nodecay   0   1.0  &
run_job 50k  27 decay200  200 0.91 &
run_job 50k  27 decay100  100 0.91 &
run_job 100k 35 nodecay   0   1.0  &
run_job 100k 35 decay200  200 0.91 &
run_job 100k 35 decay100  100 0.91 &
run_job 200k 44 nodecay   0   1.0  &
run_job 200k 44 decay200  200 0.91 &
run_job 200k 44 decay100  100 0.91 &
wait

echo "[$(date)] all 9 ep10000 jobs finished" >> "$LOGDIR/chain.log"
echo ALL_DONE > "$LOGDIR/done.flag"
