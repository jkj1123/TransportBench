#!/bin/bash
# Moved here from GPU2 (which was stuck waiting ~9.6h behind Task1 jobs).
# Waits for GPU3's hyper_mscale_deeponet (PID below) to finish, then runs
# Task2's fusion_deeponet on physical GPU 3.
PARENT_PID=2661051   # hyper_mscale_deeponet on GPU3
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
LOGDIR=./gpu3_extra_logs
DATA=./cylinder_full_2400_10pct_seed42.pt
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python

echo "[$(date)] Waiting for GPU3 hyper_mscale_deeponet (PID $PARENT_PID) to finish before starting fusion_deeponet" >> "$LOGDIR/waiter_fusion.log"
while kill -0 "$PARENT_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date)] GPU3 clear, starting fusion_deeponet on GPU3" >> "$LOGDIR/waiter_fusion.log"

CUDA_VISIBLE_DEVICES=3 "$PY" train.py --model fusion_deeponet --data_path "$DATA" --batch_size 16 --epochs 5000 > "$LOGDIR/run_fusion_deeponet.log" 2>&1
echo "[$(date)] fusion_deeponet (GPU3) finished, exit=$?" >> "$LOGDIR/waiter_fusion.log"
echo DONE > "$LOGDIR/done_fusion_gpu3.flag"
