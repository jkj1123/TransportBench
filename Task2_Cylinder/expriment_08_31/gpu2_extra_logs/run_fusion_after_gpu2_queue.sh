#!/bin/bash
# Pushed back: now waits for Task1's hyper_mscale_deeponet (gpu2_hyper_mscale_10k_logs) to
# finish on physical GPU 2, then runs Task2's fusion_deeponet.
HYPER_MSCALE_DONE_FLAG=/home/sseut1123/run/Transportbench_07_15/TransportBench/Task1_Airfoil/gpu2_hyper_mscale_10k_logs/done.flag
LOGDIR=./gpu2_extra_logs
DATA=./cylinder_full_2400_10pct_seed42.pt
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder

echo "[$(date)] Waiting for Task1 hyper_mscale_deeponet ($HYPER_MSCALE_DONE_FLAG) before starting fusion_deeponet" >> "$LOGDIR/waiter.log"
while [ ! -f "$HYPER_MSCALE_DONE_FLAG" ]; do
  sleep 30
done
echo "[$(date)] GPU2 clear, starting fusion_deeponet on GPU2" >> "$LOGDIR/waiter.log"

CUDA_VISIBLE_DEVICES=2 /home/sseut1123/anaconda3/envs/torch312/bin/python train.py --model fusion_deeponet --data_path "$DATA" --batch_size 16 --epochs 5000 > "$LOGDIR/run_fusion_deeponet.log" 2>&1
echo "[$(date)] fusion_deeponet (GPU2) finished, exit=$?" >> "$LOGDIR/waiter.log"
echo DONE > "$LOGDIR/done.flag"
