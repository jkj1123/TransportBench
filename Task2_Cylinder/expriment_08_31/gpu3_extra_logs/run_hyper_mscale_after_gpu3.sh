#!/bin/bash
# Waits for GPU3's residual_fusion_deeponet (PID given below) to finish/be stopped at 50000,
# then runs Task2's hyper_mscale_deeponet on physical GPU 3.
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
PARENT_PID=3503471   # Task1 residual_fusion_deeponet, capped at 50000 by its watcher
LOGDIR=./gpu3_extra_logs
DATA=./cylinder_full_2400_10pct_seed42.pt

echo "[$(date)] Waiting for GPU3 job (PID $PARENT_PID) to finish before starting hyper_mscale_deeponet" >> "$LOGDIR/waiter.log"
while kill -0 "$PARENT_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date)] GPU3 job finished, starting hyper_mscale_deeponet on GPU3" >> "$LOGDIR/waiter.log"

CUDA_VISIBLE_DEVICES=3 /home/sseut1123/anaconda3/envs/torch312/bin/python train.py --model hyper_mscale_deeponet --data_path "$DATA" --batch_size 192 --epochs 50000 > "$LOGDIR/run_hyper_mscale_deeponet.log" 2>&1
echo "[$(date)] hyper_mscale_deeponet (GPU3) finished, exit=$?" >> "$LOGDIR/waiter.log"
echo DONE > "$LOGDIR/done.flag"
