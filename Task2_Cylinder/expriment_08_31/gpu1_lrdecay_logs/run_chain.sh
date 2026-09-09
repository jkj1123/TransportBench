#!/bin/bash
# Sequential (not concurrent) GPU1 lr-decay chain: deeponet -> mscale_deeponet
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=1
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu1_lrdecay_logs
DATA=./cylinder_full_2400_10pct_seed42.pt
COMMON="--data_path $DATA --batch_size 192 --epochs 50000 --lr_decay_step 2000 --lr_decay_gamma 0.91 --run_tag lrdecay"

echo "[$(date)] starting deeponet" >> "$LOGDIR/chain.log"
"$PY" train.py --model deeponet $COMMON > "$LOGDIR/run_deeponet.log" 2>&1
echo "[$(date)] finished deeponet (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting mscale_deeponet" >> "$LOGDIR/chain.log"
"$PY" train.py --model mscale_deeponet $COMMON > "$LOGDIR/run_mscale_deeponet.log" 2>&1
echo "[$(date)] finished mscale_deeponet (exit=$?)" >> "$LOGDIR/chain.log"

echo ALL_DONE > "$LOGDIR/done.flag"
