#!/bin/bash
# Waits for GPU1's current lrdecay chain (deeponet -> mscale_deeponet, PID below) to finish,
# then runs hyperdeeponet -> hyper_mscale_deeponet (both lr-decay) sequentially on GPU1.
PARENT_PID=123623   # gpu1_lrdecay_logs/run_chain.sh (deeponet_lrdecay -> mscale_deeponet_lrdecay)
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=1
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu1_osc_lrdecay_logs
DATA=./cylinder_full_2400_10pct_seed42.pt
COMMON="--data_path $DATA --batch_size 192 --epochs 50000 --lr_decay_step 2000 --lr_decay_gamma 0.91 --run_tag lrdecay"

echo "[$(date)] waiting for current GPU1 chain (PID $PARENT_PID) to finish" >> "$LOGDIR/chain.log"
while kill -0 "$PARENT_PID" 2>/dev/null; do
  sleep 30
done

echo "[$(date)] starting hyperdeeponet" >> "$LOGDIR/chain.log"
"$PY" train.py --model hyperdeeponet $COMMON > "$LOGDIR/run_hyperdeeponet.log" 2>&1
echo "[$(date)] finished hyperdeeponet (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting hyper_mscale_deeponet" >> "$LOGDIR/chain.log"
"$PY" train.py --model hyper_mscale_deeponet $COMMON > "$LOGDIR/run_hyper_mscale_deeponet.log" 2>&1
echo "[$(date)] finished hyper_mscale_deeponet (exit=$?)" >> "$LOGDIR/chain.log"

echo ALL_DONE > "$LOGDIR/done.flag"
