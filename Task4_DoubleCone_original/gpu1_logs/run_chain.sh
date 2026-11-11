#!/bin/bash
# GPU1 sequential chain for Task4_DoubleCone_original: deeponet -> fno -> unet
# 2500 epochs each, no_fourier condition.
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task4_DoubleCone_original
export CUDA_VISIBLE_DEVICES=1
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=./gpu1_logs
SAVE_DIR=./checkpoints_2500ep_nofourier
TIMING_CSV=$LOGDIR/timing.csv

echo "model,event,unix_time,iso_time" > "$TIMING_CSV"

run_model() {
  local m=$1
  echo "$m,start,$(date +%s),$(date -Iseconds)" >> "$TIMING_CSV"
  echo "[$(date)] starting $m" >> "$LOGDIR/chain.log"
  "$PY" train.py --model "$m" --data_path ./double_cone.pt --batch_size 8 --epochs 2500 --no_fourier --save_dir "$SAVE_DIR" > "$LOGDIR/train_${m}.log" 2>&1
  echo "[$(date)] finished $m (exit=$?)" >> "$LOGDIR/chain.log"
  echo "$m,end,$(date +%s),$(date -Iseconds)" >> "$TIMING_CSV"
}

run_model deeponet
run_model fno
run_model unet

echo "[$(date)] running eval_extra_metrics for all 3 models" >> "$LOGDIR/chain.log"
for m in deeponet fno unet; do
  "$PY" eval_extra_metrics.py --model "$m" --data_path ./double_cone.pt \
    --checkpoint "${SAVE_DIR}/best_model_{}_nofourier.pth" --device cuda:0 \
    > "$LOGDIR/eval_${m}.log" 2>&1
done

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] chain complete" >> "$LOGDIR/chain.log"
