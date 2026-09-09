#!/bin/bash
# Parameter-count sweep for hyperdeeponet on Task2 (Cylinder), run SEQUENTIALLY on GPU0
# in increasing parameter-count order: 10k -> 20k -> 50k -> 100k -> 500k.
# Dataset: 10% subset (seed42). Same schedule as the earlier gpu1_lrdecay chain.
# Uses train_fast.py / data_loader_fast.py (overhead-fixed loader, ~1.9x faster than
# train.py's DataLoader-based path -- see 2026-09-05 investigation). train.py/data_loader.py
# are untouched.
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu0_paramsweep_logs
DATA=./cylinder_full_2400_10pct_seed42.pt
COMMON="--model hyperdeeponet --data_path $DATA --batch_size 192 --epochs 50000 --lr_decay_step 2000 --lr_decay_gamma 0.91"
TRAIN_SCRIPT=train_fast.py

declare -A SIZES=( [10k]=15 [20k]=19 [50k]=27 [100k]=35 [500k]=61 )

for tag in 10k 20k 50k 100k 500k; do
  h=${SIZES[$tag]}
  echo "[$(date)] starting hyperdeeponet_${tag} (hidden_dim=$h)" >> "$LOGDIR/chain.log"
  "$PY" "$TRAIN_SCRIPT" $COMMON --hidden_dim "$h" --run_tag "param${tag}" \
      > "$LOGDIR/run_hyperdeeponet_${tag}.log" 2>&1
  echo "[$(date)] finished hyperdeeponet_${tag} (exit=$?)" >> "$LOGDIR/chain.log"
done

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 5 finished (sequential)" >> "$LOGDIR/chain.log"
