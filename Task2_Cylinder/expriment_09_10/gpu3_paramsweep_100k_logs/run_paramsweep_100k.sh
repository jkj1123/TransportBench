#!/bin/bash
# HyperDeepONet parameter-count sweep, 10k -> 2M, on GPU3.
# Same conditions as the original gpu0_paramsweep_logs / gpu0_paramsweep_logs_1M2M sweeps,
# but: epochs=100000 (was 50000) and --lr 1e-3 (was default 3e-4).
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu3_paramsweep_100k_logs
DATA=./cylinder_full_2400_10pct_seed42.pt
COMMON="--model hyperdeeponet --data_path $DATA --lr 1e-3 --batch_size 192 --epochs 100000 --lr_decay_step 2000 --lr_decay_gamma 0.91"
TRAIN_SCRIPT=train_fast.py

declare -A SIZES=( [10k]=15 [20k]=19 [50k]=27 [100k]=35 [500k]=61 [1M]=76 [2M]=97 )

for tag in 10k 20k 50k 100k 500k 1M 2M; do
  h=${SIZES[$tag]}
  echo "[$(date)] starting hyperdeeponet_${tag} (hidden_dim=$h)" >> "$LOGDIR/chain.log"
  "$PY" "$TRAIN_SCRIPT" $COMMON --hidden_dim "$h" --run_tag "param${tag}_100k_lr1e3" \
      > "$LOGDIR/run_hyperdeeponet_${tag}.log" 2>&1
  echo "[$(date)] finished hyperdeeponet_${tag} (exit=$?)" >> "$LOGDIR/chain.log"
done

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 7 finished (sequential)" >> "$LOGDIR/chain.log"
