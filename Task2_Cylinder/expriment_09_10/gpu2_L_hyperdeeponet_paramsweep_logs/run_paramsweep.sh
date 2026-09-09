#!/bin/bash
# L_HyperDeepONet (r=4, low-rank factorized hypernetwork) parameter-count sweep,
# 10k -> 2M, on GPU2. Sequential. epochs=50000, lr=1e-3, same decay schedule as
# the other 10K-scale L_hyperdeeponet run this sweep is meant to extend.
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu2_L_hyperdeeponet_paramsweep_logs
DATA=./cylinder_full_2400_10pct_seed42.pt
COMMON="--model L_hyperdeeponet --data_path $DATA --lr 1e-3 --batch_size 192 --epochs 50000 --lr_decay_step 2000 --lr_decay_gamma 0.91"
TRAIN_SCRIPT=train_fast.py

declare -A SIZES=( [10k]=19 [20k]=27 [50k]=42 [100k]=60 [200k]=85 [500k]=135 [1M]=192 [2M]=272 )

for tag in 10k 20k 50k 100k 200k 500k 1M 2M; do
  h=${SIZES[$tag]}
  echo "[$(date)] starting L_hyperdeeponet_${tag} (hidden_dim=$h)" >> "$LOGDIR/chain.log"
  "$PY" "$TRAIN_SCRIPT" $COMMON --hidden_dim "$h" --run_tag "param${tag}" \
      > "$LOGDIR/run_L_hyperdeeponet_${tag}.log" 2>&1
  echo "[$(date)] finished L_hyperdeeponet_${tag} (exit=$?)" >> "$LOGDIR/chain.log"
done

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 8 finished (sequential)" >> "$LOGDIR/chain.log"
