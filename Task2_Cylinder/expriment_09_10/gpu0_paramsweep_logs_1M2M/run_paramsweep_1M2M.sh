#!/bin/bash
# Parameter-count sweep for hyperdeeponet on Task2 (Cylinder), run SEQUENTIALLY on GPU0.
# Extension of the original gpu0_paramsweep_logs sweep (10k/20k/50k/100k/500k) to
# larger param counts: 1M -> 2M. Same conditions (COMMON args, dataset, train_fast.py).
# hidden_dim values chosen to land closest to 1,000,000 / 2,000,000 params:
#   hidden_dim=76 -> 1,005,484 params (~1M)
#   hidden_dim=97 -> 2,014,597 params (~2M)
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu0_paramsweep_logs_1M2M
DATA=./cylinder_full_2400_10pct_seed42.pt
COMMON="--model hyperdeeponet --data_path $DATA --batch_size 192 --epochs 50000 --lr_decay_step 2000 --lr_decay_gamma 0.91"
TRAIN_SCRIPT=train_fast.py

declare -A SIZES=( [1M]=76 [2M]=97 )

for tag in 1M 2M; do
  h=${SIZES[$tag]}
  echo "[$(date)] starting hyperdeeponet_${tag} (hidden_dim=$h)" >> "$LOGDIR/chain.log"
  "$PY" "$TRAIN_SCRIPT" $COMMON --hidden_dim "$h" --run_tag "param${tag}" \
      > "$LOGDIR/run_hyperdeeponet_${tag}.log" 2>&1
  echo "[$(date)] finished hyperdeeponet_${tag} (exit=$?)" >> "$LOGDIR/chain.log"
done

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 2 finished (sequential)" >> "$LOGDIR/chain.log"
