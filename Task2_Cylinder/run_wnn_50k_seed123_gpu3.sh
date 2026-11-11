#!/bin/bash
# 50K-param tier, seed=123 (reproducibility check vs the original seed=42 50K
# runs), GPU3, sequential:
#   1) rhm_deeponet (h=48, trunk_hidden_dim=7, num_basis=48)
#   2) r_deeponet   (h=78, num_basis=78)
#   3) wm_deeponet  (h=61, num_basis=61)
# decay100 convention: lr=1e-3, lr_decay_step=100, lr_decay_gamma=0.91, epochs=5000, batch_size=16
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
DATA=./cylinder_full_2400_400_seed42.pt
LOGDIR=gpu3_wnn50k_seed123_logs
mkdir -p "$LOGDIR"
TRAIN_SCRIPT=train_fast.py
COMMON="--data_path $DATA --batch_size 16 --epochs 5000 --rel2_log_interval 5 --lr 1e-3 --lr_decay_step 100 --lr_decay_gamma 0.91 --seed 123 --run_tag cyl400_param50k_decay100_seed123"

echo "[$(date)] starting rhm_deeponet_50k_seed123" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model rhm_deeponet $COMMON --hidden_dim 48 --trunk_hidden_dim 7 --num_basis 48 \
    > "$LOGDIR/run_rhm_deeponet_50k_seed123.log" 2>&1
echo "[$(date)] finished rhm_deeponet_50k_seed123 (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting r_deeponet_50k_seed123" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model r_deeponet $COMMON --hidden_dim 78 --num_basis 78 \
    > "$LOGDIR/run_r_deeponet_50k_seed123.log" 2>&1
echo "[$(date)] finished r_deeponet_50k_seed123 (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting wm_deeponet_50k_seed123" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model wm_deeponet $COMMON --hidden_dim 61 --num_basis 61 \
    > "$LOGDIR/run_wm_deeponet_50k_seed123.log" 2>&1
echo "[$(date)] finished wm_deeponet_50k_seed123 (exit=$?)" >> "$LOGDIR/chain.log"

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 3 finished (sequential, GPU3, seed=123)" >> "$LOGDIR/chain.log"
