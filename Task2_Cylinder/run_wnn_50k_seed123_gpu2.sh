#!/bin/bash
# 50K-param tier, seed=123 (reproducibility check vs the original seed=42 50K
# runs), GPU2, sequential:
#   1) whm_deeponet (h=48, trunk_hidden_dim=7, num_basis=48)
#   2) rm_deeponet  (h=61, num_basis=61)
#   3) w_deeponet   (h=78, num_basis=78)
# decay100 convention: lr=1e-3, lr_decay_step=100, lr_decay_gamma=0.91, epochs=5000, batch_size=16
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
DATA=./cylinder_full_2400_400_seed42.pt
LOGDIR=gpu2_wnn50k_seed123_logs
mkdir -p "$LOGDIR"
TRAIN_SCRIPT=train_fast.py
COMMON="--data_path $DATA --batch_size 16 --epochs 5000 --rel2_log_interval 5 --lr 1e-3 --lr_decay_step 100 --lr_decay_gamma 0.91 --seed 123 --run_tag cyl400_param50k_decay100_seed123"

echo "[$(date)] starting whm_deeponet_50k_seed123" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model whm_deeponet $COMMON --hidden_dim 48 --trunk_hidden_dim 7 --num_basis 48 \
    > "$LOGDIR/run_whm_deeponet_50k_seed123.log" 2>&1
echo "[$(date)] finished whm_deeponet_50k_seed123 (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting rm_deeponet_50k_seed123" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model rm_deeponet $COMMON --hidden_dim 61 --num_basis 61 \
    > "$LOGDIR/run_rm_deeponet_50k_seed123.log" 2>&1
echo "[$(date)] finished rm_deeponet_50k_seed123 (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting w_deeponet_50k_seed123" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model w_deeponet $COMMON --hidden_dim 78 --num_basis 78 \
    > "$LOGDIR/run_w_deeponet_50k_seed123.log" 2>&1
echo "[$(date)] finished w_deeponet_50k_seed123 (exit=$?)" >> "$LOGDIR/chain.log"

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 3 finished (sequential, GPU2, seed=123)" >> "$LOGDIR/chain.log"
