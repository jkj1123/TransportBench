#!/bin/bash
# 100K-param tier for Ridgelet/Wavelet trunk ablations (10_28_wnn_models), GPU3, sequential:
#   1) rhm_deeponet (h=71, trunk_hidden_dim=7, num_basis=71)
#   2) r_deeponet   (h=111, num_basis=111)
#   3) wm_deeponet  (h=87, num_basis=87)
# decay100 convention: lr=1e-3, lr_decay_step=100, lr_decay_gamma=0.91, epochs=5000, batch_size=16
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
DATA=./cylinder_full_2400_400_seed42.pt
LOGDIR=gpu3_wnn100k_logs
mkdir -p "$LOGDIR"
TRAIN_SCRIPT=train_fast.py
COMMON="--data_path $DATA --batch_size 16 --epochs 5000 --rel2_log_interval 5 --lr 1e-3 --lr_decay_step 100 --lr_decay_gamma 0.91 --run_tag cyl400_param100k_decay100"

echo "[$(date)] starting rhm_deeponet_100k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model rhm_deeponet $COMMON --hidden_dim 71 --trunk_hidden_dim 7 --num_basis 71 \
    > "$LOGDIR/run_rhm_deeponet_100k.log" 2>&1
echo "[$(date)] finished rhm_deeponet_100k (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting r_deeponet_100k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model r_deeponet $COMMON --hidden_dim 111 --num_basis 111 \
    > "$LOGDIR/run_r_deeponet_100k.log" 2>&1
echo "[$(date)] finished r_deeponet_100k (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting wm_deeponet_100k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model wm_deeponet $COMMON --hidden_dim 87 --num_basis 87 \
    > "$LOGDIR/run_wm_deeponet_100k.log" 2>&1
echo "[$(date)] finished wm_deeponet_100k (exit=$?)" >> "$LOGDIR/chain.log"

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 3 finished (sequential, GPU3)" >> "$LOGDIR/chain.log"
