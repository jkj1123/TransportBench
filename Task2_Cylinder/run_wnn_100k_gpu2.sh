#!/bin/bash
# 100K-param tier for Ridgelet/Wavelet trunk ablations (10_28_wnn_models), GPU2, sequential:
#   1) whm_deeponet (h=71, trunk_hidden_dim=7, num_basis=71)
#   2) rm_deeponet  (h=87, num_basis=87)
#   3) w_deeponet   (h=111, num_basis=111)
# decay100 convention: lr=1e-3, lr_decay_step=100, lr_decay_gamma=0.91, epochs=5000, batch_size=16
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
DATA=./cylinder_full_2400_400_seed42.pt
LOGDIR=gpu2_wnn100k_logs
mkdir -p "$LOGDIR"
TRAIN_SCRIPT=train_fast.py
COMMON="--data_path $DATA --batch_size 16 --epochs 5000 --rel2_log_interval 5 --lr 1e-3 --lr_decay_step 100 --lr_decay_gamma 0.91 --run_tag cyl400_param100k_decay100"

echo "[$(date)] starting whm_deeponet_100k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model whm_deeponet $COMMON --hidden_dim 71 --trunk_hidden_dim 7 --num_basis 71 \
    > "$LOGDIR/run_whm_deeponet_100k.log" 2>&1
echo "[$(date)] finished whm_deeponet_100k (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting rm_deeponet_100k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model rm_deeponet $COMMON --hidden_dim 87 --num_basis 87 \
    > "$LOGDIR/run_rm_deeponet_100k.log" 2>&1
echo "[$(date)] finished rm_deeponet_100k (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting w_deeponet_100k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model w_deeponet $COMMON --hidden_dim 111 --num_basis 111 \
    > "$LOGDIR/run_w_deeponet_100k.log" 2>&1
echo "[$(date)] finished w_deeponet_100k (exit=$?)" >> "$LOGDIR/chain.log"

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 3 finished (sequential, GPU2)" >> "$LOGDIR/chain.log"
