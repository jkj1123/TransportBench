#!/bin/bash
# 3-way trunk-net ablation on standard DeepONet, Task2 Cylinder, 200K params,
# 400-sample dataset (cylinder_full_2400_400_seed42.pt), GPU2, sequential:
#   1) deeponet     -- baseline (MLP trunk)
#   2) w_deeponet   -- trunk replaced by a Wavelet Neural Network (WNN)
#   3) wm_deeponet  -- trunk split into WNN + MLP sub-trunks, summed
# Same recipe as the established decay100 convention used elsewhere in this
# project: lr=1e-3, lr_decay_step=100, lr_decay_gamma=0.91, epochs=5000,
# batch_size=16. num_basis=123 fixed for w/wm so all three share the same
# basis width p (hidden_dim=123 is what the plain deeponet needs to hit 200K).
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
DATA=./cylinder_full_2400_400_seed42.pt
LOGDIR=gpu2_wnn_deeponet_logs
TRAIN_SCRIPT=train_fast.py
COMMON="--data_path $DATA --batch_size 16 --epochs 5000 --rel2_log_interval 5 --lr 1e-3 --lr_decay_step 100 --lr_decay_gamma 0.91"

echo "[$(date)] starting deeponet_200k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model deeponet $COMMON --hidden_dim 123 --run_tag "cyl400_param200k_decay100" \
    > "$LOGDIR/run_deeponet_200k.log" 2>&1
echo "[$(date)] finished deeponet_200k (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting w_deeponet_200k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model w_deeponet $COMMON --hidden_dim 169 --num_basis 123 --run_tag "cyl400_param200k_decay100" \
    > "$LOGDIR/run_w_deeponet_200k.log" 2>&1
echo "[$(date)] finished w_deeponet_200k (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting wm_deeponet_200k" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model wm_deeponet $COMMON --hidden_dim 123 --num_basis 123 --run_tag "cyl400_param200k_decay100" \
    > "$LOGDIR/run_wm_deeponet_200k.log" 2>&1
echo "[$(date)] finished wm_deeponet_200k (exit=$?)" >> "$LOGDIR/chain.log"

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] all 3 finished (sequential)" >> "$LOGDIR/chain.log"
