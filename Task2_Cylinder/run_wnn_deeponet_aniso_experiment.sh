#!/bin/bash
# Re-run of the WNN-trunk ablation with ANISOTROPIC (axis-aligned) wavelet
# dilation instead of isotropic (radial). Baseline plain DeepONet is
# unchanged from the isotropic run, so only w_deeponet/wm_deeponet re-run.
# Same recipe: lr=1e-3, lr_decay_step=100, gamma=0.91, epochs=5000,
# batch_size=16, 400-sample cylinder dataset, GPU2.
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
DATA=./cylinder_full_2400_400_seed42.pt
LOGDIR=gpu2_wnn_deeponet_aniso_logs
TRAIN_SCRIPT=train_fast.py
COMMON="--data_path $DATA --batch_size 16 --epochs 5000 --rel2_log_interval 5 --lr 1e-3 --lr_decay_step 100 --lr_decay_gamma 0.91"

echo "[$(date)] starting w_deeponet_200k_aniso" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model w_deeponet $COMMON --hidden_dim 169 --num_basis 123 --run_tag "cyl400_param200k_decay100_aniso" \
    > "$LOGDIR/run_w_deeponet_200k_aniso.log" 2>&1
echo "[$(date)] finished w_deeponet_200k_aniso (exit=$?)" >> "$LOGDIR/chain.log"

echo "[$(date)] starting wm_deeponet_200k_aniso" >> "$LOGDIR/chain.log"
"$PY" "$TRAIN_SCRIPT" --model wm_deeponet $COMMON --hidden_dim 123 --num_basis 123 --run_tag "cyl400_param200k_decay100_aniso" \
    > "$LOGDIR/run_wm_deeponet_200k_aniso.log" 2>&1
echo "[$(date)] finished wm_deeponet_200k_aniso (exit=$?)" >> "$LOGDIR/chain.log"

echo ALL_DONE > "$LOGDIR/done.flag"
echo "[$(date)] both anisotropic runs finished (sequential)" >> "$LOGDIR/chain.log"
