#!/bin/bash
# HyperDeepONet, NO lr decay (lr_decay_step=0, disabled -- paper default),
# for 50K/100K/200K tiers, run in parallel on 3 separate idle GPUs.
# Same recipe as the decay100/decay200 runs otherwise: lr=1e-3, epochs=5000,
# batch_size=16, same hidden_dim per tier.
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
DATA=./cylinder_full_2400_400_seed42.pt
LOGDIR=gpu_hyperdeeponet_nodecay_logs
TRAIN_SCRIPT=train_fast.py

run_one () {
  local gpu=$1 tag=$2 h=$3
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PY" "$TRAIN_SCRIPT" --model hyperdeeponet --data_path "$DATA" \
      --batch_size 16 --epochs 5000 --rel2_log_interval 5 --lr 1e-3 \
      --hidden_dim "$h" --run_tag "cyl400_param${tag}_nodecay" \
      > "$LOGDIR/run_hyperdeeponet_${tag}_nodecay.log" 2>&1
  echo "[$(date)] finished hyperdeeponet_${tag}_nodecay on gpu$gpu (exit=$?)" >> "$LOGDIR/chain.log"
}

echo "[$(date)] starting 3 parallel no-decay runs (50k/100k/200k)" >> "$LOGDIR/chain.log"
run_one 1 50k  27 &
run_one 2 100k 35 &
run_one 3 200k 44 &
wait
echo "[$(date)] all 3 no-decay runs finished" >> "$LOGDIR/chain.log"
echo ALL_DONE > "$LOGDIR/done.flag"
