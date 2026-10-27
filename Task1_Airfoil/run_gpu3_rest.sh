#!/bin/bash
# GPU3: remaining 3 DeepONet-family models @ 30,000 epochs each, sequential.
# (hyper_mscale_deeponet, deeponet, hyperdeeponet)
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task1_Airfoil
export CUDA_VISIBLE_DEVICES=3
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu3_30k_logs
mkdir -p "$LOGDIR"

MODELS=(hyper_mscale_deeponet deeponet hyperdeeponet)

echo "=== GPU3 run started at $(date) (PID $$) ===" >> "$LOGDIR/chain.log"

for m in "${MODELS[@]}"; do
    mkdir -p "output/$m"
    echo "--- [$(date)] starting $m ---" >> "$LOGDIR/chain.log"
    START=$(date +%s)
    "$PY" train.py --model "$m" --epochs 30000 --data_path airfoil_unified_128x128.pt > "$LOGDIR/run_${m}.log" 2>&1
    exit_code=$?
    END=$(date +%s)
    DUR=$((END-START))
    HMS=$(printf '%02d:%02d:%02d' $((DUR/3600)) $((DUR%3600/60)) $((DUR%60)))
    echo "--- [$(date)] finished $m (exit=$exit_code, wall=$HMS) ---" >> "$LOGDIR/chain.log"
done

echo "=== GPU3 run ALL DONE at $(date) ===" >> "$LOGDIR/chain.log"
echo ALL_DONE > "$LOGDIR/done.flag"
