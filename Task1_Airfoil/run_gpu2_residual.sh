#!/bin/bash
# GPU2: residual_fusion_deeponet alone @ 30,000 epochs (this model dominates total
# wall time, so it runs solo while GPU3 handles the other 3 models sequentially).
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task1_Airfoil
export CUDA_VISIBLE_DEVICES=2
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu2_30k_logs
mkdir -p "$LOGDIR"

echo "=== GPU2 run started at $(date) (PID $$) ===" >> "$LOGDIR/chain.log"
mkdir -p output/residual_fusion_deeponet
START=$(date +%s)
"$PY" train.py --model residual_fusion_deeponet --epochs 30000 --data_path airfoil_unified_128x128.pt > "$LOGDIR/run_residual_fusion_deeponet.log" 2>&1
exit_code=$?
END=$(date +%s)
DUR=$((END-START))
HMS=$(printf '%02d:%02d:%02d' $((DUR/3600)) $((DUR%3600/60)) $((DUR%60)))
echo "--- [$(date)] finished residual_fusion_deeponet (exit=$exit_code, wall=$HMS) ---" >> "$LOGDIR/chain.log"
echo "=== GPU2 run ALL DONE at $(date) ===" >> "$LOGDIR/chain.log"
echo ALL_DONE > "$LOGDIR/done.flag"
