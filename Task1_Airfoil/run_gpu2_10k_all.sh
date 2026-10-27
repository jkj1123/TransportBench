#!/bin/bash
# Task1 Airfoil: 5 DeepONet-family models @ 10,000 epochs each, sequential on GPU2.
# Records wall time + rel L2 error per model into gpu2_task1_10k_logs/summary.csv
cd /home/sseut1123/run/Transportbench_07_15/TransportBench/Task1_Airfoil
export CUDA_VISIBLE_DEVICES=2
PY=/home/sseut1123/anaconda3/envs/torch312/bin/python
LOGDIR=gpu2_task1_10k_logs
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.csv"
echo "model,epochs,wall_time_sec,wall_time_hms,best_test_loss,final_test_rel2,min_test_rel2" > "$SUMMARY"

MODELS=(deeponet hyperdeeponet mscale_deeponet hyper_mscale_deeponet residual_fusion_deeponet)

echo "=== GPU2 Task1 10k-epoch run started at $(date) (PID $$) ===" >> "$LOGDIR/chain.log"

for m in "${MODELS[@]}"; do
    mkdir -p "output/$m"
    echo "--- [$(date)] starting $m ---" >> "$LOGDIR/chain.log"
    START=$(date +%s)
    "$PY" train.py --model "$m" --epochs 10000 --data_path airfoil_unified_128x128.pt > "$LOGDIR/run_${m}.log" 2>&1
    exit_code=$?
    END=$(date +%s)
    DUR=$((END-START))
    HMS=$(printf '%02d:%02d:%02d' $((DUR/3600)) $((DUR%3600/60)) $((DUR%60)))
    echo "--- [$(date)] finished $m (exit=$exit_code, wall=$HMS) ---" >> "$LOGDIR/chain.log"

    BEST_LOSS=$(grep -a "Training Complete" "$LOGDIR/run_${m}.log" | tail -1 | grep -oP 'Best Test Loss: \K[0-9.eE+-]+')
    if [ -f "output/$m/rel2_history.csv" ]; then
        FINAL_REL2=$(tail -n 1 "output/$m/rel2_history.csv" | cut -d, -f2)
        MIN_REL2=$(tail -n +2 "output/$m/rel2_history.csv" | awk -F, 'BEGIN{min=1e9} {if($2<min) min=$2} END{print min}')
    else
        FINAL_REL2="NA"
        MIN_REL2="NA"
    fi
    echo "$m,10000,$DUR,$HMS,$BEST_LOSS,$FINAL_REL2,$MIN_REL2" >> "$SUMMARY"
done

echo "=== GPU2 Task1 10k-epoch run ALL DONE at $(date) ===" >> "$LOGDIR/chain.log"
echo ALL_DONE > "$LOGDIR/done.flag"
