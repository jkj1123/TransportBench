#!/bin/bash
# Polls the tqdm output of the Task2 GPU1 chain (deeponet -> hyperdeeponet -> mscale_deeponet)
LOGDIR=/home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder/gpu1_chain_logs
DONE_FLAG=$LOGDIR/task2_chain_done.flag
PROGRESS_CSV=$LOGDIR/progress_gpu1_task2_chain.csv

MODELS="deeponet hyperdeeponet mscale_deeponet"

if [ ! -f "$PROGRESS_CSV" ]; then
  echo "model,epoch,total_epoch,unix_time,iso_time" > "$PROGRESS_CSV"
fi

declare -A last_seen

while [ ! -f "$DONE_FLAG" ]; do
  for m in $MODELS; do
    f="$LOGDIR/run_${m}.log"
    if [ -f "$f" ]; then
      line=$(tr '\r' '\n' < "$f" | grep -oE '[0-9]+/[0-9]+ \[' | tail -1)
      if [ -n "$line" ]; then
        epoch=$(echo "$line" | cut -d'/' -f1)
        total=$(echo "$line" | cut -d'/' -f2 | cut -d' ' -f1)
        key="${m}"
        if [ "${last_seen[$key]}" != "$epoch" ]; then
          echo "${m},${epoch},${total},$(date +%s),$(date -Iseconds)" >> "$PROGRESS_CSV"
          last_seen[$key]="$epoch"
        fi
      fi
    fi
  done
  sleep 60
done
echo "ALL,DONE,DONE,$(date +%s),$(date -Iseconds)" >> "$PROGRESS_CSV"
