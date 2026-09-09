#!/bin/bash
LOGDIR=/home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder/gpu1_osc_lrdecay_logs
PROGRESS_CSV=$LOGDIR/progress.csv
MODELS="hyperdeeponet hyper_mscale_deeponet"
echo "model,epoch,total_epoch,unix_time,iso_time" > "$PROGRESS_CSV"
declare -A last_seen
while [ ! -f "$LOGDIR/done.flag" ]; do
  for m in $MODELS; do
    f="$LOGDIR/run_${m}.log"
    if [ -f "$f" ]; then
      line=$(tr '\r' '\n' < "$f" | grep -oE '[0-9]+/50000 \[' | tail -1)
      if [ -n "$line" ]; then
        epoch=$(echo "$line" | cut -d'/' -f1)
        if [ "${last_seen[$m]}" != "$epoch" ]; then
          echo "${m},${epoch},50000,$(date +%s),$(date -Iseconds)" >> "$PROGRESS_CSV"
          last_seen[$m]="$epoch"
        fi
      fi
    fi
  done
  sleep 30
done
