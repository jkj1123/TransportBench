#!/bin/bash
# Waits for run_fusion_deeponet.log to appear (job actually started), then polls wall time every 60s.
LOGDIR=/home/sseut1123/run/Transportbench_07_15/TransportBench/Task2_Cylinder/gpu2_extra_logs
LOG=$LOGDIR/run_fusion_deeponet.log
DONE_FLAG=$LOGDIR/done.flag
PROGRESS_CSV=$LOGDIR/progress_gpu2_fusion_deeponet.csv

echo "model,epoch,total_epoch,unix_time,iso_time" > "$PROGRESS_CSV"

while [ ! -f "$LOG" ] && [ ! -f "$DONE_FLAG" ]; do sleep 30; done

last_seen=""
while [ ! -f "$DONE_FLAG" ]; do
  if [ -f "$LOG" ]; then
    line=$(tr '\r' '\n' < "$LOG" | grep -oE '[0-9]+/[0-9]+ \[' | tail -1)
    if [ -n "$line" ]; then
      epoch=$(echo "$line" | cut -d'/' -f1)
      total=$(echo "$line" | cut -d'/' -f2 | cut -d' ' -f1)
      if [ "$epoch" != "$last_seen" ]; then
        echo "fusion_deeponet,${epoch},${total},$(date +%s),$(date -Iseconds)" >> "$PROGRESS_CSV"
        last_seen="$epoch"
      fi
    fi
  fi
  sleep 60
done
echo "fusion_deeponet,DONE,DONE,$(date +%s),$(date -Iseconds)" >> "$PROGRESS_CSV"
