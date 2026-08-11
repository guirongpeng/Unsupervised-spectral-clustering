#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_ID="v3_v4_20260811"
WORKERS=9
PYTHON_BIN="${PYTHON_BIN:-python}"

nohup "$PYTHON_BIN" -u run_v3_v4_small_parallel.py \
  --workers "$WORKERS" \
  --run-id "$RUN_ID" \
  > "${RUN_ID}.nohup.log" 2>&1 &

echo "PID: $!"
echo "Scheduler log: ${RUN_ID}.nohup.log"
echo "Dataset logs: results/parallel_logs/${RUN_ID}/"