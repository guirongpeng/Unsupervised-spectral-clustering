#!/usr/bin/env bash
# 功能：
#   bash run_v3_v4_parallel.sh start   # 创建环境并启动一组新实验
#   bash run_v3_v4_parallel.sh status  # 统计最新一组实验状态
#   bash run_v3_v4_parallel.sh resume  # 续跑最新一组未完成实验
#   bash run_v3_v4_parallel.sh logs    # 查看最新一组总调度日志
#   bash run_v3_v4_parallel.sh logs ALLAML  # 查看指定数据集日志

set -euo pipefail

# 始终在脚本所在的项目根目录运行。
cd "$(dirname "$0")"

VENV_DIR=".venv"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
WORKERS="${WORKERS:-9}"
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# Python 并行脚本按该目录区分每一组批量实验。
LOG_ROOT="results/parallel_logs"

# PID 文件仅用于防止同一组实验被重复启动。
STATE_ROOT=".run_state/v3_v4_parallel"

usage() {
  cat <<'EOF'
Usage:
  bash run_v3_v4_parallel.sh start
  bash run_v3_v4_parallel.sh status
  bash run_v3_v4_parallel.sh resume
  bash run_v3_v4_parallel.sh logs
  bash run_v3_v4_parallel.sh logs DATASET
EOF
}

# 返回 results/parallel_logs 下修改时间最新的一组实验标识。
latest_run_id() {
  find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %f\n' 2>/dev/null |
    sort -nr |
    awk 'NR == 1 {print $2}'
}

# 判断指定批量实验的调度进程是否仍在运行。
is_running() {
  local run_id="$1"
  local pid_file="${STATE_ROOT}/${run_id}.pid"

  [ -f "$pid_file" ] || return 1

  local pid
  pid="$(<"$pid_file")"

  # 同时检查 PID 存在及其命令确实为本实验脚本。
  kill -0 "$pid" 2>/dev/null &&
    ps -p "$pid" -o args= 2>/dev/null |
    grep -Fq "run_v3_v4_small_parallel.py"
}

# 创建虚拟环境并安装缺失依赖。
prepare_environment() {
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Creating virtual environment: ${VENV_DIR}"
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
  fi

  # 激活后，后续 python / pip 均使用 .venv 环境。
  source "${VENV_DIR}/bin/activate"

  if ! python -c "import numpy, scipy, sklearn" >/dev/null 2>&1; then
    echo "Installing required packages..."
    python -m pip install -r requirements.txt \
      -i "$PIP_INDEX_URL"
  fi
}

# resume 命令依赖 config.py 中 resume=True。
check_resume_enabled() {
  if ! grep -Eq 'resume:[[:space:]]*bool[[:space:]]*=[[:space:]]*True' config.py; then
    echo "ERROR: Please set 'resume: bool = True' in config.py first."
    exit 1
  fi
}

# 使用给定 run_id 启动或续跑。
launch() {
  local run_id="$1"
  local log_dir="${LOG_ROOT}/${run_id}"
  local pid_file="${STATE_ROOT}/${run_id}.pid"

  if is_running "$run_id"; then
    echo "ERROR: Experiment ${run_id} is already running."
    echo "PID: $(<"$pid_file")"
    exit 1
  fi

  prepare_environment
  mkdir -p "$log_dir" "$STATE_ROOT"

  # 每个数据集还会由 Python 脚本写入：
  # results/parallel_logs/<run_id>/<dataset>.log
  nohup python -u run_v3_v4_small_parallel.py \
    --workers "$WORKERS" \
    --run-id "$run_id" \
    > "${log_dir}/scheduler.log" 2>&1 &

  echo "$!" > "$pid_file"

  echo "Started."
  echo "PID: $!"
  echo "Run ID: ${run_id}"
  echo "Scheduler log: ${log_dir}/scheduler.log"
  echo "Dataset logs: ${log_dir}/"
}

start() {
  # 新实验永远生成新的基础时间戳，避免覆盖或误续跑旧实验。
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)"
  launch "$run_id"
}

resume() {
  check_resume_enabled

  local run_id
  run_id="$(latest_run_id)"

  if [ -z "$run_id" ]; then
    echo "ERROR: No previous V3/V4 experiment was found."
    echo "Run 'bash run_v3_v4_parallel.sh start' first."
    exit 1
  fi

  echo "Resuming latest experiment: ${run_id}"
  launch "$run_id"
}

status() {
  local run_id
  run_id="$(latest_run_id)"

  if [ -z "$run_id" ]; then
    echo "No V3/V4 experiment found."
    exit 0
  fi

  echo "Latest Run ID: ${run_id}"
  if is_running "$run_id"; then
    echo "Process: running (PID $(<"${STATE_ROOT}/${run_id}.pid"))"
  else
    echo "Process: not running"
  fi

  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "No local virtual environment found; cannot parse result summaries."
    exit 0
  fi

  # 读取每个数据集、每个算法的 experiment_summary.json 并汇总。
  RUN_ID="$run_id" "${VENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
datasets = (
    "COIL20", "ORL", "SuCancer", "USPS", "Yale",
    "warpPIE10P", "GLIOMA", "TOX_171", "ALLAML",
)
algorithms = ("my_v3", "my_v4")

planned_total = completed_total = success_total = failed_total = 0
print(f"{'Dataset':<14} {'Algorithm':<10} {'Planned':>8} {'Done':>8} {'Success':>8} {'Failed':>8}")

for dataset in datasets:
    for algorithm in algorithms:
        summary = (
            Path("results")
            / f"{run_id}_{dataset}"
            / dataset
            / algorithm
            / "experiment_summary.json"
        )
        if not summary.is_file():
            print(f"{dataset:<14} {algorithm:<10} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            continue

        data = json.loads(summary.read_text(encoding="utf-8"))
        planned = int(data.get("planned_runs", 0))
        completed = int(data.get("completed_rows", 0))
        counts = data.get("status_counts", {})
        success = int(counts.get("success", 0))
        failed = int(counts.get("failed", 0))

        planned_total += planned
        completed_total += completed
        success_total += success
        failed_total += failed

        print(
            f"{dataset:<14} {algorithm:<10} "
            f"{planned:>8} {completed:>8} {success:>8} {failed:>8}"
        )

print("-" * 64)
print(
    f"{'TOTAL':<26} {planned_total:>8} {completed_total:>8} "
    f"{success_total:>8} {failed_total:>8}"
)
print(f"Remaining unrecorded runs: {max(planned_total - completed_total, 0)}")
PY
}

logs() {
  local run_id
  run_id="$(latest_run_id)"

  if [ -z "$run_id" ]; then
    echo "No V3/V4 experiment found."
    exit 1
  fi

  local log_file="${LOG_ROOT}/${run_id}/scheduler.log"

  # 若指定数据集，则读取对应的独立日志。
  if [ "${1:-}" != "" ]; then
    log_file="${LOG_ROOT}/${run_id}/${1}.log"
  fi

  if [ ! -f "$log_file" ]; then
    echo "ERROR: Log file not found: ${log_file}"
    exit 1
  fi

  echo "Viewing: ${log_file}"
  tail -n 80 -f "$log_file"
}

case "${1:-help}" in
  start)
    start
    ;;
  resume)
    resume
    ;;
  status)
    status
    ;;
  logs)
    logs "${2:-}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${1}"
    usage
    exit 1
    ;;
esac