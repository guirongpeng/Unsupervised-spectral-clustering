#!/usr/bin/env bash
# run_v3_v4_parallel.sh
# 功能：创建虚拟环境、安装依赖、后台并行运行 MY-V3 和 MY-V4 实验。

set -euo pipefail
# -e：任一命令失败时立即退出
# -u：使用未定义变量时立即报错
# pipefail：管道中任一命令失败则整体失败

# 切换到本脚本所在目录，确保从任意位置执行都能找到项目文件。
cd "$(dirname "$0")"

# 虚拟环境目录；首次运行会自动创建。
VENV_DIR=".venv"

# 创建虚拟环境所使用的系统 Python；可通过环境变量覆盖。
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"

# 数据集并行进程数；默认 9，可通过 WORKERS=4 bash ... 覆盖。
WORKERS="${WORKERS:-4}"

# 实验标识；默认自动生成时间戳。
# 续跑时应显式指定与原实验相同的 RUN_ID。
RUN_ID="${RUN_ID:-v3_v4_$(date +%Y%m%d_%H%M%S)}"

# 清华 PyPI 镜像，用于加快国内服务器安装依赖。
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 1. 若虚拟环境不存在，则创建。
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

# 2. 激活虚拟环境。
# 后台 nohup 进程会继承该环境，因此会使用 .venv 内的 Python 与依赖。
source "${VENV_DIR}/bin/activate"

# 3. 检查核心依赖；只要有任一个缺失，就按 requirements.txt 安装。
if ! python -c "import numpy, scipy, sklearn" >/dev/null 2>&1; then
  echo "Installing required packages from ${PIP_INDEX_URL} ..."
  python -m pip install -r requirements.txt \
    -i "$PIP_INDEX_URL"
fi

# 4. 使用 nohup 后台运行。
# 外层日志记录调度情况；每个数据集还有独立日志：
# results/parallel_logs/<RUN_ID>/<dataset>.log
nohup python -u run_v3_v4_small_parallel.py \
  --workers "$WORKERS" \
  --run-id "$RUN_ID" \
  > "${RUN_ID}.nohup.log" 2>&1 &

# 5. 输出运行信息。
echo "Started successfully."
echo "PID: $!"
echo "Run ID: $RUN_ID"
echo "Scheduler log: ${RUN_ID}.nohup.log"
echo "Dataset logs: results/parallel_logs/${RUN_ID}/"