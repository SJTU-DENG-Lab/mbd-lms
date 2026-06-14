#!/bin/bash
set -x  # Echo every command (useful for debugging)
set -e  # Exit immediately on error

# ================= NCCL & Distributed Env =================
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=22  # Bump IB timeout threshold
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# Increase global distributed timeout (e.g. 2 hours)
export TORCH_DISTRIBUTED_DEBUG=INFO

# ================= Node & Resource Params =================
# Prefer PET env vars; fall back to single-node defaults
NNODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
# Human-readable rank (1-indexed)
NODE_RANK_PLUS_ONE=$((NODE_RANK + 1))

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29531}

# Auto-detect GPU count
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
# Total world size across all nodes
NPROC_ALL_NODES=$((NPROC * NNODES))

if [[ "${NNODES}" == "1" && "${NPROC}" == "1" ]]; then
    echo "🧪 Local single-node single-GPU mode detected (FSDP still launched via torchrun with one rank)."
fi

# ================= Task Paths =================
PREFIX="multi_bd_train"
# Prefer the repo's own VeOmni to avoid stale site-packages from the environment
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/VeOmni:${REPO_ROOT}:${PYTHONPATH}"

# Overridable via env; defaults to SDAR multi-BD distill v2
TASK_REL_PATH="${TASK_REL_PATH:-sdar/train_sdar_multi_bd_distill_v2}"
SCRIPT="${PREFIX}/tasks/${TASK_REL_PATH}.py"
CONFIG="${CONFIG:-${PREFIX}/configs/sft/${TASK_REL_PATH}.yaml}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "❌ Task script not found: ${SCRIPT}"
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "❌ Config file not found: ${CONFIG}"
    exit 1
fi

# Log directory — timestamped and per-rank to prevent clobbering
LOG_DIR="${PREFIX}/logs/${TASK_REL_PATH}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/distributed_train_rank${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

# ================= Diagnostic Info =================
echo "============================================================"
echo "🚀 Distributed Training Debug Info"
echo "============================================================"
echo "📅 Time           : $(date)"
echo "💻 Hostname       : $(hostname)"
echo "🌐 Network IF     : ${NCCL_SOCKET_IFNAME} (IPv4 Only)"
echo "------------------------------------------------------------"
echo "🔢 Node Rank      : ${NODE_RANK} (Current Node: ${NODE_RANK_PLUS_ONE} / ${NNODES})"
echo "🏭 GPUs per Node  : ${NPROC}"
echo "🌍 World Size     : ${NPROC_ALL_NODES} GPUs total"
echo "👑 Master Addr    : ${MASTER_ADDR}:${MASTER_PORT}"
echo "------------------------------------------------------------"
echo "📜 Task Script    : ${SCRIPT}"
echo "⚙️  Config File    : ${CONFIG}"
echo "📝 Log File       : ${LOG_FILE}"
echo "🐍 PYTHONPATH      : ${PYTHONPATH}"
python - <<'PY'
import inspect
import veomni
from veomni.data.data_collator import MakeMicroBatchCollator
print(f"📦 veomni path     : {veomni.__file__}")
print(
    "✅ coverage collator: "
    + ("expanded_features" in inspect.getsource(MakeMicroBatchCollator.__call__)).__str__()
)
PY
echo "============================================================"

# ================= Launch Args =================
DIST_ARGS="--nproc_per_node=${NPROC} --nnodes=${NNODES} --node_rank=${NODE_RANK}"

if [[ "$NNODES" == "1" ]]; then
    echo "⚠️  Running in Standalone Mode (Single Node)"
    DIST_ARGS="${DIST_ARGS} --standalone"
else
    echo "🔗 Running in Distributed Mode (Connecting to Master)"
    # Explicit rendezvous backend to guard against stale defaults
    DIST_ARGS="${DIST_ARGS} --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

# ================= Launch Training =================
# 2>&1 | tee keeps output both on the terminal and in the log file
torchrun ${DIST_ARGS} "${SCRIPT}" "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
