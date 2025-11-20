#!/bin/bash
#SBATCH --job-name=qwen3-vl-finetune
#SBATCH --partition=kill-shared
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint="hopper"
#SBATCH --gres=gpu:1

set -euo pipefail

module purge >/dev/null 2>&1 || true

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "Writing outputs to ${RESULTS_DIR}"

echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "KOA_ML_CODE_ROOT=${KOA_ML_CODE_ROOT:-unset}"
echo "KOA_RUN_DIR=${KOA_RUN_DIR:-unset}"
echo "KOA_RUN_METADATA_DIR=${KOA_RUN_METADATA_DIR:-unset}"
echo "KOA_SHARED_ENV=${KOA_SHARED_ENV:-unset}"
echo

echo "==== GPU Info ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo

echo "==== CUDA_VISIBLE_DEVICES ====="
# Make sure we never request more processes than visible GPUs to avoid invalid device errors.
detect_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
        IFS=',' read -r -a _cuda_devices <<< "${CUDA_VISIBLE_DEVICES}"
        echo "${#_cuda_devices[@]}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | grep -c 'GPU '
    else
        echo 1
    fi
}

available_gpus=$(detect_gpu_count)
requested_procs=${NPROC_PER_NODE:-}
requested_procs_num=${NPROC_PER_NODE:-0}
if [[ -z "${requested_procs}" || "${requested_procs_num}" -gt "${available_gpus}" ]]; then
    if [[ -n "${requested_procs}" && "${available_gpus}" -gt 0 ]]; then
        echo "Reducing NPROC_PER_NODE=${requested_procs} to match available GPUs (${available_gpus})." >&2
    fi
    if [[ "${available_gpus}" -lt 1 ]]; then
        available_gpus=1
    fi
    NPROC_PER_NODE=${available_gpus}
fi

export NPROC_PER_NODE
echo "Using NPROC_PER_NODE: ${NPROC_PER_NODE}"

echo "==== GPU Memory Detection ===="
# Query first GPU's name and total VRAM (in MB)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)

echo "Detected GPU: ${GPU_NAME} with ${GPU_MEM} MB VRAM"

# Default settings
BATCH_SIZE=4
DTYPE="bfloat16"

# Adjust batch size and other configs based on VRAM capacity
if (( GPU_MEM < 50000 )); then
    echo "⚠️ GPU has <50GB VRAM — too small"
    exit 1
elif (( GPU_MEM < 90000 )); then
    BATCH_SIZE=4
else
    BATCH_SIZE=8
fi

export BATCH_SIZE

echo "Using batch size: ${BATCH_SIZE}"

echo "==== Python Environment ====="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
else
  echo "python not found"
fi
echo

source "scripts/setup_env.sh"

export REPO_DIR
export RESULTS_DIR
export WANDB_DIR="${RESULTS_DIR}"

cd "${REPO_DIR}/qwen-vl-finetune"
srun uv run bash scripts/sft_qwen3_8b_instruct.sh
