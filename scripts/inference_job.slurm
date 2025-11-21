#!/bin/bash
#SBATCH --job-name=qwen-inference
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --gres=gpu:1

# #SBATCH --partition=gpuA40x4-preempt
# # SBATCH --constraint="hopper|ada|ampere"
set -euo pipefail

module purge >/dev/null 2>&1 || true

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
SHARED_CACHE_ROOT="${KOA_REMOTE_ROOT:-${HOME}/koa-remote}/cache"
CACHE_ROOT="${KOA_CACHE_ROOT:-${SHARED_CACHE_ROOT}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${CACHE_ROOT}"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${XDG_CACHE_HOME}/hf"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
export APPTAINER_CACHEDIR="${XDG_CACHE_HOME}/apptainer"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${APPTAINER_CACHEDIR}"

echo "==== Cache Directories ====="
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "TMPDIR=${TMPDIR}"
echo "APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
echo

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
detect_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
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

echo "==== Python Environment ====="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
else
  echo "python not found"
fi
echo

export REPO_DIR
export RESULTS_DIR

export MODEL_PATH="${QWEN_VL_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
export OUTPUT_FILE="${RESULTS_DIR}/predictions.jsonl"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-10000}"
export HF_DATASET_NAME="${HF_DATASET_NAME:-oumi-ai/multimodal-open-r1-8192-filtered-mid-ic}"
export DATA_FILE="${DATA_FILE:-}"
export IMAGE_ROOT="${IMAGE_ROOT:-}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export SKIP_GENERATION="${SKIP_GENERATION:-1}"
export RESUME_RUN="${RESUME_RUN:-1}"

echo "==== Inference Parameters ===="
echo "Model path: ${MODEL_PATH}"
if [[ -n "${HF_DATASET_NAME}" ]]; then
  echo "HF dataset: ${HF_DATASET_NAME}"
fi
if [[ -n "${DATA_FILE}" ]]; then
  echo "Data file: ${DATA_FILE}"
fi
if [[ -n "${IMAGE_ROOT}" ]]; then
  echo "Image root: ${IMAGE_ROOT}"
fi
echo "Output file: ${OUTPUT_FILE}"
echo "Sample limit: ${SAMPLE_LIMIT}"
echo "Batch size: ${BATCH_SIZE}"
if [[ "${SKIP_GENERATION}" == "1" ]]; then
  echo "Skip generation: yes"
else
  echo "Skip generation: no"
fi
if [[ "${RESUME_RUN}" == "1" ]]; then
  echo "Resume run: yes"
else
  echo "Resume run: no"
fi
echo

CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04}"
declare -a CONTAINER_BINDS
if [[ -n "${KOA_REMOTE_ROOT:-}" && -d "${KOA_REMOTE_ROOT}" ]]; then
  CONTAINER_BINDS+=("${KOA_REMOTE_ROOT}:${KOA_REMOTE_ROOT}")
fi
if [[ -z "${KOA_REMOTE_ROOT:-}" || ! -d "${KOA_REMOTE_ROOT}" ]]; then
  CONTAINER_BINDS+=("${JOB_DIR}:${JOB_DIR}")
fi
if [[ -n "${DATA_FILE}" ]]; then
  DATA_ROOT="$(dirname "${DATA_FILE}")"
  if [[ -n "${DATA_ROOT}" && -d "${DATA_ROOT}" ]]; then
    CONTAINER_BINDS+=("${DATA_ROOT}:${DATA_ROOT}")
  fi
fi
if [[ -n "${IMAGE_ROOT}" && -d "${IMAGE_ROOT}" ]]; then
  CONTAINER_BINDS+=("${IMAGE_ROOT}:${IMAGE_ROOT}")
fi

declare -a CONTAINER_CMD=("apptainer" "exec" "--nv")
for bind in "${CONTAINER_BINDS[@]}"; do
  CONTAINER_CMD+=("--bind" "${bind}")
done
CONTAINER_CMD+=("${CONTAINER_IMAGE}")

"${CONTAINER_CMD[@]}" bash -c '
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source "$REPO_DIR/scripts/setup_env.sh"
cd "$REPO_DIR"

CMD=(uv run evaluation/mmmu/run_inference.py
  --model-path "$MODEL_PATH"
  --output-file "$OUTPUT_FILE"
  --batch-size "$BATCH_SIZE"
  --limit "$SAMPLE_LIMIT"
)

if [[ -n "${HF_DATASET_NAME:-}" ]]; then
  CMD+=("--hf-dataset-name" "$HF_DATASET_NAME")
fi
if [[ -n "${DATA_FILE:-}" ]]; then
  CMD+=("--data-file" "$DATA_FILE")
fi
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  CMD+=("--image-root" "$IMAGE_ROOT")
fi
if [[ "${SKIP_GENERATION}" == "1" ]]; then
  CMD+=("--skip-generation")
fi
if [[ "${RESUME_RUN}" == "1" ]]; then
  CMD+=("--resume")
fi

printf "Running inference command:"
printf " %q" "${CMD[@]}"
printf "\n"
"${CMD[@]}"
'
