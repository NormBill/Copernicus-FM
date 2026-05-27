#!/usr/bin/env bash
set -euo pipefail

# Simple sequential sweep for the task evidence validity experiments.
# It runs eight training/evaluation jobs in order:
#   1-4. fixed memory_size=1024, sweep validity_beta
#   5-8. fixed validity_beta=0.01, sweep memory_size
#
# Run:
#   bash scripts/run_evidence_beta_memory_8runs.sh
#
# Quick dry run:
#   EPOCHS=1 bash scripts/run_evidence_beta_memory_8runs.sh

cd "$(dirname "$0")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

DATASET="${DATASET:-cobench_eurosat_s2}"
DATASET_CONFIG="${DATASET_CONFIG:-src/configs/dataset/cobench_eurosat_s2.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-src/configs/model/copernicusfm_cls.yaml}"

LR="${LR:-0.1}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
STRATEGY="${STRATEGY:-auto}"
DEVICE="${DEVICE:-cuda}"

MEMORY_DIR="${MEMORY_DIR:-$(pwd)/outputs/evidence}"
mkdir -p "${MEMORY_DIR}"

build_memory_if_missing() {
  local size="$1"
  local path="${MEMORY_DIR}/eurosat_s2_train${size}.pt"

  if [ ! -f "${path}" ]; then
    echo "[Evidence] Building train-split-only memory: size=${size}"
    python tools/build_task_evidence_memory.py \
      --model-config "${MODEL_CONFIG}" \
      --dataset-config "${DATASET_CONFIG}" \
      --output "${path}" \
      --feature-type pooled \
      --memory-size "${size}" \
      --max-batches -1 \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}"
  else
    echo "[Evidence] Reusing memory: ${path}"
  fi
}

# Memory files needed by the eight runs.
build_memory_if_missing 128
build_memory_if_missing 256
build_memory_if_missing 512
build_memory_if_missing 1024

# ---------------------------------------------------------------------------
# 1. Sweep validity_beta with memory_size=1024, beta=0.0
# ---------------------------------------------------------------------------
export TASK_EVIDENCE_MEMORY="${MEMORY_DIR}/eurosat_s2_train1024.pt"
python src/main.py \
  output_dir=outputs/evidence_beta_0.0_mem1024_s${SEED} \
  model=copernicusfm_cls_evidence_score_only \
  dataset="${DATASET}" \
  lr="${LR}" \
  task=classification \
  num_gpus="${NUM_GPUS}" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  warmup_epochs=0 \
  seed="${SEED}" \
  strategy="${STRATEGY}" \
  model.validity_alpha=1.0 \
  model.validity_bias=1.0 \
  model.validity_beta=0.0

# ---------------------------------------------------------------------------
# 2. Sweep validity_beta with memory_size=1024, beta=0.01
# ---------------------------------------------------------------------------
export TASK_EVIDENCE_MEMORY="${MEMORY_DIR}/eurosat_s2_train1024.pt"
python src/main.py \
  output_dir=outputs/evidence_beta_0.01_mem1024_s${SEED} \
  model=copernicusfm_cls_evidence_score_only \
  dataset="${DATASET}" \
  lr="${LR}" \
  task=classification \
  num_gpus="${NUM_GPUS}" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  warmup_epochs=0 \
  seed="${SEED}" \
  strategy="${STRATEGY}" \
  model.validity_alpha=1.0 \
  model.validity_bias=1.0 \
  model.validity_beta=0.01

# ---------------------------------------------------------------------------
# 3. Sweep validity_beta with memory_size=1024, beta=0.05
# ---------------------------------------------------------------------------
export TASK_EVIDENCE_MEMORY="${MEMORY_DIR}/eurosat_s2_train1024.pt"
python src/main.py \
  output_dir=outputs/evidence_beta_0.05_mem1024_s${SEED} \
  model=copernicusfm_cls_evidence_score_only \
  dataset="${DATASET}" \
  lr="${LR}" \
  task=classification \
  num_gpus="${NUM_GPUS}" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  warmup_epochs=0 \
  seed="${SEED}" \
  strategy="${STRATEGY}" \
  model.validity_alpha=1.0 \
  model.validity_bias=1.0 \
  model.validity_beta=0.05

# ---------------------------------------------------------------------------
# 4. Sweep validity_beta with memory_size=1024, beta=0.1
# ---------------------------------------------------------------------------
export TASK_EVIDENCE_MEMORY="${MEMORY_DIR}/eurosat_s2_train1024.pt"
python src/main.py \
  output_dir=outputs/evidence_beta_0.1_mem1024_s${SEED} \
  model=copernicusfm_cls_evidence_score_only \
  dataset="${DATASET}" \
  lr="${LR}" \
  task=classification \
  num_gpus="${NUM_GPUS}" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  warmup_epochs=0 \
  seed="${SEED}" \
  strategy="${STRATEGY}" \
  model.validity_alpha=1.0 \
  model.validity_bias=1.0 \
  model.validity_beta=0.1

# ---------------------------------------------------------------------------
# 5. Sweep memory_size with beta=0.01, memory_size=128
# ---------------------------------------------------------------------------
export TASK_EVIDENCE_MEMORY="${MEMORY_DIR}/eurosat_s2_train128.pt"
python src/main.py \
  output_dir=outputs/evidence_mem128_beta0.01_s${SEED} \
  model=copernicusfm_cls_evidence_score_only \
  dataset="${DATASET}" \
  lr="${LR}" \
  task=classification \
  num_gpus="${NUM_GPUS}" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  warmup_epochs=0 \
  seed="${SEED}" \
  strategy="${STRATEGY}" \
  model.validity_alpha=1.0 \
  model.validity_bias=1.0 \
  model.validity_beta=0.01

# ---------------------------------------------------------------------------
# 6. Sweep memory_size with beta=0.01, memory_size=256
# ---------------------------------------------------------------------------
export TASK_EVIDENCE_MEMORY="${MEMORY_DIR}/eurosat_s2_train256.pt"
python src/main.py \
  output_dir=outputs/evidence_mem256_beta0.01_s${SEED} \
  model=copernicusfm_cls_evidence_score_only \
  dataset="${DATASET}" \
  lr="${LR}" \
  task=classification \
  num_gpus="${NUM_GPUS}" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  warmup_epochs=0 \
  seed="${SEED}" \
  strategy="${STRATEGY}" \
  model.validity_alpha=1.0 \
  model.validity_bias=1.0 \
  model.validity_beta=0.01

# ---------------------------------------------------------------------------
# 7. Sweep memory_size with beta=0.01, memory_size=512
# ---------------------------------------------------------------------------
export TASK_EVIDENCE_MEMORY="${MEMORY_DIR}/eurosat_s2_train512.pt"
python src/main.py \
  output_dir=outputs/evidence_mem512_beta0.01_s${SEED} \
  model=copernicusfm_cls_evidence_score_only \
  dataset="${DATASET}" \
  lr="${LR}" \
  task=classification \
  num_gpus="${NUM_GPUS}" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  warmup_epochs=0 \
  seed="${SEED}" \
  strategy="${STRATEGY}" \
  model.validity_alpha=1.0 \
  model.validity_bias=1.0 \
  model.validity_beta=0.01

# ---------------------------------------------------------------------------
# 8. Sweep memory_size with beta=0.01, memory_size=1024
# ---------------------------------------------------------------------------
export TASK_EVIDENCE_MEMORY="${MEMORY_DIR}/eurosat_s2_train1024.pt"
python src/main.py \
  output_dir=outputs/evidence_mem1024_beta0.01_s${SEED} \
  model=copernicusfm_cls_evidence_score_only \
  dataset="${DATASET}" \
  lr="${LR}" \
  task=classification \
  num_gpus="${NUM_GPUS}" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  epochs="${EPOCHS}" \
  warmup_epochs=0 \
  seed="${SEED}" \
  strategy="${STRATEGY}" \
  model.validity_alpha=1.0 \
  model.validity_bias=1.0 \
  model.validity_beta=0.01

echo "[Evidence] Finished eight sequential evidence runs."
