#!/usr/bin/env bash
set -euo pipefail

cd /home/test/wsk/mirage_drug_miniprototype
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

FRAME="outputs/chembl_assay_primary_20260729/benchmark_frame.csv"
OUT_ROOT="outputs/revision_e22_pretrained_frozen_strict_20260804"
LOG="${OUT_ROOT}.log"
mkdir -p "${OUT_ROOT}"
: > "${LOG}"

SPLITS=(assay_cold target_cold temporal)
SEEDS=(42 43 44 45 46)
GPUS=(1 2)

run_one() {
  local split="$1"
  local seed="$2"
  local gpu="$3"
  local out="${OUT_ROOT}/frozen_esm2_chemberta_ridge_${split}_s${seed}"
  if [[ -s "${out}/external_metrics.json" ]]; then
    echo "[SKIP] ${split} seed ${seed}" | tee -a "${LOG}"
    return 0
  fi
  echo "[RUN] ${split} seed ${seed} gpu ${gpu}" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" ./.venv/bin/python scripts/run_external_pretrained_frozen_strict_from_frame.py \
    --benchmark-frame "${FRAME}" \
    --split-method "${split}" \
    --seed "${seed}" \
    --output-dir "${out}" \
    --cache-dir "${OUT_ROOT}/cache" \
    --encode-batch-size 64 \
    --device cuda \
    >> "${LOG}" 2>&1
  echo "[DONE] ${split} seed ${seed}" | tee -a "${LOG}"
}

run_one assay_cold 42 1

jobs_running=0
gpu_idx=0
for split in "${SPLITS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ "${split}" == "assay_cold" && "${seed}" == "42" ]]; then
      continue
    fi
    gpu="${GPUS[$gpu_idx]}"
    run_one "${split}" "${seed}" "${gpu}" &
    jobs_running=$((jobs_running + 1))
    gpu_idx=$(((gpu_idx + 1) % ${#GPUS[@]}))
    if [[ "${jobs_running}" -ge "${#GPUS[@]}" ]]; then
      wait -n
      jobs_running=$((jobs_running - 1))
    fi
  done
done
wait

./.venv/bin/python scripts/summarize_e22_pretrained_frozen_strict.py | tee -a "${LOG}"
echo "[ALL DONE] pretrained frozen strict baseline" | tee -a "${LOG}"

