#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/test/wsk/mirage_drug_miniprototype"
PY="$ROOT/.venv/bin/python"
BENCH="$ROOT/outputs/chembl_assay_primary_20260729"
OUT_BASE="$ROOT/outputs/revision_e19_component_nomissing_20260803"
LOG_DIR="$ROOT/logs/revision_e19_component_nomissing_20260803"
WAIT_LOG="$ROOT/logs/revision_e16_component_10seed_extension_20260803/master.log"
mkdir -p "$OUT_BASE" "$LOG_DIR"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MAX_JOBS="${MAX_JOBS:-6}"
GPU_POOL=(0 1 2 5 6 7)

while ! grep -q "ALL_DONE component_10seed_extension" "$WAIT_LOG" 2>/dev/null; do
  echo "WAIT_E16 component_nomissing $(date -Is)" >> "$LOG_DIR/master.log"
  sleep 120
done

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    sleep 30
  done
}

is_done() {
  local root="$1"
  local split="$2"
  local seed="$3"
  [[ -s "$root/${split}_s${seed}/metrics.json" && -s "$root/${split}_s${seed}/predictions.csv" ]]
}

run_one() {
  local split="$1"
  local seed="$2"
  local gpu="$3"
  local run_root="$OUT_BASE/${split}_s${seed}_root"
  local log_file="$LOG_DIR/${split}_s${seed}.log"

  if is_done "$run_root" "$split" "$seed"; then
    echo "SKIP component_nomissing $split s$seed $(date -Is)" >> "$LOG_DIR/master.log"
    return 0
  fi

  echo "START component_nomissing $split s$seed gpu=$gpu $(date -Is)" >> "$LOG_DIR/master.log"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_chembl_assay_v2_eval.py \
      --benchmark-dir "$BENCH" \
      --output-dir "$run_root" \
      --splits "$split" \
      --seeds "$seed" \
      --missing-sequence-prob 0.0 \
      --missing-text-prob 0.0 \
      --n-neighbors 8 \
      --retrieval-reference-size 5000
    echo "DONE component_nomissing $split s$seed $(date -Is)" >> "$LOG_DIR/master.log"
  ) > "$log_file" 2>&1 &
}

echo "QUEUE_START component_nomissing $(date -Is)" >> "$LOG_DIR/master.log"
i=0
for split in assay_cold target_cold temporal; do
  for seed in 42 43 44 45 46; do
    wait_for_slot
    gpu="${GPU_POOL[$((i % ${#GPU_POOL[@]}))]}"
    run_one "$split" "$seed" "$gpu"
    i=$((i + 1))
  done
done

wait
cd "$ROOT"
find "$OUT_BASE" -type f -name metrics.json | sort > "$OUT_BASE/completed_metrics.txt"
echo "ALL_DONE component_nomissing $(date -Is)" >> "$LOG_DIR/master.log"
