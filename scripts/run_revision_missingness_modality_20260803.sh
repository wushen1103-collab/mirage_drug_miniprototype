#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/test/wsk/mirage_drug_miniprototype"
PY="$ROOT/.venv/bin/python"
BENCH="$ROOT/outputs/chembl_assay_primary_20260729"
OUT_BASE="$ROOT/outputs/revision_e09b_missingness_modality_20260803"
LOG_DIR="$ROOT/logs/revision_e09b_missingness_modality_20260803"
mkdir -p "$OUT_BASE" "$LOG_DIR"

MAX_JOBS=2
GPU_POOL=(6 7)

is_done() {
  local out_dir="$1"
  if [[ ! -f "$out_dir/summary.csv" ]]; then
    return 1
  fi
  local n
  n=$(find "$out_dir" -mindepth 2 -maxdepth 2 -name metrics.json 2>/dev/null | wc -l)
  [[ "$n" -ge 9 ]]
}

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    sleep 30
  done
}

run_one() {
  local name="$1"
  local seq_prob="$2"
  local text_prob="$3"
  local gpu="$4"
  local out_dir="$OUT_BASE/$name"
  local log_file="$LOG_DIR/$name.log"

  if is_done "$out_dir"; then
    echo "SKIP completed $name" | tee -a "$LOG_DIR/master.log"
    return 0
  fi

  echo "START $name seq=$seq_prob text=$text_prob gpu=$gpu $(date -Is)" | tee -a "$LOG_DIR/master.log"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_chembl_assay_v2_eval.py \
      --benchmark-dir "$BENCH" \
      --output-dir "$out_dir" \
      --splits assay_cold target_cold temporal \
      --seeds 42 43 44 \
      --missing-sequence-prob "$seq_prob" \
      --missing-text-prob "$text_prob" \
      --n-neighbors 8 \
      --retrieval-reference-size 5000
    echo "DONE $name $(date -Is)" >> "$LOG_DIR/master.log"
  ) > "$log_file" 2>&1 &
}

declare -a TASKS=(
  "text_only_015 0.00 0.15"
  "text_only_035 0.00 0.35"
  "text_only_050 0.00 0.50"
  "text_only_070 0.00 0.70"
  "sequence_only_015 0.15 0.00"
  "sequence_only_035 0.35 0.00"
  "sequence_only_050 0.50 0.00"
  "sequence_only_070 0.70 0.00"
  "sequence_text_015 0.15 0.15"
  "sequence_text_035 0.35 0.35"
)

i=0
for task in "${TASKS[@]}"; do
  wait_for_slot
  read -r name seq_prob text_prob <<< "$task"
  gpu="${GPU_POOL[$((i % ${#GPU_POOL[@]}))]}"
  run_one "$name" "$seq_prob" "$text_prob" "$gpu"
  i=$((i + 1))
done

wait
echo "ALL_DONE missingness_modality $(date -Is)" | tee -a "$LOG_DIR/master.log"
