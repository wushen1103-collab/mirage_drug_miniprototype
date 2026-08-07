#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/test/wsk/mirage_drug_miniprototype"
PY="$ROOT/.venv/bin/python"
FRAME="$ROOT/outputs/chembl_assay_primary_20260729/benchmark_frame.csv"
OUT_BASE="$ROOT/outputs/revision_e18_external_chembl_strict_mgraphdta_20260803"
LOG_DIR="$ROOT/logs/revision_e18_external_chembl_strict_mgraphdta_20260803"
MGRAPHDTA_SRC="/home/test/wsk/Finish/ModTune/experiments/docking_free/MGraphDTA/model.py"
MGRAPHDTA_WRAPPER="/tmp/mgraphdta_wrapper_revision_20260803"

mkdir -p "$OUT_BASE" "$LOG_DIR" "$MGRAPHDTA_WRAPPER/regression"
ln -sf "$MGRAPHDTA_SRC" "$MGRAPHDTA_WRAPPER/regression/model.py"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
MAX_JOBS="${MAX_JOBS:-2}"
GPU_POOL=(6 7)

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    sleep 60
  done
}

is_done() {
  local out="$1"
  [[ -s "$out/external_metrics.json" && -s "$out/test_prediction_with_binary.csv" ]]
}

run_one() {
  local split="$1"
  local seed="$2"
  local gpu="$3"
  local out="$OUT_BASE/mgraphdta_${split}_s${seed}"
  local log_file="$LOG_DIR/mgraphdta_${split}_s${seed}.log"

  if is_done "$out"; then
    echo "SKIP external_chembl mgraphdta $split s$seed $(date -Is)" >> "$LOG_DIR/master.log"
    return 0
  fi

  echo "START external_chembl mgraphdta $split s$seed gpu=$gpu $(date -Is)" >> "$LOG_DIR/master.log"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_external_chembl_strict_from_frame.py \
      --benchmark-frame "$FRAME" \
      --model mgraphdta \
      --split-method "$split" \
      --seed "$seed" \
      --output-dir "$out" \
      --mgraphdta-repo "$MGRAPHDTA_WRAPPER" \
      --epochs 60 \
      --patience 10 \
      --batch-size 256 \
      --num-workers 4 \
      --graph-workers 1
    echo "DONE external_chembl mgraphdta $split s$seed $(date -Is)" >> "$LOG_DIR/master.log"
  ) > "$log_file" 2>&1 &
}

echo "QUEUE_START external_chembl_strict_mgraphdta $(date -Is)" >> "$LOG_DIR/master.log"
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
find "$OUT_BASE" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/external_metrics.json' ';' -print | sort > "$OUT_BASE/completed_runs.txt"
echo "ALL_DONE external_chembl_strict_mgraphdta $(date -Is)" >> "$LOG_DIR/master.log"
