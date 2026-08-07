#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/test/wsk/mirage_drug_miniprototype"
PY="$ROOT/.venv/bin/python"
GRAPHDTA_REPO="/home/test/wsk/Finish/ModTune/experiments/docking_free/GraphDTA"
MGRAPHDTA_SRC="/home/test/wsk/Finish/ModTune/experiments/docking_free/MGraphDTA/model.py"
MGRAPHDTA_WRAPPER="/tmp/mgraphdta_wrapper_revision_20260803"
OUT_GRAPH="$ROOT/outputs/revision_e08b_external_graphdta_standard_20260803"
OUT_MGRAPH="$ROOT/outputs/revision_e08b_external_mgraphdta_standard_20260803"
LOG_DIR="$ROOT/logs/revision_e08b_external_graph_models_20260803"
mkdir -p "$OUT_GRAPH" "$OUT_MGRAPH" "$LOG_DIR" "$MGRAPHDTA_WRAPPER/regression"
ln -sf "$MGRAPHDTA_SRC" "$MGRAPHDTA_WRAPPER/regression/model.py"

MAX_JOBS=4
GPU_POOL=(0 1 2 5)

is_done() {
  local out_dir="$1"
  [[ -f "$out_dir/external_metrics.json" ]]
}

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    sleep 60
  done
}

run_graphdta() {
  local dataset="$1"
  local split="$2"
  local seed="$3"
  local gpu="$4"
  local out_dir="$OUT_GRAPH/${dataset}_${split}_s${seed}"
  local log_file="$LOG_DIR/graphdta_${dataset}_${split}_s${seed}.log"
  if is_done "$out_dir"; then
    echo "SKIP completed graphdta $dataset $split s$seed" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  echo "START graphdta $dataset $split s$seed gpu=$gpu $(date -Is)" | tee -a "$LOG_DIR/master.log"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_external_graphdta_official.py \
      --graphdta-repo "$GRAPHDTA_REPO" \
      --dataset "$dataset" \
      --split-method "$split" \
      --seed "$seed" \
      --output-dir "$out_dir" \
      --model-variant gin \
      --epochs 60 \
      --patience 10 \
      --batch-size 256 \
      --num-workers 4 \
      --graph-workers 1
    echo "DONE graphdta $dataset $split s$seed $(date -Is)" >> "$LOG_DIR/master.log"
  ) > "$log_file" 2>&1 &
}

run_mgraphdta() {
  local dataset="$1"
  local split="$2"
  local seed="$3"
  local gpu="$4"
  local out_dir="$OUT_MGRAPH/${dataset}_${split}_s${seed}"
  local log_file="$LOG_DIR/mgraphdta_${dataset}_${split}_s${seed}.log"
  if is_done "$out_dir"; then
    echo "SKIP completed mgraphdta $dataset $split s$seed" | tee -a "$LOG_DIR/master.log"
    return 0
  fi
  echo "START mgraphdta $dataset $split s$seed gpu=$gpu $(date -Is)" | tee -a "$LOG_DIR/master.log"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_external_mgraphdta_official.py \
      --mgraphdta-repo "$MGRAPHDTA_WRAPPER" \
      --dataset "$dataset" \
      --split-method "$split" \
      --seed "$seed" \
      --output-dir "$out_dir" \
      --epochs 80 \
      --patience 12 \
      --batch-size 256 \
      --num-workers 4 \
      --graph-workers 1
    echo "DONE mgraphdta $dataset $split s$seed $(date -Is)" >> "$LOG_DIR/master.log"
  ) > "$log_file" 2>&1 &
}

datasets=(BindingDB_Kd DAVIS KIBA)
splits=(cold_drug cold_target cold_pair)
seeds=(42 43 44)

i=0
for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    for seed in "${seeds[@]}"; do
      wait_for_slot
      gpu="${GPU_POOL[$((i % ${#GPU_POOL[@]}))]}"
      run_graphdta "$dataset" "$split" "$seed" "$gpu"
      i=$((i + 1))
    done
  done
done

for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    for seed in "${seeds[@]}"; do
      wait_for_slot
      gpu="${GPU_POOL[$((i % ${#GPU_POOL[@]}))]}"
      run_mgraphdta "$dataset" "$split" "$seed" "$gpu"
      i=$((i + 1))
    done
  done
done

wait
cd "$ROOT"
"$PY" scripts/build_external_baseline_report.py \
  --outputs-root outputs \
  --output-dir outputs/revision_completion_20260803/external_report_after_graph_models
echo "ALL_DONE external_graph_models $(date -Is)" | tee -a "$LOG_DIR/master.log"
