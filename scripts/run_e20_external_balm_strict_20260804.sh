#!/usr/bin/env bash
set -u
cd /home/test/wsk/mirage_drug_miniprototype

ROOT="outputs/revision_e20_balm_strict_20260804"
LOG="outputs/revision_e20_balm_strict_20260804.log"
FRAME="outputs/chembl_assay_primary_20260729/benchmark_frame.csv"
BALM_REPO="/home/test/ext_balm"
PROTEIN_MODEL="/home/test/wsk/hf_models/esm2_t30_150m_ur50d"
DRUG_MODEL="/home/test/wsk/hf_models/chemberta_77m_mtr"
mkdir -p "$ROOT" outputs

run_one() {
  local gpu="$1"
  local split="$2"
  local seed="$3"
  local out="${ROOT}/balm_projection_${split}_s${seed}"
  if [[ -f "${out}/external_metrics.json" ]]; then
    echo "===== $(date '+%F %T') SKIP existing BALM strict ${split} seed=${seed} =====" | tee -a "$LOG"
    return 0
  fi
  mkdir -p "$out"
  echo "===== $(date '+%F %T') START BALM strict ${split} seed=${seed} gpu=${gpu} =====" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" ./.venv/bin/python scripts/run_external_balm_strict_from_frame.py \
    --balm-repo "$BALM_REPO" \
    --benchmark-frame "$FRAME" \
    --split-method "$split" \
    --seed "$seed" \
    --output-dir "$out" \
    --model-variant projection \
    --protein-model "$PROTEIN_MODEL" \
    --drug-model "$DRUG_MODEL" \
    --protein-max-length 1024 \
    --drug-max-length 512 \
    --epochs 30 \
    --patience 6 \
    --batch-size 4 \
    --gradient-accumulation-steps 16 \
    >> "$LOG" 2>&1
  local status=$?
  echo "===== $(date '+%F %T') DONE BALM strict ${split} seed=${seed} status=${status} =====" | tee -a "$LOG"
  return "$status"
}

run_one 0 assay_cold 42 &
run_one 5 assay_cold 43 &
run_one 6 assay_cold 44 &
run_one 7 assay_cold 45 &
wait

run_one 0 assay_cold 46 &
run_one 5 target_cold 42 &
run_one 6 target_cold 43 &
run_one 7 target_cold 44 &
wait

run_one 0 target_cold 45 &
run_one 5 target_cold 46 &
run_one 6 temporal 42 &
run_one 7 temporal 43 &
wait

run_one 0 temporal 44 &
run_one 5 temporal 45 &
run_one 6 temporal 46 &
wait

echo "===== $(date '+%F %T') ALL BALM STRICT RUNS FINISHED =====" | tee -a "$LOG"

