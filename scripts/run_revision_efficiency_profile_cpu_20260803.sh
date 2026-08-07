#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/test/wsk/mirage_drug_miniprototype"
PY="$ROOT/.venv/bin/python"
BENCH="$ROOT/outputs/chembl_assay_primary_20260729"
OUT_BASE="$ROOT/outputs/revision_e15c_efficiency_profile_cpu_20260803"
LOG_DIR="$ROOT/logs/revision_e15c_efficiency_profile_cpu_20260803"
mkdir -p "$OUT_BASE" "$LOG_DIR"

cd "$ROOT"

if [[ ! -f "$OUT_BASE/mirage_assay_cold_s42/summary.csv" ]]; then
  /usr/bin/time -v -o "$LOG_DIR/mirage_assay_cold_s42.time.txt" \
    env CUDA_VISIBLE_DEVICES="" "$PY" scripts/run_chembl_assay_v2_eval.py \
      --benchmark-dir "$BENCH" \
      --output-dir "$OUT_BASE/mirage_assay_cold_s42" \
      --splits assay_cold \
      --seeds 42 \
      --missing-sequence-prob 0.35 \
      --missing-text-prob 0.35 \
      --n-neighbors 8 \
      --retrieval-reference-size 5000 \
      > "$LOG_DIR/mirage_assay_cold_s42.log" 2>&1
fi

"$PY" scripts/build_revision_efficiency_snapshot.py \
  --outputs-root outputs \
  --patterns "revision_e03_chembl_full_20260729/*" "revision_e03_tdc_fixed_localhf_20260730/*" \
  --output-dir outputs/revision_e15c_efficiency_profile_cpu_20260803/snapshot

echo "ALL_DONE efficiency_profile_cpu $(date -Is)" | tee -a "$LOG_DIR/master.log"
