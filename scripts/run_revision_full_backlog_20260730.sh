#!/usr/bin/env bash
set -u

REPO="/home/test/wsk/mirage_drug_miniprototype"
cd "${REPO}" || exit 1

PY=".venv/bin/python"
LOGDIR="logs/revision_full_backlog_20260730"
mkdir -p "${LOGDIR}"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

SMILES_MODEL="/home/test/wsk/hf_models/chemberta_zinc_base_v1"
TEXT_MODEL="/home/test/wsk/hf_models/all_minilm_l6_v2"
SEQ_MODEL="/home/test/wsk/hf_models/esm2_t6_8m_ur50d"

stamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(stamp)] $*"
}

run_or_skip_tdc() {
  local dataset="$1"
  local split="$2"
  local seed="$3"
  local out="outputs/revision_e03_tdc_fixed_localhf_20260730/${dataset}_${split}_s${seed}"
  local logfile="${LOGDIR}/E03_TDC_${dataset}_${split}_s${seed}.log"
  if [[ -s "${out}/metrics.json" && -s "${out}/predictions.csv" ]]; then
    log "SKIP E03-TDC ${dataset} ${split} seed=${seed}"
    return 0
  fi
  mkdir -p "${out}"
  log "START E03-TDC ${dataset} ${split} seed=${seed}"
  "${PY}" scripts/run_tdc_official_experiment.py \
    --dataset "${dataset}" \
    --split-method "${split}" \
    --seed "${seed}" \
    --output-dir "${out}" \
    --missing-sequence-prob 0.35 \
    --missing-text-prob 0.35 \
    --n-neighbors 8 \
    --retrieval-reference-size 5000 \
    --enable-pretrained-smiles \
    --pretrained-smiles-model "${SMILES_MODEL}" \
    --enable-pretrained-text \
    --pretrained-text-model "${TEXT_MODEL}" \
    --enable-pretrained-sequence \
    --pretrained-sequence-model "${SEQ_MODEL}" \
    > "${logfile}" 2>&1
  local status=$?
  log "DONE E03-TDC ${dataset} ${split} seed=${seed} status=${status}"
  return "${status}"
}

run_or_skip_chembl_sensitivity() {
  local name="$1"
  shift
  local out="outputs/${name}"
  local logfile="${LOGDIR}/${name}.log"
  local count=0
  if [[ -d "${out}" ]]; then
    count=$(find "${out}" -name metrics.json 2>/dev/null | wc -l | tr -d " ")
  fi
  if [[ "${count}" -ge 9 ]]; then
    log "SKIP ${name} metrics=${count}"
    return 0
  fi
  log "START ${name}"
  "${PY}" scripts/run_chembl_assay_v2_eval.py \
    --benchmark-dir outputs/chembl_assay_primary_20260729 \
    --output-dir "${out}" \
    --splits assay_cold target_cold temporal \
    --seeds 42 43 44 \
    --missing-sequence-prob 0.35 \
    --missing-text-prob 0.35 \
    "$@" \
    > "${logfile}" 2>&1
  local status=$?
  log "DONE ${name} status=${status}"
  return "${status}"
}

collect_run_dirs() {
  find \
    outputs/revision_e03_chembl_full_20260729 \
    outputs/revision_e10_davis_official_localhf_20260729 \
    outputs/revision_e03_tdc_fixed_localhf_20260730 \
    -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
}

run_posthoc_reports() {
  mapfile -t RUN_DIRS < <(collect_run_dirs)
  if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
    log "SKIP posthoc reports: no run dirs found"
    return 0
  fi
  log "START E02 shortcut audit all runs n=${#RUN_DIRS[@]}"
  "${PY}" scripts/build_retrieval_shortcut_audit.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --output-dir outputs/revision_e02_shortcut_all_20260730 \
    --k 8 \
    --chunk-size 192 \
    > "${LOGDIR}/E02_shortcut_all.log" 2>&1
  log "DONE E02 shortcut audit status=$?"

  log "START E12 conflict mechanism all runs"
  "${PY}" scripts/build_conflict_analysis.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --output-dir outputs/revision_e12_conflict_all_20260730 \
    --current-model mask \
    --retrieval-model historical_retrieval_evidence \
    --fusion-models mirage_full mirage_w_o_gate mirage_w_o_probe mirage_w_o_anchor hybrid_blend_avg \
    --metric-splits val test_clean test_missing \
    --n-bins 4 \
    > "${LOGDIR}/E12_conflict_all.log" 2>&1
  log "DONE E12 conflict status=$?"

  log "START E04 regression bridge all runs"
  "${PY}" scripts/build_official_regression_report.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --output-dir outputs/revision_e04_regression_bridge_all_20260730 \
    --models mirage_full mask historical_retrieval_evidence retrieval hybrid_blend_avg \
    > "${LOGDIR}/E04_bridge_all.log" 2>&1
  log "DONE E04 bridge status=$?"

  log "START E04 direct regression all runs"
  "${PY}" scripts/run_revision_direct_regression_from_runs.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --output-dir outputs/revision_e04_direct_regression_all_20260730 \
    --k 8 \
    --chunk-size 192 \
    --max-train 30000 \
    --max-val 8000 \
    --max-test 8000 \
    > "${LOGDIR}/E04_direct_regression_all.log" 2>&1
  log "DONE E04 direct regression status=$?"
}

run_external_deepdta_standard() {
  local base="outputs/revision_e08_external_deepdta_standard_20260730"
  for dataset in BindingDB_Kd DAVIS KIBA; do
    for split in cold_drug cold_target cold_pair; do
      for seed in 42 43 44; do
        local out="${base}/${dataset}_${split}_s${seed}"
        local logfile="${LOGDIR}/E08_DeepDTA_${dataset}_${split}_s${seed}.log"
        if [[ -s "${out}/external_metrics.json" ]]; then
          log "SKIP E08 DeepDTA ${dataset} ${split} seed=${seed}"
          continue
        fi
        mkdir -p "${out}"
        log "START E08 DeepDTA ${dataset} ${split} seed=${seed}"
        "${PY}" scripts/run_external_deepdta_official.py \
          --dataset "${dataset}" \
          --split-method "${split}" \
          --seed "${seed}" \
          --output-dir "${out}" \
          --epochs 80 \
          --patience 10 \
          --batch-size 256 \
          > "${logfile}" 2>&1
        log "DONE E08 DeepDTA ${dataset} ${split} seed=${seed} status=$?"
      done
    done
  done
}

run_external_graph_family_if_available() {
  local graph_repo=""
  local mgraph_repo=""
  graph_repo=$(find /home/test/wsk -maxdepth 4 -type d -iname "GraphDTA*" 2>/dev/null | head -1 || true)
  mgraph_repo=$(find /home/test/wsk -maxdepth 4 -type d -iname "MGraphDTA*" 2>/dev/null | head -1 || true)

  if [[ -n "${graph_repo}" ]]; then
    local base="outputs/revision_e08_external_graphdta_standard_20260730"
    for dataset in BindingDB_Kd DAVIS KIBA; do
      for split in cold_drug cold_target cold_pair; do
        for seed in 42 43 44; do
          local out="${base}/${dataset}_${split}_s${seed}"
          if [[ -s "${out}/external_metrics.json" ]]; then
            continue
          fi
          mkdir -p "${out}"
          log "START E08 GraphDTA ${dataset} ${split} seed=${seed}"
          "${PY}" scripts/run_external_graphdta_official.py \
            --graphdta-repo "${graph_repo}" \
            --dataset "${dataset}" \
            --split-method "${split}" \
            --seed "${seed}" \
            --output-dir "${out}" \
            --epochs 80 \
            --patience 10 \
            > "${LOGDIR}/E08_GraphDTA_${dataset}_${split}_s${seed}.log" 2>&1
          log "DONE E08 GraphDTA ${dataset} ${split} seed=${seed} status=$?"
        done
      done
    done
  else
    log "SKIP GraphDTA external rerun: repo not found"
  fi

  if [[ -n "${mgraph_repo}" ]]; then
    local base="outputs/revision_e08_external_mgraphdta_standard_20260730"
    for dataset in BindingDB_Kd DAVIS KIBA; do
      for split in cold_drug cold_target cold_pair; do
        for seed in 42 43 44; do
          local out="${base}/${dataset}_${split}_s${seed}"
          if [[ -s "${out}/external_metrics.json" ]]; then
            continue
          fi
          mkdir -p "${out}"
          log "START E08 MGraphDTA ${dataset} ${split} seed=${seed}"
          "${PY}" scripts/run_external_mgraphdta_official.py \
            --mgraphdta-repo "${mgraph_repo}" \
            --dataset "${dataset}" \
            --split-method "${split}" \
            --seed "${seed}" \
            --output-dir "${out}" \
            --epochs 80 \
            --patience 10 \
            > "${LOGDIR}/E08_MGraphDTA_${dataset}_${split}_s${seed}.log" 2>&1
          log "DONE E08 MGraphDTA ${dataset} ${split} seed=${seed} status=$?"
        done
      done
    done
  else
    log "SKIP MGraphDTA external rerun: repo not found"
  fi
}

log "QUEUE START revision_full_backlog_20260730"

log "PHASE 1: standard DTA fixed-model reruns"
for dataset in BindingDB_Kd KIBA; do
  for split in cold_drug cold_target cold_pair; do
    for seed in 42 43 44; do
      run_or_skip_tdc "${dataset}" "${split}" "${seed}" || true
    done
  done
done

log "PHASE 2: posthoc reports after available fixed-model runs"
run_posthoc_reports || true

log "PHASE 3: retrieval sensitivity on CHEMBL_ASSAY"
run_or_skip_chembl_sensitivity revision_e14_k1_sensitivity_20260730 --n-neighbors 1 --retrieval-reference-size 5000 || true
run_or_skip_chembl_sensitivity revision_e14_k4_sensitivity_20260730 --n-neighbors 4 --retrieval-reference-size 5000 || true
run_or_skip_chembl_sensitivity revision_e14_k16_sensitivity_20260730 --n-neighbors 16 --retrieval-reference-size 5000 || true
run_or_skip_chembl_sensitivity revision_e14_ref1000_sensitivity_20260730 --n-neighbors 8 --retrieval-reference-size 1000 || true
run_or_skip_chembl_sensitivity revision_e14_ref20000_sensitivity_20260730 --n-neighbors 8 --retrieval-reference-size 20000 || true

log "PHASE 4: external standard DTA baselines"
run_external_deepdta_standard || true
run_external_graph_family_if_available || true

log "PHASE 5: final posthoc refresh including any late external/fixed outputs"
run_posthoc_reports || true

log "QUEUE FINISHED revision_full_backlog_20260730"

