# MIRAGE-DTA Mini Prototype

This repository contains the code used to prototype the MIRAGE-DTA study:
conflict-aware multimodal information fusion for drug-target affinity prediction
under missing modalities, cold-start entities, and distribution shifts.

## Associated manuscript

This repository accompanies the manuscript:

**Conflict-aware Evidence Arbitration for Multimodal Information Fusion under
Missing Modalities and Distribution Shifts: A Drug-Target Affinity Study**

Authors: Shenkuang Wu, Jing Zhang, Lianming Zhang, and Pingping Dong.

The code is released to support reproduction of the implemented models,
benchmark construction procedures, threshold-sensitivity analyses,
retrieval-shortcut audits, and evaluation summaries reported in the manuscript.

The repository is intentionally code-focused. Large generated artifacts are not
tracked here: cached datasets, model checkpoints, logs, external dependency
bundles, result tables, and plotting/LaTeX outputs are excluded from git.

## What is included

- `src/mirage_mini/`: core data loading, split construction, featurization,
  retrieval augmentation, fusion/gating models, metrics, reporting utilities,
  and external-baseline adapters.
- `scripts/run_*.py`: experiment entry points for MIRAGE-DTA internal runs,
  TDC/CHEMBL benchmark runs, threshold sensitivity, and external baselines.
- `scripts/build_*.py`: lightweight analysis/audit scripts used to summarize
  reliability, conflict stratification, retrieval shortcut exposure, DAVIS
  failure diagnostics, and bootstrap deltas. These are retained because they
  reproduce scientific claims, not because they render paper figures.
- `tests/`: unit tests and small synthetic-data checks for the pipeline and
  baseline adapters.

## What is not included

- `outputs/`: generated predictions, reports, checkpoints, and queues.
- `data_cache/`: downloaded or preprocessed benchmark caches.
- `logs/`, `wandb/`, `.venv/`, `vendor/`: local runtime artifacts.
- Paper figure/table rendering code and LaTeX manuscript files.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Some external-baseline scripts require the corresponding upstream repositories
or additional dependencies. The scripts expose `--*-repo` arguments where needed
and are designed to keep those third-party repositories outside this git repo.

## Quick smoke test

```bash
python -m pytest tests/test_pipeline.py tests/test_reporting.py
```

## Minimal internal experiment

```bash
python scripts/run_mini_experiment.py   --dataset BindingDB_Kd   --sample-size 3000   --cache-dir data_cache   --output-dir outputs/smoke_bindingdb
```

## Official-style TDC run

```bash
python scripts/run_tdc_official_experiment.py   --dataset KIBA   --split-mode cold_drug   --seed 42   --cache-dir data_cache   --output-dir outputs/tdc_kiba_cold_drug_s42
```

## CHEMBL threshold sensitivity

```bash
python scripts/run_chembl_threshold_sensitivity.py   --split-mode temporal   --seed 42   --cache-dir data_cache   --output-dir outputs/chembl_threshold_temporal_s42
```

## External baselines

External baselines are provided as adapters. Example:

```bash
python scripts/run_external_deepdta_official.py   --dataset CHEMBL_ASSAY   --split-method temporal   --seed 42   --cache-dir data_cache   --output-dir outputs/external_chembl_deepdta_temporal_s42
```

BALM, CASTER, DeepDTAGen, KANPM-DTA, MixingDTA, and other adapters may require
local clones of their official repositories. Keep those clones outside this repo
and pass their paths through the script arguments.

## Reproducibility policy

The manuscript's paper figures and tables are generated from result artifacts
outside this repository. This git repo is meant to provide the executable
research code: benchmark construction, training/evaluation, external-baseline
adapters, and claim-level audit scripts. Seed-level outputs can be released as a
separate supplementary artifact.

## Data availability

The raw datasets used by the study are publicly available from BindingDB, DAVIS,
KIBA, ChEMBL, and Therapeutics Data Commons. This repository provides the scripts
needed to reconstruct the processed benchmarks and evaluation summaries from
those public resources. Large intermediate outputs, downloaded caches, trained
checkpoints, and plotting/LaTeX artifacts are intentionally not tracked in git.
