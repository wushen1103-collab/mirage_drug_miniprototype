# MIRAGE-DTA Revision Method Lock

## Fixed primary model

The revised primary model is `mirage_full`. It is fixed before test evaluation and is not the validation-selected `fullsuite_val_select` output. The primary model uses the mask-aware direct current-evidence probability `p_c` from `mask` and a distinct training-memory probability `p_r` from `historical_retrieval_evidence`.

| Paper component | Operational definition | Code location |
| --- | --- | --- |
| Current evidence | Mask-aware direct classifier over sparse molecular, sequence, and text features. Assay context and target text are concatenated into the text field. The direct branch deliberately does not consume retrieval statistics. | `run_model_suite`, current branch `mask` |
| Historical evidence | A separate classifier consumes only multi-view retrieval statistics computed from the training partition. | `MultiViewRetrievalAugmentor`, `historical_retrieval_evidence` |
| Self exclusion | A training query has its own memory row set to negative infinity before top-K selection; the nearest legal neighbor is retained. | `retrieval.py`, `_cosine_topk_block`, `transform_train` |
| Gate | Cross-fitted logistic gate predicts whether the current branch has lower squared error than the retrieval branch. Its output is the current-branch weight `gamma`. | `_fit_conflict_aware_arbitration`, `fit_arbitration_gate` |
| Reliability probe | Logistic probe predicts correctness of the cross-fitted gated decision. It returns `r` in `[0,1]`; cross-fitted validation predictions are used for anchor calibration. | `_fit_conflict_aware_arbitration`, `fit_reliability_probe` |
| Anchor | Training-partition class prevalence `p_a`, with strength `a` selected from a fixed grid by cross-fitted validation Brier score. It is applied only to the low-reliability residual. | `_fit_conflict_aware_arbitration` |

For a test query, the gate forms

`p_g = gamma p_c + (1 - gamma) p_r`,

and the fixed full model returns

`p = r p_g + (1 - r)[(1 - a)p_g + a p_a]`.

The component ablations are fixed as follows: `mirage_w_o_gate` replaces `p_g` with equal fusion, `mirage_w_o_probe` replaces learned reliability with a fixed 0.5 reliability value, and `mirage_w_o_anchor` reports `p_g` without anchor shrinkage.

## Evaluation restrictions

1. The retrieval memory contains only training rows.
2. Validation rows are used for gate cross-fitting, reliability-probe fitting, ordinary blend tuning, and early stopping where applicable.
3. No test label is used by retrieval, gate fitting, probe fitting, threshold construction, or model selection.
4. The CHEMBL primary binary task uses an absolute pChEMBL threshold. Within-assay q40/q60 labels are retrospective auxiliary analyses only.

## Reproducibility outputs

Each formal run writes `metrics.json`, `predictions.csv`, `validation_predictions.csv`, and `train_preview.csv`. Predictions contain `mirage_gate_weight_*` and `mirage_reliability_*` fields for mechanism analysis.
