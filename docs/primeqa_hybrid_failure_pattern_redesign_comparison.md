# PrimeQA Hybrid Failure-Pattern Redesign Comparison

This document records Stage 109.

## Scope

Stage 109 implements the Stage108 frozen failure-pattern redesign comparison.
It compares the 7 frozen candidate configs against the Stage102 verified BM25
top10 answer-pipeline baseline.

This stage loads only the frozen Stage68 train/dev split files and the PrimeQA
training/dev document file. It uses train grouped cross-validation for
selection and dev only as a single validation pass. It does not load the frozen
test split, does not run final metrics, does not use dev for config selection
or threshold tuning, does not add fallback strategies, and does not change
runtime defaults.

The comparison was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: user confirmed Stage109 train grouped-CV plus dev validation comparison on 2026-07-16 after Stage108 protocol freeze; test locked; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\run_primeqa_hybrid_failure_pattern_redesign_comparison.py ^
  --user-confirmed-comparison ^
  --confirmation-note "user confirmed Stage109 train grouped-CV plus dev validation comparison on 2026-07-16 after Stage108 protocol freeze; test locked; runtime defaults unchanged; no fallback strategies"
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109.json
```

The run completed successfully in `203.503s` according to the Stage109 timing
block.

## Loaded Data

```text
documents: 28482
train rows: 562
train answerable rows: 370
train unanswerable rows: 192
dev rows: 121
dev answerable rows: 76
dev unanswerable rows: 45
test_split_loaded: false
```

Train grouped CV used 5 folds:

| Fold | Rows | Groups |
| --- | ---: | ---: |
| fold_1 | 113 | 107 |
| fold_2 | 113 | 107 |
| fold_3 | 112 | 106 |
| fold_4 | 112 | 106 |
| fold_5 | 112 | 107 |

Raw group values were not written to the report.

## Baseline

```text
baseline_config_id: stage102_verified_bm25_top10_answer_pipeline
train-CV weighted target score: 728.70
train-full weighted target score: 728.70
dev weighted target score: 163.55

train-CV verified:
  answerable_refusal_rate: 0.0459
  unanswerable_refusal_rate: 0.0625
  gold_doc_citation_rate: 0.4958
  average_token_f1: 0.2017

dev verified:
  answerable_refusal_rate: 0.1053
  unanswerable_refusal_rate: 0.0889
  gold_doc_citation_rate: 0.6029
  average_token_f1: 0.2040
```

## Train-CV Selection

Train-CV objective:

```text
1.75 * answerability_false_answer
+ 1.80 * gold_span_beats_selected_answer
+ 1.50 * evidence_selection_miss
```

Train-CV selectability guards:

```text
max_train_cv_answerable_refusal_rate_delta: 0.02
max_train_cv_average_token_f1_drop: 0.005
max_train_cv_gold_doc_citation_rate_drop: 0.015
max_train_cv_retrieval_context_miss_delta: 0
requires_negative_train_cv_weighted_delta: true
no_op_candidate_selectable: false
```

Train-CV ranking:

| Rank | Config | Train-CV delta | Selectable | Changed answers |
| ---: | --- | ---: | --- | ---: |
| 1 | jsgc_support2_evidence7_anchor_top2_v1 | -199.65 | false | 534 |
| 2 | cpsc_title_query_anchor_top2_mcpd3_rank3_v1 | -123.05 | false | 543 |
| 3 | jsgc_support2_evidence6_title_anchor_top2_v1 | -115.90 | false | 551 |
| 4 | saag_support2_evidence7_rank3_v1 | -111.15 | false | 117 |
| 5 | saag_support2_evidence6_rank5_v1 | -31.65 | false | 49 |
| 6 | cpsc_anchor_top2_mcpd3_rank3_v1 | -28.50 | false | 533 |
| 7 | cpsc_anchor_top3_mcpd3_rank3_v1 | 0.00 | false | 0 |

No candidate was train-CV selectable:

```text
selected_config_id: null
selected_candidate_family_id: null
selectable_config_count: 0 / 7
```

## Guard Failures

The main pattern is clear: the redesigned configs reduce the weighted target
bucket score, but they do it by refusing too many answerable questions under
the frozen guard.

| Config | Train-CV delta | Dev delta | Failed train-CV guards |
| --- | ---: | ---: | --- |
| saag_support2_evidence7_rank3_v1 | -111.15 | -24.75 | answerable_refusal_rate_delta |
| saag_support2_evidence6_rank5_v1 | -31.65 | -1.75 | answerable_refusal_rate_delta |
| cpsc_anchor_top2_mcpd3_rank3_v1 | -28.50 | -0.30 | answerable_refusal_rate_delta, average_token_f1_drop, gold_doc_citation_rate_drop |
| cpsc_anchor_top3_mcpd3_rank3_v1 | 0.00 | 0.00 | train_cv_weighted_target_delta_negative |
| cpsc_title_query_anchor_top2_mcpd3_rank3_v1 | -123.05 | -47.15 | answerable_refusal_rate_delta |
| jsgc_support2_evidence7_anchor_top2_v1 | -199.65 | -48.25 | answerable_refusal_rate_delta |
| jsgc_support2_evidence6_title_anchor_top2_v1 | -115.90 | -34.95 | answerable_refusal_rate_delta |

Observed answerable-refusal deltas were much larger than the allowed `0.02`
for most configs:

```text
jsgc_support2_evidence7_anchor_top2_v1: +0.3784
cpsc_title_query_anchor_top2_mcpd3_rank3_v1: +0.3027
jsgc_support2_evidence6_title_anchor_top2_v1: +0.2352
saag_support2_evidence7_rank3_v1: +0.1919
saag_support2_evidence6_rank5_v1: +0.0568
cpsc_anchor_top2_mcpd3_rank3_v1: +0.0541
cpsc_anchor_top3_mcpd3_rank3_v1: +0.0000
```

The `cpsc_anchor_top3_mcpd3_rank3_v1` config was effectively a no-op and was
blocked by the frozen no-op rule.

## Dev Validation

Dev was not used for selection or tuning. Because train-CV selected no config,
there was no selected candidate to validate on dev.

```text
validation_split: dev
selected_config_id: null
status: no_train_cv_selectable_config
dev_validation_passed: false
```

Several configs also improved the dev weighted target score, but they had
already failed train-CV selectability. They cannot be selected from dev.

## Decision

```text
status: primeqa_hybrid_failure_pattern_redesign_completed_no_train_cv_selectable_config
selected_config_id: null
selectable_config_count: 0
dev_validation_passed: false
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_direction: record_failure_pattern_redesign_stop_decision
recommended_next_stage: Stage110: record a stop decision for the frozen Stage108 redesign family because no candidate satisfied train-CV selectability.
```

Stage109 therefore does not justify runtime defaultization and does not open the
final test gate.

## Guard Checks

All `20 / 20` Stage109 guard checks passed.

Key checks:

```text
user_confirmed_stage109_comparison: passed
only_train_dev_splits_loaded: passed
loaded_samples_are_train_dev_only: passed
candidate_config_count_matches_stage108: passed
train_cv_fold_count_matches_protocol: passed
train_cv_selection_uses_train_only_fields: passed
train_cv_selection_blocks_noop_candidates: passed
dev_validation_not_used_for_selection: passed
stage109_final_test_metrics_not_run: passed
stage109_runtime_defaults_unchanged: passed
stage109_fallback_strategies_not_added: passed
public_outputs_have_no_forbidden_keys: passed
```

## Validation

Targeted validation:

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_comparison.py scripts\run_primeqa_hybrid_failure_pattern_redesign_comparison.py tests\test_primeqa_hybrid_failure_pattern_redesign_comparison.py
python -m pytest tests\test_primeqa_hybrid_failure_pattern_redesign_comparison.py -q
python scripts\run_primeqa_hybrid_failure_pattern_redesign_comparison.py --user-confirmed-comparison ...
```

Full validation:

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

Result:

```text
targeted ruff: passed
targeted pytest: 2 passed
Stage109 run: passed
guard checks: 20 / 20 passed
full ruff: passed
full pytest: 297 passed
git diff --check: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_train_cv_weighted_target_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_dev_weighted_target_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_train_cv_selectability.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_changed_answer_counts.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_dev_metric_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_decision_flags.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_guard_check_status.svg
```

## What I Learned

- The frozen Stage108 redesign family did move the target bucket score in the
  desired direction on train-CV, but the movement was not acceptable because it
  substantially increased answerable refusals.
- The strongest train-CV target-score improvement,
  `jsgc_support2_evidence7_anchor_top2_v1`, had a `-199.65` target delta but
  also an answerable-refusal delta of `+0.3784`, far beyond the allowed `0.02`.
- The milder support gate `saag_support2_evidence6_rank5_v1` still exceeded the
  answerable-refusal guard with `+0.0568`, so simply lowering the evidence
  threshold was not enough.
- The no-op guard worked as intended: `cpsc_anchor_top3_mcpd3_rank3_v1` changed
  nothing and could not advance.
- The honest next step is a stop decision for the frozen Stage108 redesign
  family, not a dev-selected candidate, not a runtime change, and not final-test
  evaluation.

## Next Step

Stage110: record a stop decision for the frozen Stage108 failure-pattern
redesign family. The stop decision should explain why all candidates failed
train-CV selectability, why dev cannot rescue them, and keep test locked,
runtime defaults unchanged, and fallback strategies disabled.
