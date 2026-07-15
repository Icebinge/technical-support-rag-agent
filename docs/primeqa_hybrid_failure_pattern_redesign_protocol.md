# PrimeQA Hybrid Failure-Pattern Redesign Protocol

This document records Stage 108.

## Scope

Stage 108 freezes a train/dev-only protocol for the next failure-pattern-driven
answer-pipeline redesign. It is a protocol freeze only.

This stage reads only the saved public-safe Stage107 report. It does not load
train/dev/test split files, does not load corpus documents, does not run
retrieval or answer metrics, does not run final metrics, does not select from
dev-only observations, does not add fallback strategies, and does not change
runtime defaults.

The protocol freeze was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: user confirmed Stage108 failure-pattern redesign protocol freeze on 2026-07-16 after Stage107 validation-failure analysis; test locked; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\freeze_primeqa_hybrid_failure_pattern_redesign_protocol.py ^
  --user-confirmed-protocol ^
  --confirmation-note "user confirmed Stage108 failure-pattern redesign protocol freeze on 2026-07-16 after Stage107 validation-failure analysis; test locked; runtime defaults unchanged; no fallback strategies"
```

The run completed successfully and wrote console output to:

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108.console.txt
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108.json
```

## Stage107 Basis

Stage108 uses only the Stage107 public-safe aggregate findings:

```text
dev_failure_count: 117
dev_failure_rate: 0.9669
answerable_failure_rate: 1.0
unanswerable_false_answer_rate: 0.9111
answerable_gold_context_absent_rate: 0.3026
context_present_gold_span_beats_selected_rate: 0.7736
context_present_evidence_selection_miss_rate: 0.2264
stage105_selected_config_was_dev_noop: true
stage105_dev_better_nonselectable_config_count: 7
```

The key redesign implication is:

```text
retrieval misses remain, but 53 answerable dev rows had gold context in top10
and still failed in evidence selection or answer composition. Stage109 should
therefore target answer-pipeline behavior first, while monitoring retrieval
context misses as a boundary.
```

## Frozen Candidate Families

Stage108 freezes three redesigned candidate families and seven total configs.
These are not runtime defaults; they are only the Stage109 comparison grid.

| Family | Configs | Primary target | Main guard risk |
| --- | ---: | --- | --- |
| support_aware_answerability_gate_candidate_v1 | 2 | answerability_false_answer | answerable over-refusal |
| context_present_span_composer_candidate_v1 | 3 | gold_span_beats_selected_answer, evidence_selection_miss | citation or token-F1 regression |
| joint_support_gate_span_composer_candidate_v1 | 2 | all three target buckets | combined refusal and citation regression |

Frozen config IDs:

```text
saag_support2_evidence7_rank3_v1
saag_support2_evidence6_rank5_v1
cpsc_anchor_top2_mcpd3_rank3_v1
cpsc_anchor_top3_mcpd3_rank3_v1
cpsc_title_query_anchor_top2_mcpd3_rank3_v1
jsgc_support2_evidence7_anchor_top2_v1
jsgc_support2_evidence6_title_anchor_top2_v1
```

Stage108 intentionally does not reuse the stopped Stage104 config IDs
(`amg_*`, `ewr_*`, `jgw_*`).

## Train Selection Rule

Stage109 must select only from train grouped CV:

```text
selection_split: train
selection_mode: train_grouped_cross_validation_then_full_train_refit
train_cv_fold_count: 5
dev_config_selection_allowed: false
dev_threshold_tuning_allowed: false
test_access_allowed: false
```

The grouping policy is evaluation-only:

```text
group_key_inputs:
  normalized_question_text_for_grouping_only
  gold_document_group_marker_for_grouping_only
  unanswerable_marker_for_grouping_only
raw_group_values_written_to_reports: false
group_keys_allowed_as_runtime_features: false
dev_rows_used_for_grouping: false
test_rows_used_for_grouping: false
```

Train-CV objective:

```text
1.75 * answerability_false_answer
1.80 * gold_span_beats_selected_answer
1.50 * evidence_selection_miss
```

No-op candidates are not selectable:

```text
requires_negative_train_cv_weighted_delta: true
no_op_candidate_selectable: false
```

This directly addresses Stage105, where the selected config passed train guards
but changed zero dev answers and had zero target improvement.

## Selectability Guards

Stage109 train-CV guards are stricter than Stage104 because Stage105 showed
that aggressive changes can reduce target buckets while failing answerable
refusal or citation safety:

```text
max_train_cv_answerable_refusal_rate_delta: 0.02
max_train_cv_average_token_f1_drop: 0.005
max_train_cv_gold_doc_citation_rate_drop: 0.015
max_train_cv_retrieval_context_miss_delta: 0
```

## Dev Validation Rule

Dev remains one validation pass of the single train-CV-selected config:

```text
validation_split: dev
validated_item: single train-CV-selected config
dev_selection_allowed: false
dev_retuning_allowed: false
dev_threshold_tuning_allowed: false
test_access_allowed: false
```

Dev pass conditions:

```text
dev_weighted_target_delta_must_be_negative: true
dev_answerable_refusal_rate_delta_must_not_exceed: 0.02
dev_average_token_f1_drop_must_not_exceed: 0.005
dev_gold_doc_citation_rate_drop_must_not_exceed: 0.015
```

## Runtime Feature Boundary

Allowed runtime signals:

```text
question_route
retrieved_document_rank
retrieval_score
evidence_sentence_score
evidence_support_count
question_title_overlap
question_text_overlap
citation_rank
selected_evidence_window_position
```

Forbidden runtime signals:

```text
gold_answer_text
gold_document_identifier
dataset_split_membership
validation_or_test_label
source_candidate_document_identifier_list
raw_private_document_text_as_reported_feature
```

Stage108 does not change runtime defaults.

## Decision

```text
status: primeqa_hybrid_failure_pattern_redesign_protocol_frozen
protocol_id: primeqa_hybrid_failure_pattern_redesign_protocol_v1
recommended_next_direction: run_failure_pattern_redesign_train_cv_dev_validation
candidate_family_count: 3
candidate_config_count: 7
train_selection_mode: train_grouped_cross_validation_then_full_train_refit
can_run_train_dev_comparison_after_user_confirmation: true
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Stage108 does not justify runtime defaultization and does not open the final-test
gate.

## Guard Checks

All `23 / 23` Stage108 guard checks passed.

## Visualizations

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_candidate_family_priorities.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_candidate_config_counts.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_target_bucket_weights.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_train_cv_guard_thresholds.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_protocol_decision_flags.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_guard_check_status.svg
```

## Validation

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_protocol.py scripts\freeze_primeqa_hybrid_failure_pattern_redesign_protocol.py tests\test_primeqa_hybrid_failure_pattern_redesign_protocol.py
pytest -q tests\test_primeqa_hybrid_failure_pattern_redesign_protocol.py
python scripts\freeze_primeqa_hybrid_failure_pattern_redesign_protocol.py --user-confirmed-protocol ...
```

Result:

```text
ruff: passed
pytest: 4 passed
Stage108 run: passed
guard checks: 23 / 23 passed
```

Full validation is recorded in `docs\learning_journal.md`.

## Next Step

Stage109 should happen only after user confirmation. It should implement the
frozen candidate components and run the train grouped-CV plus dev validation
comparison. It must select only from train-CV, validate once on dev, keep test
locked, keep runtime defaults unchanged, and add no fallback strategies.
