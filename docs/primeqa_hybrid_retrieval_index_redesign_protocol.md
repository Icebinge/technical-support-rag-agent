# PrimeQA Hybrid Retrieval/Index Redesign Protocol

This document records Stage 113.

## Scope

Stage 113 freezes the train/dev-only retrieval/index redesign protocol based on
the Stage112 `retrieval_context_miss` root-cause audit.

This is a protocol freeze only. It reads only the saved public-safe Stage112
report. It does not load train/dev/test split files, does not load corpus
documents, does not run retrieval or answer metrics, does not run final metrics,
does not select from dev-only observations, does not add fallback strategies,
and does not change runtime defaults.

The protocol freeze was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: user confirmed Stage113 train-dev retrieval/index redesign protocol freeze on 2026-07-16 after Stage112 root-cause audit; protocol freeze only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\freeze_primeqa_hybrid_retrieval_index_redesign_protocol.py ^
  --user-confirmed-protocol ^
  --confirmation-note "user confirmed Stage113 train-dev retrieval/index redesign protocol freeze on 2026-07-16 after Stage112 root-cause audit; protocol freeze only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113.json
```

The run completed successfully in `0.002s` according to the Stage113 timing
block.

## Stage112 Basis

Stage112 audit cases:

```text
train: 125
dev: 23
total: 148
audit_case_rate_among_answerable: 0.3318
```

Primary root causes:

| Root cause | Count |
| --- | ---: |
| title_heading_mismatch | 74 |
| query_expression_gap | 65 |
| long_document_score_dilution | 4 |
| entity_version_error_code_mismatch | 3 |
| bm25_field_weighting_or_index_structure | 2 |

High-signal diagnostic dimensions:

| Dimension | Count |
| --- | ---: |
| title_heading_mismatch | 137 |
| bm25_field_weighting_or_index_structure | 121 |
| entity_version_error_code_mismatch | 80 |
| query_expression_gap | 65 |
| long_document_score_dilution | 37 |
| section_boundary_or_span_locality | 10 |

Gold document diagnostic rank buckets:

```text
not_found_top50: 110
rank_21_to_50: 24
rank_11_to_20: 14
```

Stage113 therefore targets retrieval/index structure rather than answer
composition or answerability thresholding.

## Candidate Families

| Family | Priority | Configs | Stage112 basis |
| --- | ---: | ---: | --- |
| title_heading_weighted_bm25_candidate_v1 | 0.95 | 3 | `title_heading_mismatch` 74 primary cases, 137 high-signal cases, plus `query_expression_gap` 65 primary cases |
| section_level_index_rollup_candidate_v1 | 0.90 | 3 | `bm25_field_weighting_or_index_structure` 121 high-signal cases and 110 `not_found_top50` cases |
| entity_version_error_code_handling_candidate_v1 | 0.80 | 2 | `entity_version_error_code_mismatch` 80 high-signal cases |

All candidate families use only runtime-visible question/document text signals.
Gold document identifiers, gold answer spans, and test labels are forbidden as
runtime features.

## Frozen Candidate Configs

| Config | Family | Retrieval mode |
| --- | --- | --- |
| thw_title2_heading2_body1_doc_bm25_v1 | title_heading_weighted_bm25_candidate_v1 | weighted_document_bm25 |
| thw_title3_heading2_body1_doc_bm25_v1 | title_heading_weighted_bm25_candidate_v1 | weighted_document_bm25 |
| thw_title_heading_query_view_rrf_v1 | title_heading_weighted_bm25_candidate_v1 | document_bm25_rrf |
| slr_section_top1_doc_rollup_v1 | section_level_index_rollup_candidate_v1 | section_bm25_document_rollup |
| slr_section_top3_rrf_doc_rollup_v1 | section_level_index_rollup_candidate_v1 | section_document_rrf |
| slr_heading_section_title_rollup_v1 | section_level_index_rollup_candidate_v1 | heading_section_title_rollup |
| evc_special_token_exact_boost_v1 | entity_version_error_code_handling_candidate_v1 | bm25_with_runtime_special_token_boost |
| evc_special_token_title_heading_boost_v1 | entity_version_error_code_handling_candidate_v1 | weighted_bm25_with_special_token_boost |

These are protocol candidates only. Stage113 does not implement or evaluate
them.

## Selection Rules

Stage114, if confirmed, must use:

```text
selection_split: train
selection_mode: train_grouped_cross_validation_then_full_train_refit
train_group_key: normalized_question_plus_answer_document_or_technote
minimum_train_folds: 5
validation_split: dev
dev_validation_mode: single_pass_no_retuning
```

Primary objective:

```text
reduce_retrieval_context_miss
required_train_cv_delta: negative
weight: 2.0
```

Secondary objectives:

```text
improve_gold_doc_recall_at_10: positive train-CV delta, weight 1.5
avoid_average_token_f1_regression: max train-CV drop 0.005
avoid_gold_doc_citation_rate_regression: max train-CV drop 0.015
```

Train-CV guard thresholds:

```text
max_train_cv_average_token_f1_drop: 0.005
max_train_cv_gold_doc_citation_rate_drop: 0.015
max_train_cv_answerable_refusal_rate_delta: 0.02
max_train_cv_answerability_false_answer_delta: 0
max_train_cv_evidence_selection_miss_delta: 0
max_train_cv_gold_span_beats_selected_delta: 0
max_train_cv_changed_answer_rate: 0.25
```

A config is forbidden if it:

```text
has nonnegative train-CV retrieval_context_miss delta
exceeds any train-CV regression guard
is a no-op
uses gold document identifiers as runtime features
uses test data
```

Dev rules:

```text
dev_selection_allowed: false
dev_retuning_allowed: false
dev_threshold_tuning_allowed: false
dev_report_required: true
```

Test rules:

```text
test_access_allowed: false
final_test_metrics_allowed: false
test_tuning_allowed: false
```

Runtime rules:

```text
default_runtime_policy: unchanged
fallback_strategies_enabled: false
```

## Decision

```text
status: primeqa_hybrid_retrieval_index_redesign_protocol_frozen
protocol_id: primeqa_hybrid_retrieval_index_redesign_protocol_v1
recommended_next_direction: run_retrieval_index_redesign_train_cv_dev_validation
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Stage113 therefore provides no justification for runtime defaultization or
final-test gate opening.

## Guard Checks

All `20 / 20` guard checks passed.

Key checks:

```text
source_stage112_is_expected: passed
source_stage112_analysis_id_matches: passed
stage112_audit_completed: passed
user_confirmed_stage113_protocol: passed
stage112_test_split_was_not_loaded: passed
stage112_audit_has_train_dev_cases: passed
required_primary_root_causes_present: passed
required_high_signal_dimensions_present: passed
candidate_families_match_stage112_findings: passed
candidate_configs_are_selection_eligible_and_nonempty: passed
selection_is_train_grouped_cv: passed
dev_is_validation_only: passed
test_access_forbidden: passed
runtime_defaults_unchanged: passed
fallback_strategies_disabled: passed
stage113_does_not_load_splits_or_corpus: passed
stage113_does_not_run_retrieval_or_answer_metrics: passed
stage113_final_test_metrics_not_run: passed
source_stage112_report_is_public_safe: passed
stage113_public_outputs_have_no_forbidden_keys: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_stage112_primary_root_causes.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_stage112_high_signal_dimensions.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_candidate_family_priorities.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_candidate_config_counts.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_selection_guard_thresholds.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_protocol_decision_flags.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_guard_check_status.svg
```

## Validation

Targeted validation:

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_index_redesign_protocol.py scripts\freeze_primeqa_hybrid_retrieval_index_redesign_protocol.py tests\test_primeqa_hybrid_retrieval_index_redesign_protocol.py
python -m pytest tests\test_primeqa_hybrid_retrieval_index_redesign_protocol.py -q
python scripts\freeze_primeqa_hybrid_retrieval_index_redesign_protocol.py --user-confirmed-protocol ...
```

Result:

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage113 run: passed
guard checks: 20 / 20 passed
```

Full validation is recorded in `docs\learning_journal.md`.

## Next Step

Stage114 should happen only after user confirmation. It should run the frozen
train grouped-CV retrieval/index redesign comparison and one dev validation
pass. It must keep test locked, avoid dev-only selection, avoid threshold
tuning, keep runtime defaults unchanged, and add no fallback strategies.
