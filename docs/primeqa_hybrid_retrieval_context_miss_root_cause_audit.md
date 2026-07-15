# PrimeQA Hybrid Retrieval-Context-Miss Root-Cause Audit

This document records Stage 112.

## Scope

Stage 112 runs the train/dev-only retrieval-context-miss root-cause audit
frozen in Stage111.

This is an audit result, not a candidate-selection result. It loads only the
Stage111 frozen protocol, Stage68 train/dev split files, and PrimeQA
training/dev corpus sections. It does not load the test split, does not run
final metrics, does not select or tune candidates from dev, does not add
fallback strategies, and does not change runtime defaults.

The audit was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: user confirmed Stage112 train-dev retrieval-context-miss root-cause audit on 2026-07-16; test locked; no final metrics; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\run_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py ^
  --user-confirmed-audit ^
  --confirmation-note "user confirmed Stage112 train-dev retrieval-context-miss root-cause audit on 2026-07-16; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112.json
```

The run completed successfully in `235.469s` according to the Stage112 timing
block.

## Data Loaded

```text
document_count: 28482
section_count: 216648
train_rows: 562
dev_rows: 121
train_answerable_rows: 370
dev_answerable_rows: 76
test_split_loaded: false
```

The audit set exactly matched the Stage102 `retrieval_context_miss` counts:

```text
train: 125
dev: 23
total: 148
```

Rates among answerable rows:

```text
train: 125 / 370 = 0.3378
dev: 23 / 76 = 0.3026
total: 148 / 446 = 0.3318
```

## Primary Root Causes

Primary root-cause buckets across train/dev:

| Root cause | Count |
| --- | ---: |
| title_heading_mismatch | 74 |
| query_expression_gap | 65 |
| long_document_score_dilution | 4 |
| entity_version_error_code_mismatch | 3 |
| bm25_field_weighting_or_index_structure | 2 |

By split:

| Root cause | Train | Dev |
| --- | ---: | ---: |
| title_heading_mismatch | 63 | 11 |
| query_expression_gap | 55 | 10 |
| entity_version_error_code_mismatch | 1 | 2 |
| long_document_score_dilution | 4 | 0 |
| bm25_field_weighting_or_index_structure | 2 | 0 |

Common train/dev primary root causes:

```text
entity_version_error_code_mismatch
query_expression_gap
title_heading_mismatch
```

## Diagnostic Signals

High-signal dimension counts across all 148 audit cases:

| Dimension | High-signal count |
| --- | ---: |
| title_heading_mismatch | 137 |
| bm25_field_weighting_or_index_structure | 121 |
| entity_version_error_code_mismatch | 80 |
| query_expression_gap | 65 |
| long_document_score_dilution | 37 |
| section_boundary_or_span_locality | 10 |

These are diagnostic signals, not runtime features.

Gold document diagnostic rank buckets:

| Bucket | Count |
| --- | ---: |
| not_found_top50 | 110 |
| rank_21_to_50 | 24 |
| rank_11_to_20 | 14 |

Question route counts:

| Route | Count |
| --- | ---: |
| other | 57 |
| error_or_log | 40 |
| install_upgrade_config | 32 |
| how_to_or_lookup | 12 |
| security_bulletin_vulnerability_detail | 4 |
| limitation_or_restriction | 3 |

Confidence bands:

| Band | Count |
| --- | ---: |
| high_multi_signal | 103 |
| medium_two_signal | 39 |
| low_single_signal | 6 |

## Interpretation

The largest primary buckets are `title_heading_mismatch` and
`query_expression_gap`. This means the next retrieval/index redesign should
start from title/heading and query-expression alignment, not from answer
composition or refusal tuning.

The strongest secondary diagnostic signals are also important:

```text
bm25_field_weighting_or_index_structure: 121 / 148 high-signal cases
entity_version_error_code_mismatch: 80 / 148 high-signal cases
```

This suggests Stage113 should freeze a protocol for retrieval/index redesign
that can test title/heading weighting, section-level indexing, and
entity/version/error-code handling under train/dev-only rules.

This interpretation does not select a candidate, does not tune thresholds, and
does not justify runtime defaultization.

## Public-Safe Output

Public case samples contain only the frozen Stage111 fields:

```text
sample_id
split
retrieval_context_miss_root_cause_bucket
question_route
gold_doc_rank_bucket
query_expression_gap_bucket
title_heading_overlap_bucket
section_locality_bucket
document_length_bucket
entity_version_error_code_bucket
index_structure_signal_bucket
confidence_band
```

The report does not output raw question text, raw answer text, document text,
document title, retrieved document IDs, cited document IDs, source document IDs,
answer document IDs, matched token strings, or query term strings.

## Decision

```text
status: primeqa_hybrid_retrieval_context_miss_root_cause_audit_completed
analysis_id: primeqa_hybrid_retrieval_context_miss_root_cause_audit_v1
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Recommended next stage:

```text
Stage113: after user confirmation, freeze a train/dev-only retrieval/index redesign protocol based on the Stage112 aggregate root-cause audit; do not select from dev-only observations, do not open the final test gate, keep runtime defaults unchanged, and add no fallback strategies.
```

## Guard Checks

All `19 / 19` guard checks passed.

Key checks:

```text
source_stage111_is_expected: passed
source_stage111_protocol_id_matches: passed
stage111_protocol_is_frozen: passed
stage111_recommends_stage112_audit: passed
user_confirmed_stage112_audit: passed
stage112_contract_reports_train_dev_only: passed
stage112_contract_forbids_selection_and_defaultization: passed
gold_doc_ids_are_offline_labeling_only: passed
loaded_splits_are_train_dev_only: passed
test_split_not_loaded: passed
audit_case_counts_match_stage102_retrieval_context_miss: passed
audit_dimensions_match_stage111_protocol: passed
diagnostic_depth_matches_stage111_contract: passed
public_case_fields_match_stage111_contract: passed
public_outputs_have_no_forbidden_keys: passed
final_test_metrics_not_run: passed
dev_selection_and_threshold_tuning_not_run: passed
runtime_defaults_remain_unchanged: passed
fallback_strategies_remain_disabled: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_audit_case_counts_by_split.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_primary_root_cause_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_dimension_high_signal_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_gold_rank_bucket_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_question_route_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_guard_check_status.svg
```

## Validation

Targeted validation before the real run:

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_context_miss_root_cause_audit.py scripts\run_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py tests\test_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py
python -m pytest tests\test_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py -q
```

Result:

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage112 run: passed
guard checks: 19 / 19 passed
```

Full validation is recorded in `docs\learning_journal.md`.

## Next Step

Stage113 should happen only after user confirmation. It should freeze a
train/dev-only retrieval/index redesign protocol based on Stage112 aggregate
findings. It must keep test locked, avoid dev-only selection, avoid threshold
tuning, keep runtime defaults unchanged, and add no fallback strategies.
