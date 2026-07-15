# PrimeQA Hybrid Retrieval-Context-Miss Audit Protocol

This document records Stage 111.

## Scope

Stage 111 freezes the train/dev-only retrieval-context-miss root-cause audit
protocol selected by the user as route A after the Stage110 stop decision.

This is a protocol freeze only. It reads only saved public-safe Stage102,
Stage107, and Stage110 reports. It does not load train/dev/test split files,
does not load corpus documents, does not run retrieval or answer metrics, does
not run final metrics, does not select from dev-only observations, does not add
fallback strategies, and does not change runtime defaults.

The protocol freeze was explicitly confirmed for this run:

```text
route_id: retrieval_context_miss_root_cause_audit_protocol
confirmed: true
confirmation_note: user selected route A and confirmed Stage111 retrieval-context-miss root-cause audit protocol on 2026-07-16; protocol freeze only; test locked; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol.py ^
  --user-confirmed-protocol ^
  --confirmation-note "user selected route A and confirmed Stage111 retrieval-context-miss root-cause audit protocol on 2026-07-16; protocol freeze only; test locked; runtime defaults unchanged; no fallback strategies"
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111.json
```

The run completed successfully in `0.002s` according to the Stage111 timing
block.

## Evidence Used

Stage102 showed that retrieval context misses remain a material failure bucket:

```text
train_retrieval_context_miss_count: 125
dev_retrieval_context_miss_count: 23
train_answerability_false_answer_count: 180
dev_answerability_false_answer_count: 41
train_gold_span_beats_selected_count: 174
dev_gold_span_beats_selected_count: 41
train_verified_average_token_f1: 0.2017
dev_verified_average_token_f1: 0.204
```

Stage107 confirmed that the dev failure set has answerable rows whose gold
context is absent from the verified retrieval context:

```text
dev_failure_count: 117
dev_failure_rate: 0.9669
answerable_failure_rate: 1.0
answerable_gold_context_absent_count: 23
answerable_gold_context_absent_rate: 0.3026
answerable_gold_context_present_count: 53
context_present_failure_count: 53
```

Stage110 stopped the failure-pattern redesign family and required a
user-confirmed next direction:

```text
decision_status: primeqa_hybrid_failure_pattern_redesign_family_stopped
stopped_family_id: failure_pattern_redesign_candidate_family
requires_user_confirmation_before_next_protocol: true
stage109_selectable_config_count: 0
stage109_config_count: 7
```

## Frozen Audit Dimensions

| Dimension | Priority | Purpose |
| --- | ---: | --- |
| query_expression_gap | 0.95 | Check whether missed questions use wording that poorly overlaps the gold section text under lexical BM25 retrieval. |
| title_heading_mismatch | 0.85 | Check whether document titles or section headings fail to match title/body vocabulary from the user question. |
| section_boundary_or_span_locality | 0.80 | Check whether answer spans are isolated inside section boundaries or far from section-level lexical anchors. |
| long_document_score_dilution | 0.75 | Check whether long documents or long sections dilute BM25 scores for relevant gold context. |
| entity_version_error_code_mismatch | 0.70 | Check whether product names, versions, APARs, error codes, or CVE-like tokens are missing or mismatched. |
| bm25_field_weighting_or_index_structure | 0.65 | Check whether the current BM25 field or index structure underweights titles, headings, or local sections. |

All audit dimensions are offline audit dimensions only. They are not runtime
features.

## Stage112 Run Contract

Stage112 may run only after user confirmation. If confirmed, it may load:

```text
Stage111 frozen protocol
Stage68 train split
Stage68 dev split
PrimeQA training/dev corpus sections
Stage102 public-safe report for baseline bucket targets
```

Stage112 must not load:

```text
test split
final-test labels
runtime oracle document identifiers
raw question, answer, or document text in public outputs
```

Additional frozen rules:

```text
audit_population: answerable train/dev rows classified as retrieval_context_miss under the Stage102 verified BM25 top10 baseline
retrieval_depth_for_diagnostic_only: 50
stage112_may_use_gold_doc_id_for_offline_labeling: true
gold_doc_id_allowed_as_runtime_feature: false
reported_splits: train, dev
selection_or_threshold_tuning_allowed: false
candidate_defaultization_allowed: false
final_test_metrics_allowed: false
```

## Public-Safe Output Contract

Stage112 public outputs should be aggregate-only by default. Case-level outputs,
if written, may contain only bucketed diagnostic fields such as:

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

Forbidden public fields include raw question text, raw answer text, gold answer
text, document text, document titles, raw query terms, retrieved document IDs,
cited document IDs, and source document IDs.

## Decision

```text
status: primeqa_hybrid_retrieval_context_miss_audit_protocol_frozen
protocol_id: primeqa_hybrid_retrieval_context_miss_audit_protocol_v1
route_id: retrieval_context_miss_root_cause_audit_protocol
recommended_next_direction: run_retrieval_context_miss_root_cause_audit_train_dev
can_continue_train_dev_development: true
can_run_train_dev_audit_after_user_confirmation: true
requires_user_confirmation_before_train_dev_audit: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Stage111 therefore provides no justification for runtime defaultization or
final-test gate opening.

## Guard Checks

All `24 / 24` guard checks passed.

Key checks:

```text
source_stage102_is_expected: passed
source_stage102_analysis_id_matches: passed
stage102_has_train_dev_retrieval_context_misses: passed
source_stage107_is_expected: passed
source_stage107_protocol_id_matches: passed
stage107_confirms_dev_gold_context_absent: passed
source_stage110_is_expected: passed
stage110_stopped_failure_pattern_redesign_family: passed
stage110_requires_user_confirmed_next_direction: passed
user_confirmed_stage111_protocol: passed
audit_dimensions_cover_offered_route_a: passed
stage112_contract_is_train_dev_only: passed
gold_doc_ids_are_offline_audit_only: passed
stage112_selection_and_defaultization_forbidden: passed
stage111_does_not_load_splits_or_corpus: passed
stage111_does_not_run_retrieval_or_answer_metrics: passed
stage111_final_test_metrics_not_run: passed
public_outputs_have_no_forbidden_keys: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_retrieval_context_miss_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_audit_dimension_priorities.svg
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_stage112_data_access_contract.svg
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_protocol_decision_flags.svg
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_guard_check_status.svg
```

## Validation

Targeted validation already passed:

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_context_miss_audit_protocol.py scripts\freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol.py tests\test_primeqa_hybrid_retrieval_context_miss_audit_protocol.py
python -m pytest tests\test_primeqa_hybrid_retrieval_context_miss_audit_protocol.py -q
python scripts\freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol.py --user-confirmed-protocol ...
```

Result:

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage111 run: passed
guard checks: 24 / 24 passed
```

Full validation is recorded in `docs\learning_journal.md`.

## Next Step

Stage112 should happen only after user confirmation. It should run the frozen
train/dev-only retrieval-context-miss root-cause audit, report train and dev
separately, keep test locked, avoid candidate selection or threshold tuning,
keep runtime defaults unchanged, and add no fallback strategies.
