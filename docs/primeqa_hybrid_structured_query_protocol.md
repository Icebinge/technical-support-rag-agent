# PrimeQA Hybrid Structured Query Protocol

This document records Stage 88.

## Scope

Stage 88 freezes the train/dev-only protocol for the next Stage84 second-wave
candidate after Stage87 stopped lexical cluster diversity:

```text
structured_query_keyphrase_compaction_design
```

This stage reads the public-safe Stage84 design report and Stage87 stop decision
report. It does not run retrieval metrics, does not load the frozen test split,
does not run final metrics, does not use source `DOC_IDS` as runtime retrieval
evidence, and does not change runtime defaults.

The candidate was explicitly confirmed for this run:

```text
confirmed: true
confirmed_candidate_id: structured_query_keyphrase_compaction_design
confirmation_note: user confirmed Stage88 structured query protocol freeze in current turn
```

## Command

```text
python scripts\freeze_primeqa_hybrid_structured_query_protocol.py ^
  --user-confirmed-candidate ^
  --confirmed-candidate-id structured_query_keyphrase_compaction_design ^
  --confirmation-note "user confirmed Stage88 structured query protocol freeze in current turn" ^
  --output artifacts\primeqa_hybrid_structured_query_protocol_stage88.json ^
  --visualization-dir artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals
```

The final recorded run completed in `0.000s`.

## Frozen Protocol

```text
protocol_id: structured_query_keyphrase_compaction_train_dev_v1
candidate_id: structured_query_keyphrase_compaction_design
protocol_status: frozen_requires_user_confirmation_before_metric_run
source_stages:
  Stage 84
  Stage 87
```

Baseline retriever:

```text
config_id: full_document_bm25_baseline
bm25_k1: 1.5
bm25_b: 0.75
candidate_depth: 50
primary_top_k: 10
```

Candidate config grid:

| Config | Query view | Max unique terms | Min unique terms |
| --- | --- | ---: | ---: |
| sqkc_action_error_product_v1 | action_error_product_version_terms | 18 | 4 |
| sqkc_title_guarded_action_error_v1 | title_guarded_action_error_product_terms | 16 | 4 |
| sqkc_error_first_compact_v1 | error_identifier_first_terms | 14 | 3 |
| sqkc_noun_phrase_compact_v1 | deterministic_noun_phrase_like_terms | 20 | 4 |

Compaction contract:

```text
query_terms_source: runtime question title and body text
normalization:
  casefold, split punctuation, preserve code-like spans, remove configured
  stopwords, and de-duplicate by first occurrence
ordering:
  error_code_or_log_identifier
  product_component_or_feature
  version_or_platform
  action_intent
  title_guard_terms
  deterministic_noun_phrase_like_terms
query_text_written_to_report: false
candidate_depth_unchanged: true
```

No alternate retriever or replacement behavior is included outside the frozen
config grid.

## Feature Contract

Runtime-allowed feature groups:

```text
query_structure_features:
  query_token_count
  query_unique_token_count
  title_token_count
  body_token_count

deterministic_token_class_features:
  is_error_code_or_log_identifier
  is_product_component_or_feature
  is_version_or_platform
  is_action_intent
  is_quoted_or_code_like

token_position_features:
  first_occurrence_index
  appears_in_title
  appears_in_body
  bucket_order_index

token_filter_features:
  token_length
  token_frequency_within_query
  stopword_list_membership
```

Prohibited runtime features:

```text
source_DOC_IDS
answer document IDs
gold document rank
gold labels
frozen test split membership
raw document text
raw document title
```

Public-safe changed-case fields:

```text
sample_id
split
baseline_rank
challenger_rank
config_id
query_view_id
query_token_count
compacted_query_token_count
token_bucket_counts
```

## Selection Rule

```text
selection_split: train
validation_split: dev
rule: Select the candidate config on train by hit@10, then hit@5, hit@1,
      MRR@10, fewer top10 regressions, fewer rank-down within top10, lower
      average compacted token count, then config_id. Dev is validation only.
dev_selection_forbidden: true
test_selection_forbidden: true
```

## Guard Checks

All `23 / 23` guard checks passed:

```text
source_stage84_report_is_stage84: passed
source_stage87_report_is_stage87: passed
user_confirmed_structured_query_protocol: passed
stage87_stopped_lcdr_route: passed
confirmed_candidate_matches_stage87_next_candidate: passed
stage87_requires_confirmation_before_next_protocol: passed
stage87_final_test_metrics_locked: passed
stage87_forbids_test_tuning: passed
stage87_runtime_default_unchanged: passed
stage84_final_test_metrics_locked: passed
stage84_forbids_test_tuning: passed
stage84_runtime_default_unchanged: passed
stage84_candidate_is_recommended_for_protocol_design: passed
stage84_candidate_target_contract_requires_train_selected_dev_hit10: passed
stage84_candidate_contract_forbids_dev_selection: passed
protocol_id_is_fixed: passed
candidate_config_grid_is_predeclared: passed
query_feature_contract_is_runtime_only: passed
source_doc_ids_forbidden_in_runtime_features: passed
report_fields_are_public_safe: passed
stage88_freezes_protocol_without_metrics: passed
stage88_final_test_metrics_not_run: passed
stage88_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals\stage88_structured_query_config_token_limits.svg
artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals\stage88_structured_query_feature_group_counts.svg
artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals\stage88_structured_query_protocol_decision_flags.svg
artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals\stage88_structured_query_guard_check_status.svg
```

Stage88 JSON SHA256:

```text
12A9045D7B65208EFF840E931E296FE56ED4D02A5852DA3C299011BA14767A7C
```

Visualization SHA256:

```text
stage88_structured_query_config_token_limits.svg: 2E3043E29343BEB0FBA12DE53131D8D5A4803AB1771870F08023BDA052F20326
stage88_structured_query_feature_group_counts.svg: 90D455A2D32F7DD2647C8532E0E9F2CBADE6D612FC171C5350924EC3B0F1B2C1
stage88_structured_query_guard_check_status.svg: 11F2670D9C3A9AD8058F91F383E6D0F13BBEAA072B887363FFB165A7AE0D3CA7
stage88_structured_query_protocol_decision_flags.svg: 67FD7B7B6651A8209912306BD4030068F93EDDE389B38EF97DF46E8A2AF5B14B
```

## Validation

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_structured_query_protocol.py scripts\freeze_primeqa_hybrid_structured_query_protocol.py tests\test_primeqa_hybrid_structured_query_protocol.py: passed
pytest -q tests\test_primeqa_hybrid_structured_query_protocol.py: 3 passed
Select-String raw question / answer / document / snippet field patterns over Stage88 JSON: no matches
git check-ignore Stage88 JSON and SVG artifacts: ignored by .gitignore
ruff check .: passed
pytest -q: 229 passed
git diff --check: passed
```

## Decision

```text
status: primeqa_hybrid_structured_query_protocol_frozen
protocol_id: structured_query_keyphrase_compaction_train_dev_v1
candidate_id: structured_query_keyphrase_compaction_design
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_run_train_dev_metrics_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage89 should run the frozen train/dev-only structured query keyphrase
compaction comparison after user confirmation. The frozen test split remains
locked, final metrics must not be run, source `DOC_IDS` must not be used as
runtime retrieval evidence, and runtime defaults remain unchanged.
