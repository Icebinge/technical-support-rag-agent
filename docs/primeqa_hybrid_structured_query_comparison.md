# PrimeQA Hybrid Structured Query Comparison

This document records Stage 89.

## Scope

Stage 89 runs the user-confirmed train/dev-only comparison for the frozen
Stage88 protocol:

```text
structured_query_keyphrase_compaction_train_dev_v1
```

This stage selects across the frozen structured-query candidate grid on train,
validates the selected config on dev, keeps the frozen test split locked, does
not run final metrics, does not use source `DOC_IDS` as runtime retrieval
evidence, does not write raw question, compacted query, answer, or document text
to the report, and does not change runtime defaults.

The protocol metric run was explicitly confirmed for this run:

```text
confirmed: true
confirmed_protocol_id: structured_query_keyphrase_compaction_train_dev_v1
confirmation_note: user confirmed Stage89 train/dev metric run in current turn
```

## Command

```text
python scripts\run_primeqa_hybrid_structured_query_comparison.py ^
  --user-confirmed-protocol ^
  --confirmed-protocol-id structured_query_keyphrase_compaction_train_dev_v1 ^
  --confirmation-note "user confirmed Stage89 train/dev metric run in current turn" ^
  --output artifacts\primeqa_hybrid_structured_query_comparison_stage89.json ^
  --visualization-dir artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals
```

The run completed in `44.736s`.

Loaded data:

```text
document_count: 28482
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
test_split_loaded: false
```

## Frozen Grid

| Config | Query view | Max unique terms | Min unique terms |
| --- | --- | ---: | ---: |
| sqkc_action_error_product_v1 | action_error_product_version_terms | 18 | 4 |
| sqkc_title_guarded_action_error_v1 | title_guarded_action_error_product_terms | 16 | 4 |
| sqkc_error_first_compact_v1 | error_identifier_first_terms | 14 | 3 |
| sqkc_noun_phrase_compact_v1 | deterministic_noun_phrase_like_terms | 20 | 4 |

Selection rule:

```text
Select the structured query candidate config on train only by hit@10, then
hit@5, hit@1, MRR@10, fewer top10 regressions, fewer rank-down cases within
top10, lower average compacted query token count, then config_id. Dev is
validation only.
```

## Train Metrics

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | not found @50 | rank 11-50 | avg compacted terms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_document_bm25_baseline | 0.4243 | 0.6054 | 0.6622 | 0.5023 | 93 | 32 | 55.1703 |
| sqkc_action_error_product_v1 | 0.4135 | 0.5973 | 0.6351 | 0.4921 | 86 | 49 | 14.6811 |
| sqkc_title_guarded_action_error_v1 | 0.4162 | 0.5892 | 0.6595 | 0.4940 | 80 | 46 | 13.5514 |
| sqkc_error_first_compact_v1 | 0.3919 | 0.5541 | 0.6081 | 0.4653 | 98 | 47 | 12.2649 |
| sqkc_noun_phrase_compact_v1 | 0.4027 | 0.5919 | 0.6351 | 0.4858 | 92 | 43 | 15.4243 |

Train-selected config:

```text
sqkc_title_guarded_action_error_v1
```

Train comparison for the selected config:

```text
hit@10_delta: -0.0027
top10_improvement_count: 14
top10_regression_count: 15
top10_net_improvement_count: -1
rank_up_within_top10_count: 26
rank_down_within_top10_count: 38
not_found_count_at_50_delta: -13
rank_11_to_50_count_delta: +14
average_compacted_query_token_count_delta: -41.6189
```

## Dev Metrics

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | not found @50 | rank 11-50 | avg compacted terms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_document_bm25_baseline | 0.4342 | 0.6579 | 0.6974 | 0.5331 | 17 | 6 | 49.6316 |
| sqkc_action_error_product_v1 | 0.4211 | 0.6184 | 0.6316 | 0.4991 | 17 | 11 | 14.8026 |
| sqkc_title_guarded_action_error_v1 | 0.4342 | 0.6316 | 0.6447 | 0.5124 | 16 | 11 | 13.5395 |
| sqkc_error_first_compact_v1 | 0.4342 | 0.6053 | 0.6579 | 0.5039 | 16 | 10 | 12.2632 |
| sqkc_noun_phrase_compact_v1 | 0.4474 | 0.6447 | 0.6711 | 0.5314 | 18 | 7 | 15.3947 |

Dev comparison for the train-selected config:

```text
hit@10_delta: -0.0527
top10_improvement_count: 1
top10_regression_count: 5
top10_net_improvement_count: -4
rank_up_within_top10_count: 5
rank_down_within_top10_count: 5
not_found_count_at_50_delta: -1
rank_11_to_50_count_delta: +5
average_compacted_query_token_count_delta: -36.0921
```

Important boundary:

```text
The train-selected structured-query config fails the Stage88 target metric
contract. Dev hit@10 decreases from 0.6974 to 0.6447, and dev top10
regressions outnumber improvements.
```

The best dev hit@10 among structured-query candidates was
`sqkc_noun_phrase_compact_v1` at `0.6711`, but it was not the train-selected
config and still remained below the BM25 baseline. It must not be selected by
dev-only performance.

## Guard Checks

All `21 / 21` guard checks passed:

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
source_stage88_report_is_stage88: passed
stage88_protocol_id_matches: passed
stage88_candidate_id_matches: passed
user_confirmed_frozen_protocol: passed
confirmed_protocol_id_matches: passed
stage88_allows_train_dev_metrics_after_confirmation: passed
stage88_final_test_metrics_locked: passed
stage88_forbids_test_tuning: passed
stage88_default_runtime_policy_unchanged: passed
candidate_config_grid_matches_frozen_protocol: passed
source_stage75_report_is_stage75: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
source_doc_ids_not_used_as_runtime_evidence: passed
changed_case_fields_public_safe: passed
raw_or_compacted_query_text_not_written: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_train_hit_at_10.svg
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_dev_hit_at_10.svg
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_dev_top10_changes.svg
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_average_compacted_terms.svg
```

Stage89 JSON SHA256:

```text
592EF83368BB8DB268114677E1BF9FDCAC1C4E37E8BC10099917CA7956BDF4A4
```

Visualization SHA256:

```text
stage89_structured_query_average_compacted_terms.svg: 3BA48FD64FCCA65D2D30CCC0600726A108392B5A1D417F97C9590A5779D50CDF
stage89_structured_query_dev_delta_hit_at_10.svg: 2520DBEC41AE2263845D67DB1C4AACB3ADD29686655AA9EA9D080FE8616C6D94
stage89_structured_query_dev_hit_at_10.svg: 1F27A1005A1074C5F5FE9C7F5E8846FF591E4D0712BF137268E3B68468E38327
stage89_structured_query_dev_top10_changes.svg: 92150CEC7848DBDB319AD0C1A39D4DA98EAB5F44309F4310CE3D7BFA08A50A08
stage89_structured_query_train_hit_at_10.svg: 62807A6A5CD10C34C6E85E88A644DED5008B67413D3A27764E7F8ACBC2A11B8A
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_structured_query_comparison.py scripts\run_primeqa_hybrid_structured_query_comparison.py tests\test_primeqa_hybrid_structured_query_comparison.py: passed
pytest -q tests\test_primeqa_hybrid_structured_query_comparison.py: 3 passed
Select-String raw question / answer / document / snippet / query-term field patterns over Stage89 JSON: no matches
git check-ignore Stage89 JSON and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 232 passed
git diff --check: passed
```

## Decision

```text
status: primeqa_hybrid_structured_query_comparison_completed
selected_config_id: sqkc_title_guarded_action_error_v1
selected_query_view_id: title_guarded_action_error_product_terms
selected_dev_hit10_delta: -0.0527
selected_dev_top10_improvements: 1
selected_dev_top10_regressions: 5
selected_dev_rank_up_within_top10: 5
selected_dev_rank_down_within_top10: 5
selected_dev_not_found_at_search_depth_delta: -1
selected_dev_rank_11_to_50_count_delta: 5
selected_dev_average_compacted_query_token_count: 13.5395
primary_contract_passed: false
secondary_contract_passed: false
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage90 stopped structured query keyphrase compaction as a retrieval-recall
route. The next candidate in the Stage84 queue is
`section_signal_guarded_expansion_design`.

Stage91 confirmed and froze the train/dev-only protocol for
`section_signal_guarded_expansion_design` as
`section_signal_guarded_expansion_train_dev_v1`.

Stage92 ran the frozen train/dev-only section signal guarded expansion
comparison. The train-selected config was
`ssgx_section_top50_injection_guard_v1`, but it had dev hit@10 delta `0.0000`
and dev search-depth net improvement `0`, so the route does not advance.

Stage93 stopped section signal guarded expansion as a retrieval-recall route
and left runtime defaults unchanged. The remaining Stage84 queue is:

```text
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Stage94 confirmed and froze the train/dev-only protocol for
`score_margin_bm25_normalization_gate_design` as
`score_margin_bm25_normalization_gate_train_dev_v1`. Stage94 did not run
retrieval metrics and did not change runtime defaults.

Stage95 ran the frozen train/dev-only score-margin BM25 normalization gate
comparison after user confirmation. The train-selected config was
`smbn_rank11_20_long_doc_b095_margin_v1`, but it had dev hit@10 delta `0.0000`
and dev rank 11-50 count delta `0`, so the route does not advance.

The current next step is Stage96: stop score-margin BM25 normalization as a
retrieval-recall route unless a new train/dev-only protocol is explicitly
confirmed. The frozen test split remains locked, final metrics must not be run,
source `DOC_IDS` must not be used as runtime retrieval evidence, dev-only
observations must not select runtime rules, and runtime defaults remain
unchanged.
