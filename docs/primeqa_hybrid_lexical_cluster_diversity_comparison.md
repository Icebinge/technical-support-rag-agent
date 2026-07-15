# PrimeQA Hybrid Lexical Cluster Diversity Comparison

This document records Stage 86.

## Scope

Stage 86 runs the user-confirmed train/dev-only comparison for the frozen
Stage85 protocol:

```text
lexical_cluster_diversity_rerank_train_dev_v1
```

This stage selects across the frozen lexical cluster diversity candidate grid on
train, validates the selected config on dev, keeps the frozen test split locked,
does not run final metrics, does not use source `DOC_IDS` as runtime retrieval
evidence, does not write raw question or document text to the report, and does
not change runtime defaults.

The protocol metric run was explicitly confirmed for this run:

```text
confirmed: true
confirmed_protocol_id: lexical_cluster_diversity_rerank_train_dev_v1
confirmation_note: user confirmed Stage86 train/dev metric run in current turn
```

## Command

```text
python scripts\run_primeqa_hybrid_lexical_cluster_diversity_comparison.py ^
  --user-confirmed-protocol ^
  --confirmed-protocol-id lexical_cluster_diversity_rerank_train_dev_v1 ^
  --confirmation-note "user confirmed Stage86 train/dev metric run in current turn" ^
  --output artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86.json ^
  --visualization-dir artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals
```

The run completed in `29.330s`.

## Frozen Grid

| Config | Duplicate penalty | Cluster key | Minimum title overlap terms | Minimum cluster size |
| --- | ---: | --- | ---: | ---: |
| lcdr_penalty_0_03_title_query_cluster | 0.03 | title_query_overlap_hash | 3 | 2 |
| lcdr_penalty_0_06_title_query_cluster | 0.06 | title_query_overlap_hash | 3 | 2 |
| lcdr_penalty_0_09_title_query_cluster | 0.09 | title_query_overlap_hash | 3 | 2 |
| lcdr_penalty_0_12_title_query_cluster | 0.12 | title_query_overlap_hash | 3 | 2 |

Selection rule:

```text
Select the lexical cluster diversity candidate config on train only by hit@10,
then hit@5, hit@1, MRR@10, fewer top10 regressions, fewer rank-down cases
within top10, then config_id. Dev is validation only.
```

## Train Metrics

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | not found @50 | rank 11-50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lcdr_penalty_0_06_title_query_cluster | 0.4243 | 0.6054 | 0.6676 | 0.5031 | 93 | 30 |
| lcdr_penalty_0_09_title_query_cluster | 0.4243 | 0.6054 | 0.6676 | 0.5030 | 93 | 30 |
| lcdr_penalty_0_12_title_query_cluster | 0.4243 | 0.6027 | 0.6676 | 0.5022 | 93 | 30 |
| lcdr_penalty_0_03_title_query_cluster | 0.4243 | 0.6054 | 0.6649 | 0.5027 | 93 | 31 |
| full_document_bm25_baseline | 0.4243 | 0.6054 | 0.6622 | 0.5023 | 93 | 32 |

Train-selected config:

```text
lcdr_penalty_0_06_title_query_cluster
```

Train comparison for the selected config:

```text
hit@10_delta: +0.0054
top10_improvement_count: 4
top10_regression_count: 2
rank_up_within_top10_count: 2
rank_down_within_top10_count: 0
rank_11_to_50_count_delta: -2
```

## Dev Metrics

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | not found @50 | rank 11-50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_document_bm25_baseline | 0.4342 | 0.6579 | 0.6974 | 0.5331 | 17 | 6 |
| lcdr_penalty_0_03_title_query_cluster | 0.4342 | 0.6579 | 0.6974 | 0.5331 | 17 | 6 |
| lcdr_penalty_0_06_title_query_cluster | 0.4342 | 0.6579 | 0.6974 | 0.5331 | 17 | 6 |
| lcdr_penalty_0_09_title_query_cluster | 0.4342 | 0.6579 | 0.6974 | 0.5331 | 17 | 6 |
| lcdr_penalty_0_12_title_query_cluster | 0.4342 | 0.6579 | 0.6974 | 0.5298 | 17 | 6 |

Dev comparison for the train-selected config:

```text
hit@10_delta: +0.0000
top10_improvement_count: 0
top10_regression_count: 0
rank_up_within_top10_count: 0
rank_down_within_top10_count: 0
rank_11_to_50_count_delta: 0
not_found_count_at_50_delta: 0
```

Important boundary:

```text
Stage86 improved train hit@10 but did not improve dev hit@10. The Stage84
target metric contract requires dev hit@10 to improve over BM25 baseline, so
this route does not advance to runtime and does not open the final test gate.
```

## Guard Checks

All `19 / 19` guard checks passed:

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
source_stage85_report_is_stage85: passed
stage85_protocol_id_matches: passed
stage85_candidate_id_matches: passed
user_confirmed_frozen_protocol: passed
confirmed_protocol_id_matches: passed
stage85_allows_train_dev_metrics_after_confirmation: passed
stage85_final_test_metrics_locked: passed
stage85_default_runtime_policy_unchanged: passed
candidate_config_grid_matches_frozen_protocol: passed
source_stage75_report_is_stage75: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
source_doc_ids_not_used_as_runtime_evidence: passed
changed_case_fields_public_safe: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_train_hit_at_10.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_dev_hit_at_10.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_dev_top10_changes.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_dev_answer_duplicate_buckets.svg
```

Stage86 JSON SHA256:

```text
2F00764F52AA1279A7F526E5ACF0735E6FA90C6D27E63026FE011630D7EB4195
```

Visualization SHA256:

```text
stage86_lcdr_dev_answer_duplicate_buckets.svg: C106089E411BC1977C3C181150D5929F4A71801533432AB10BD157B95FD932D0
stage86_lcdr_dev_delta_hit_at_10.svg: 6F5E95E5EDCDDB9B510E091B26B477D9B482F5EC03F167A2816F2623A9DDB0B5
stage86_lcdr_dev_hit_at_10.svg: 001C0AF383446EBD801A6B43A1AAB1994C049F276A9FD2B6401E59E30949D640
stage86_lcdr_dev_top10_changes.svg: 66682BE713A073DC0CDFD0E7322A298A0511BEE36907765CEABF0822AC568445
stage86_lcdr_train_hit_at_10.svg: 89C9C170D88643E6EE90DF1B65A6731143D964C4CC60C25E2D762A01B90F1E09
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_lexical_cluster_diversity_comparison.py scripts\run_primeqa_hybrid_lexical_cluster_diversity_comparison.py tests\test_primeqa_hybrid_lexical_cluster_diversity_comparison.py: passed
pytest -q tests\test_primeqa_hybrid_lexical_cluster_diversity_comparison.py: 3 passed
Select-String raw question / answer / document / snippet field patterns over Stage86 JSON: no matches
git check-ignore Stage86 JSON and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 223 passed
git diff --check: passed
```

## Decision

```text
status: primeqa_hybrid_lexical_cluster_diversity_comparison_completed
selected_config_id: lcdr_penalty_0_06_title_query_cluster
selected_dev_hit10_delta: 0.0
selected_dev_top10_improvements: 0
selected_dev_top10_regressions: 0
selected_dev_rank_up_within_top10: 0
selected_dev_rank_down_within_top10: 0
selected_dev_not_found_at_search_depth_delta: 0
selected_dev_rank_11_to_50_count_delta: 0
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage87 has stopped lexical cluster diversity as a retrieval-recall route. The
current route's defaultization is blocked, and the next candidate is
`structured_query_keyphrase_compaction_design`.

The current next step is Stage88: confirm and freeze the train/dev-only protocol
for `structured_query_keyphrase_compaction_design`. The frozen test split
remains locked, final metrics must not be run, source `DOC_IDS` must not be used
as runtime retrieval evidence, and runtime defaults remain unchanged.
