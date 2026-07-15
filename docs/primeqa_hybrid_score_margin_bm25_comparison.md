# PrimeQA Hybrid Score-Margin BM25 Comparison

This document records Stage 95.

## Scope

Stage 95 runs the frozen train/dev-only comparison for
`score_margin_bm25_normalization_gate_train_dev_v1` after the user confirmed
the Stage94 protocol.

This stage loads only the frozen Stage68 train/dev splits and the PrimeQA
training/dev document file. It does not load the frozen test split, does not
run final metrics, does not use source `DOC_IDS` as runtime retrieval evidence,
does not choose runtime rules from dev-only observations, does not write raw
question, answer, document, query-term, or matched-token text, and does not
change runtime defaults.

## Command

```text
python scripts\run_primeqa_hybrid_score_margin_bm25_comparison.py ^
  --user-confirmed-protocol ^
  --confirmed-protocol-id score_margin_bm25_normalization_gate_train_dev_v1 ^
  --confirmation-note "user confirmed Stage95 train/dev metric run in current turn" ^
  --output artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95.json ^
  --visualization-dir artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals
```

Console output was also captured in:

```text
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95.console.txt
```

The run completed in `87.7s` from the shell timing.

## Loaded Data

```text
document_count: 28482
average_document_token_count: 475.7771
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
test_split_loaded: false
```

## Frozen Configs

```text
smbn_rank11_20_long_doc_b095_margin_v1
smbn_rank21_50_long_doc_b095_high_confidence_v1
smbn_rank11_20_short_doc_b055_margin_v1
smbn_rank11_50_dual_length_band_margin_v1
```

Train selection rule:

```text
Select the score-margin BM25 config on train only by hit@10, then fewer rank
11-50 near misses, then fewer top10 regressions, then hit@5, hit@1, MRR@10,
lower top10 promotion budget, then config_id; dev is validation only.
```

## Train Metrics

Baseline:

```text
full_document_bm25_baseline:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  MRR@50: 0.5061
  not_found@50: 93
  rank_11_to_50_count: 32
```

All frozen configs had identical train hit@10 and rank 11-50 count to the
baseline:

```text
smbn_rank11_20_long_doc_b095_margin_v1: hit@10 0.6622, rank_11_to_50_count 32, gate actions 1
smbn_rank21_50_long_doc_b095_high_confidence_v1: hit@10 0.6622, rank_11_to_50_count 32, gate actions 0
smbn_rank11_20_short_doc_b055_margin_v1: hit@10 0.6622, rank_11_to_50_count 32, gate actions 4
smbn_rank11_50_dual_length_band_margin_v1: hit@10 0.6622, rank_11_to_50_count 32, gate actions 4
```

Train-selected config:

```text
smbn_rank11_20_long_doc_b095_margin_v1:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  MRR@50: 0.5061
  not_found@50: 93
  rank_11_to_50_count: 32
  score_margin_gate_promotion_count: 1
  length_band_gate_count: 1
```

Train-selected comparison:

```text
hit@10_delta: +0.0000
top10_improvement_count: 0
top10_regression_count: 0
top10_net_improvement_count: 0
search_depth_improvement_count: 0
search_depth_regression_count: 0
search_depth_net_improvement_count: 0
not_found_count_at_50_delta: 0
rank_11_to_50_count_delta: 0
```

## Dev Metrics

Baseline:

```text
full_document_bm25_baseline:
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  MRR@10: 0.5331
  MRR@50: 0.5365
  not_found@50: 17
  rank_11_to_50_count: 6
```

All frozen configs had identical dev hit@10 and rank 11-50 count to the
baseline:

```text
smbn_rank11_20_long_doc_b095_margin_v1: hit@10 0.6974, rank_11_to_50_count 6, gate actions 0
smbn_rank21_50_long_doc_b095_high_confidence_v1: hit@10 0.6974, rank_11_to_50_count 6, gate actions 0
smbn_rank11_20_short_doc_b055_margin_v1: hit@10 0.6974, rank_11_to_50_count 6, gate actions 2
smbn_rank11_50_dual_length_band_margin_v1: hit@10 0.6974, rank_11_to_50_count 6, gate actions 1
```

Train-selected dev comparison:

```text
selected_config_id: smbn_rank11_20_long_doc_b095_margin_v1
hit@10_delta: +0.0000
rank_11_to_50_count_delta: 0
top10_improvement_count: 0
top10_regression_count: 0
top10_net_improvement_count: 0
search_depth_improvement_count: 0
search_depth_regression_count: 0
search_depth_net_improvement_count: 0
not_found_count_at_50_delta: 0
score_margin_gate_promotion_count: 0
length_band_gate_count: 0
```

## Decision

```text
status: primeqa_hybrid_score_margin_bm25_comparison_completed
selected_config_id: smbn_rank11_20_long_doc_b095_margin_v1
selected_dev_hit10_delta: 0.0000
selected_dev_rank_11_to_50_count_delta: 0
selected_dev_top10_improvements: 0
selected_dev_top10_regressions: 0
selected_dev_score_margin_gate_promotion_count: 0
primary_contract_passed: false
secondary_contract_passed: false
guard_contract_passed: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

The route does not advance. The frozen gate produced a few train/dev promotion
actions across non-selected configs, but the train-selected config produced no
dev actions and did not improve dev hit@10 or reduce dev rank 11-50 near
misses. This fails the Stage84/Stage94 primary and secondary metric contracts.

## Guard Checks

All `23 / 23` guard checks passed:

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
source_stage94_report_is_stage94: passed
stage94_protocol_id_matches: passed
stage94_candidate_id_matches: passed
user_confirmed_frozen_protocol: passed
confirmed_protocol_id_matches: passed
stage94_allows_train_dev_metrics_after_confirmation: passed
stage94_final_test_metrics_locked: passed
stage94_forbids_test_tuning: passed
stage94_default_runtime_policy_unchanged: passed
candidate_config_grid_matches_frozen_protocol: passed
stage94_train_selection_rule_forbids_dev_selection: passed
stage94_train_selection_rule_forbids_stage82_dev_selection: passed
historical_stage82_signal_is_motivation_only: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
source_doc_ids_not_used_as_runtime_evidence: passed
answer_doc_ids_not_used_as_runtime_evidence: passed
changed_case_fields_public_safe: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_train_hit_at_10.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_hit_at_10.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_rank_11_to_50_delta.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_top10_changes.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_gate_actions.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_guard_check_status.svg
```

Stage95 JSON SHA256:

```text
B712AFC1197DECD5EB50BB3471A58F0D0902AE35641B3C720AED855BC0CB3237
```

Visualization SHA256:

```text
stage95_score_margin_bm25_dev_delta_hit_at_10.svg: 78522CED909FF4E61C3A44847E204D061F8F218C6A3E586168C859FBBA4F43AF
stage95_score_margin_bm25_dev_gate_actions.svg: D728CA8A8C70D571CD8BE14198B9347D4D1BCCEC1C1B18259116BE096566CC9F
stage95_score_margin_bm25_dev_hit_at_10.svg: F98C235BF5040A7885EE9BDC276745E88064AC20D0E0670215B9B358583C68FD
stage95_score_margin_bm25_dev_rank_11_to_50_delta.svg: 62EB74B362A245EC44E151F95602AA470CA1CB9A048BDC8AEEA0E44C51B471E4
stage95_score_margin_bm25_dev_top10_changes.svg: 6FF10AD6FD8A0C30756DF8164F61AF450D7F84F65BED5FA7CD87ABF1F0C59A56
stage95_score_margin_bm25_guard_check_status.svg: A6010A3B3B3FA954ECB94EC561AE583000C6CF4AD131B24666CCF4A86DEF5DD3
stage95_score_margin_bm25_train_hit_at_10.svg: 646BEF677C7E8C3D023CC876DEC76A84E27CB43BE625D45A56C31AC506263F0C
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_score_margin_bm25_comparison.py scripts\run_primeqa_hybrid_score_margin_bm25_comparison.py tests\test_primeqa_hybrid_score_margin_bm25_comparison.py: passed
pytest -q tests\test_primeqa_hybrid_score_margin_bm25_comparison.py: 3 passed
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage95 JSON: no matches
git check-ignore Stage95 JSON, console, and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 250 passed
git diff --check: passed
```

## Next Step

Stage96 should stop score-margin BM25 normalization as a retrieval-recall route
unless a new train/dev-only protocol is explicitly confirmed. The frozen test
split remains locked, final metrics must not be run, source `DOC_IDS` must not
be used as runtime retrieval evidence, dev-only observations must not select
runtime rules, and runtime defaults remain unchanged.
