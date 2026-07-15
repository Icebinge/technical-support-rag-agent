# PrimeQA Hybrid Section Signal Protocol

This document records Stage 91.

## Scope

Stage 91 confirms and freezes the train/dev-only protocol for
`section_signal_guarded_expansion_design` after Stage90 stopped the structured
query compaction route.

This is a protocol-freeze checkpoint. It reads the public-safe Stage84 and
Stage90 reports, does not run retrieval metrics, does not load train/dev/test
split files, does not run final metrics, does not use source `DOC_IDS` as
runtime retrieval evidence, and does not change runtime defaults.

The candidate confirmation was explicit for this run:

```text
confirmed_candidate_id: section_signal_guarded_expansion_design
confirmed: true
confirmation_note: user confirmed Stage91 section-signal protocol freeze in current turn
```

## Command

```text
python scripts\freeze_primeqa_hybrid_section_signal_protocol.py ^
  --user-confirmed-candidate ^
  --confirmed-candidate-id section_signal_guarded_expansion_design ^
  --confirmation-note "user confirmed Stage91 section-signal protocol freeze in current turn" ^
  --output artifacts\primeqa_hybrid_section_signal_protocol_stage91.json ^
  --visualization-dir artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals
```

The run completed in `0.001s`.

## Candidate

```text
candidate_id: section_signal_guarded_expansion_design
protocol_id: section_signal_guarded_expansion_train_dev_v1
category: section_signal_gate
risk_level: medium
implementation_readiness: 0.64
priority_score: 174
target_miss_count: 119
target_miss_count_by_split: train 102 / dev 17
```

Stage84 rationale:

```text
Stage79 section rollup regressed overall but rescued a small set of deep-rank
misses. A second-wave protocol should preserve the lesson as a gated section
signal design rather than repeating an ungated section replacement.
```

Stage84 target metric contract:

```text
primary: dev hit@10 must improve over BM25 baseline
secondary: search-depth improvements must exceed regressions
guard: section signal must not demote existing BM25 top10 hits by default
```

## Frozen Protocol

Baseline retriever:

```text
config_id: full_document_bm25_baseline
bm25_k1: 1.5
bm25_b: 0.75
candidate_depth: 50
primary_top_k: 10
```

Section signal source:

```text
source_retriever_id: section_bm25_max_section_rollup_stage79
section_score_scope: best section BM25 score per parent document
section_candidate_depth: 50
document_candidate_depth: 50
primary_top_k: 10
```

Predeclared candidate config grid:

```text
ssgx_shadow_no_top10_demotion_v1:
  promotion_mode: shadow_after_top10
  eligible_baseline_rank: 11-50
  section_rank_max: 50
  minimum_section_to_document_score_ratio: 1.10
  maximum_top10_promotions_per_query: 0
  protected_bm25_top_rank_count: 10

ssgx_rank11_20_margin_guard_v1:
  promotion_mode: single_rank10_promotion
  eligible_baseline_rank: 11-20
  section_rank_max: 30
  minimum_section_to_document_score_ratio: 1.20
  maximum_document_score_margin_to_rank10: 0.08
  maximum_top10_promotions_per_query: 1
  protected_bm25_top_rank_count: 5

ssgx_rank21_50_high_confidence_v1:
  promotion_mode: single_rank10_promotion
  eligible_baseline_rank: 21-50
  section_rank_max: 20
  minimum_section_to_document_score_ratio: 1.45
  maximum_document_score_margin_to_rank10: 0.05
  maximum_top10_promotions_per_query: 1
  protected_bm25_top_rank_count: 8

ssgx_section_top50_injection_guard_v1:
  promotion_mode: single_rank10_section_candidate_injection
  eligible_baseline_rank: 51+
  section_rank_max: 15
  minimum_section_to_document_score_ratio: 1.60
  maximum_top10_promotions_per_query: 1
  protected_bm25_top_rank_count: 8
```

Train selection rule:

```text
Select the candidate config on train by hit@10, then search-depth net
improvements, fewer top10 regressions, hit@5, hit@1, MRR@10, lower top10
demotion budget, then config_id. Dev is validation only.
```

Public-safe changed-case fields:

```text
sample_id
split
baseline_rank
challenger_rank
config_id
section_signal_bucket
baseline_rank_bucket
section_rank_bucket
score_ratio_bucket
score_margin_bucket
promotion_reason_code
top10_protection_action
```

## Decision

```text
status: primeqa_hybrid_section_signal_protocol_frozen
protocol_id: section_signal_guarded_expansion_train_dev_v1
candidate_id: section_signal_guarded_expansion_design
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_run_train_dev_metrics_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Guard Checks

All `26 / 26` guard checks passed:

```text
source_stage84_report_is_stage84: passed
source_stage90_report_is_stage90: passed
user_confirmed_section_signal_protocol: passed
stage90_stopped_structured_query_route: passed
confirmed_candidate_matches_stage90_next_candidate: passed
stage90_next_candidate_summary_matches: passed
stage90_requires_confirmation_before_next_protocol: passed
stage90_final_test_metrics_locked: passed
stage90_forbids_test_tuning: passed
stage90_runtime_default_unchanged: passed
stage84_final_test_metrics_locked: passed
stage84_forbids_test_tuning: passed
stage84_runtime_default_unchanged: passed
stage84_candidate_is_recommended_for_protocol_design: passed
stage84_candidate_contract_requires_dev_hit10_gain: passed
stage84_candidate_contract_requires_search_depth_net_positive: passed
stage84_candidate_guard_protects_bm25_top10: passed
protocol_id_is_fixed: passed
candidate_config_grid_is_predeclared: passed
section_signal_contract_uses_runtime_scores_only: passed
promotion_configs_are_guarded: passed
source_doc_ids_forbidden_in_runtime_features: passed
report_fields_are_public_safe: passed
stage91_freezes_protocol_without_metrics: passed
stage91_final_test_metrics_not_run: passed
stage91_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_config_promotion_budgets.svg
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_config_ratio_thresholds.svg
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_feature_group_counts.svg
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_protocol_decision_flags.svg
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_guard_check_status.svg
```

Stage91 JSON SHA256:

```text
41FC88ACD104042BBA8B9230477856D06359BC8A8F00DBC617564EAFA9707EA9
```

Visualization SHA256:

```text
stage91_section_signal_config_promotion_budgets.svg: 1BEDB21A2F540E26A1DDB84641AC4A0AC9C87BBC0861CEB46662C73110C1F8BC
stage91_section_signal_config_ratio_thresholds.svg: 9BB1471F36CFBBBA901D92B82714CC2F3BEB9F78C9844E80C8315F3027F9E1DC
stage91_section_signal_feature_group_counts.svg: 63EAB34291809D593A6B6AB7AB85A1073504E0C70F4B4C1E88264C55EC636D4F
stage91_section_signal_guard_check_status.svg: C5FB2D262B1330B937F2D6A1AF8973A0A996DDEFD8A10754C7E888CE86BD3234
stage91_section_signal_protocol_decision_flags.svg: 03AF6BFD684954FF00D2421E90E188AD57ED39A3AC95C62638AF3F37B55C6D5C
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_section_signal_protocol.py scripts\freeze_primeqa_hybrid_section_signal_protocol.py tests\test_primeqa_hybrid_section_signal_protocol.py: passed
pytest -q tests\test_primeqa_hybrid_section_signal_protocol.py: 3 passed
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage91 JSON: no matches
git check-ignore Stage91 JSON and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 238 passed
git diff --check: passed
```

## Next Step

Stage92 ran the frozen train/dev-only section signal guarded expansion
comparison after user confirmation. The train-selected config was
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

The current next step is Stage95: run the frozen train/dev-only score-margin
BM25 normalization gate comparison after user confirmation. The frozen test
split remains locked, final metrics must not be run, source `DOC_IDS` must not
be used as runtime retrieval evidence, Stage82 dev-only `b=0.95` observations
must not select the runtime rule, and runtime defaults remain unchanged.
