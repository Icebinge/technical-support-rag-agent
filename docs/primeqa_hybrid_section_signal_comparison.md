# PrimeQA Hybrid Section Signal Comparison

This document records Stage 92.

## Scope

Stage 92 runs the frozen train/dev-only comparison for
`section_signal_guarded_expansion_train_dev_v1` after the user confirmed the
Stage91 protocol.

This stage loads only the frozen Stage68 train/dev splits and the PrimeQA
training/dev document-section file. It does not load the frozen test split,
does not run final metrics, does not use source `DOC_IDS` as runtime retrieval
evidence, does not write raw question, answer, document, or section text, and
does not change runtime defaults.

## Command

```text
python scripts\run_primeqa_hybrid_section_signal_comparison.py ^
  --user-confirmed-protocol ^
  --confirmed-protocol-id section_signal_guarded_expansion_train_dev_v1 ^
  --confirmation-note "user confirmed Stage92 train/dev metric run in current turn" ^
  --output artifacts\primeqa_hybrid_section_signal_comparison_stage92.json ^
  --visualization-dir artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals
```

The run completed in `69.522s`.

## Loaded Data

```text
document_count: 28482
section_count: 216648
documents_with_sections: 28482
documents_without_sections: 0
average_sections_per_document: 7.6065
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
test_split_loaded: false
```

## Frozen Configs

```text
ssgx_shadow_no_top10_demotion_v1
ssgx_rank11_20_margin_guard_v1
ssgx_rank21_50_high_confidence_v1
ssgx_section_top50_injection_guard_v1
```

Train selection rule:

```text
Select the section signal config on train only by hit@10, then search-depth net
improvements, then fewer top10 regressions, then hit@5, hit@1, MRR@10, lower
top10 demotion budget, then config_id; dev is validation only.
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

Train-selected config:

```text
ssgx_section_top50_injection_guard_v1:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  MRR@50: 0.5063
  not_found@50: 92
  rank_11_to_50_count: 33
  section_signal_promotion_count: 92
  protected_top10_demotion_count: 92
```

Train-selected comparison:

```text
hit@10_delta: +0.0000
MRR@50_delta: +0.0002
top10_improvement_count: 1
top10_regression_count: 1
top10_net_improvement_count: 0
search_depth_improvement_count: 1
search_depth_regression_count: 0
search_depth_net_improvement_count: +1
not_found_count_at_50_delta: -1
rank_11_to_50_count_delta: +1
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

All frozen configs had identical dev hit@10 to the baseline:

```text
ssgx_shadow_no_top10_demotion_v1: hit@10 0.6974, delta +0.0000
ssgx_rank11_20_margin_guard_v1: hit@10 0.6974, delta +0.0000
ssgx_rank21_50_high_confidence_v1: hit@10 0.6974, delta +0.0000
ssgx_section_top50_injection_guard_v1: hit@10 0.6974, delta +0.0000
```

Train-selected dev comparison:

```text
selected_config_id: ssgx_section_top50_injection_guard_v1
hit@10_delta: +0.0000
top10_improvement_count: 0
top10_regression_count: 0
top10_net_improvement_count: 0
search_depth_improvement_count: 0
search_depth_regression_count: 0
search_depth_net_improvement_count: 0
not_found_count_at_50_delta: 0
rank_11_to_50_count_delta: 0
section_signal_promotion_count: 22
protected_top10_demotion_count: 22
```

## Decision

```text
status: primeqa_hybrid_section_signal_comparison_completed
selected_config_id: ssgx_section_top50_injection_guard_v1
selected_dev_hit10_delta: 0.0000
selected_dev_search_depth_net_improvement_count: 0
selected_dev_top10_improvements: 0
selected_dev_top10_regressions: 0
selected_dev_not_found_at_search_depth_delta: 0
selected_dev_section_signal_promotion_count: 22
selected_dev_protected_top10_demotion_count: 22
primary_contract_passed: false
secondary_contract_passed: false
guard_contract_passed: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

The route does not advance. The train-selected config made 22 dev promotion
actions but did not improve dev hit@10, search-depth recall, not-found@50, or
rank 11-50 count. Because Stage91 required dev hit@10 improvement and positive
search-depth net improvement, the primary and secondary contracts both failed.

## Guard Checks

All `20 / 20` guard checks passed:

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
source_stage91_report_is_stage91: passed
stage91_protocol_id_matches: passed
stage91_candidate_id_matches: passed
user_confirmed_frozen_protocol: passed
confirmed_protocol_id_matches: passed
stage91_allows_train_dev_metrics_after_confirmation: passed
stage91_final_test_metrics_locked: passed
stage91_forbids_test_tuning: passed
stage91_default_runtime_policy_unchanged: passed
candidate_config_grid_matches_frozen_protocol: passed
section_index_has_nonempty_sections: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
source_doc_ids_not_used_as_runtime_evidence: passed
changed_case_fields_public_safe: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_train_hit_at_10.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_dev_hit_at_10.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_dev_search_depth_net.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_dev_top10_changes.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_guard_check_status.svg
```

Stage92 JSON SHA256:

```text
9D2DCA0C4FA2A34320DC3700CB24CAB06088E2670107F8D6A8C932817640E338
```

Visualization SHA256:

```text
stage92_section_signal_dev_delta_hit_at_10.svg: B3AE28DC10167253C7E132DB2FF48764408277FB00CA5C78A053EDEB46885E54
stage92_section_signal_dev_hit_at_10.svg: C1CDAE269291D1FC2BC1760B3A06BD5E34A7E56A2C2D1ECE495E673CE39E5F5A
stage92_section_signal_dev_search_depth_net.svg: FE45CF43AED44CFF296A94369C3D7DE2CF7D8EA3DD07E04627BAFF0D76F22D7E
stage92_section_signal_dev_top10_changes.svg: 106CE640DF79DA1C93AECA461A1CFE9939C1608B90C9A7C84ED120DE4CB53E2E
stage92_section_signal_guard_check_status.svg: 08B2F17B19C731B8E4E9B062C5DE1A0724631DF18677492B4F7BB5D856C65E7C
stage92_section_signal_train_hit_at_10.svg: 800805855133EFAFD2625FF94F3ABC1B4D858C4D000E7D2D152C70935FA47127
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_section_signal_comparison.py scripts\run_primeqa_hybrid_section_signal_comparison.py tests\test_primeqa_hybrid_section_signal_comparison.py: passed
pytest -q tests\test_primeqa_hybrid_section_signal_comparison.py: 3 passed
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage92 JSON: no matches
git check-ignore Stage92 JSON, console, and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 241 passed
git diff --check: passed
```

## Next Step

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
