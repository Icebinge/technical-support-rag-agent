# PrimeQA Hybrid Second Wave Retrieval Candidate Design

This document records Stage 84.

## Scope

Stage 84 designs a second-wave train/dev-only retrieval candidate set after
Stage83 confirmed that the original Stage76 retrieval-recall candidates were
exhausted.

This stage consumes saved public-safe Stage75-Stage83 reports. It does not load
the frozen test split, does not run new retrieval metrics, does not run final
metrics, does not use source `DOC_IDS` as runtime retrieval evidence, and does
not change runtime defaults.

The route was explicitly confirmed for this run:

```text
route_id: second_wave_retrieval_candidate_design
confirmed: true
confirmation_note: user confirmed recommended Stage84 route in current turn
```

## Command

```text
python scripts\design_primeqa_hybrid_second_wave_retrieval_candidates.py ^
  --user-confirmed-route ^
  --confirmation-note "user confirmed recommended Stage84 route in current turn" ^
  --output artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json ^
  --visualization-dir artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals
```

The run completed in `0.006s`.

## Stage75 Miss Summary

```text
evaluated_questions: 446
hit_at_top_k: 0.6682
miss_count: 148
miss_count_by_split:
  train: 125
  dev: 23
source_doc_ids_oracle_presence_count: 148
```

Rank buckets:

```text
not_found_top50: 110
rank_21_to_50: 24
rank_11_to_20: 14
```

Important miss drivers:

```text
unique_terms_16_plus: 125
top1_query_overlap_exceeds_gold: 103
gold_doc_query_overlap_ratio_lt_0_25: 20
```

The source `DOC_IDS` oracle presence count remains diagnostic only. It is not a
runtime retrieval signal.

## Prior Route Evidence

```text
Stage77 query-view ablation:
  selected_dev_hit10_delta: -0.0395
  selected_dev_top10_net: -3
  lesson: simple title-only or de-duplicated query replacement regressed

Stage78 fielded title/text BM25:
  selected_dev_hit10_delta: 0.0000
  selected_dev_top10_net: 0
  selected_dev_mrr_delta: +0.0219
  lesson: title signal helped MRR/hit@1 but did not improve hit@10

Stage79 section BM25 rollup:
  selected_dev_hit10_delta: -0.0527
  selected_dev_top10_net: -4
  dev_search_depth_net: -1
  lesson: ungated section rollup regressed overall

Stage81 dense+sparse RRF:
  selected_dev_hit10_delta: -0.0132
  selected_dev_top10_net: -1
  selected_dev_not_found_delta: -6
  selected_dev_search_depth_net: +6
  lesson: dense route reduced not-found@50 but regressed top10

Stage82 BM25 k1/b grid:
  selected_dev_hit10_delta: 0.0000
  best_dev_non_selected_config_id: bm25_grid__k1_1_20__b_0_95
  best_dev_non_selected_hit10_delta: +0.0131
  lesson: dev-only b=0.95 signal cannot select a runtime rule
```

## Candidate Designs

| Candidate | Priority | Target misses | Dev targets | Prior signal | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| lexical_cluster_diversity_rerank_design | 210 | 143 | 22 | 0.50 | recommended for protocol design |
| structured_query_keyphrase_compaction_design | 207 | 143 | 22 | 0.35 | recommended for protocol design |
| section_signal_guarded_expansion_design | 174 | 119 | 17 | 0.45 | recommended for protocol design |
| score_margin_bm25_normalization_gate_design | 171 | 111 | 18 | 0.48 | recommended for protocol design |
| selective_dense_sparse_low_overlap_gate_design | 159 | 111 | 17 | 0.68 | recommended for protocol design |
| source_doc_ids_oracle_union_blocked | 0 | 148 | 23 | 0.00 | blocked |

Recommended execution order:

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

The recommended next candidate is:

```text
lexical_cluster_diversity_rerank_design
```

This is still a design-stage recommendation. It is not a runtime change and is
not a train/dev metric result.

## Guard Checks

All `14 / 14` guard checks passed:

```text
source_reports_are_expected_stages: passed
user_confirmed_stage84_recommended_route: passed
stage83_recommended_route_matches_stage84: passed
stage83_required_confirmation_was_respected: passed
stage76_candidates_are_exhausted: passed
stage83_has_no_runtime_advancing_candidate: passed
source_doc_ids_candidate_not_reintroduced: passed
source_doc_ids_candidate_remains_blocked: passed
all_source_decisions_keep_final_test_locked: passed
all_source_decisions_forbid_test_tuning: passed
all_source_decisions_keep_runtime_defaults_unchanged: passed
stage84_design_only_no_new_retrieval_metrics: passed
stage84_final_test_metrics_not_run: passed
stage84_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_candidate_priority_scores.svg
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_candidate_target_misses.svg
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_candidate_dev_targets.svg
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_candidate_prior_signal_scores.svg
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_allowed_vs_blocked_candidates.svg
```

Stage84 JSON SHA256:

```text
AB09F566ADF085A05DDBB28BF7065B4FD9320C8148D7E2F723D73D73F50EB8EF
```

Visualization SHA256:

```text
stage84_second_wave_allowed_vs_blocked_candidates.svg: 0E487E4ABED942B356CCC5EA3CF6CB0D3DFB6DC5F3F617C8A1046AB3C879DEAF
stage84_second_wave_candidate_dev_targets.svg: 98C08E3FDB17FCD59E41985CC117F04173A840633B9535F0104138318EDD51C5
stage84_second_wave_candidate_prior_signal_scores.svg: 359580A964F6D3F314974155B7AAE03BBC7E5606D9535B7F7A34FD4F4DDB5503
stage84_second_wave_candidate_priority_scores.svg: EE2AF96C5B22636A388FF78F559EF9D46D18A80585FE6AC931C7FB763E40AF1F
stage84_second_wave_candidate_target_misses.svg: 5EF01CCF820FE76BFDCC4BCF0753B9587D9EDD61CAA5640685D4C9CA216BBD54
```

## Validation

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_second_wave_retrieval_candidate_design.py scripts\design_primeqa_hybrid_second_wave_retrieval_candidates.py tests\test_primeqa_hybrid_second_wave_retrieval_candidate_design.py: passed
pytest -q tests\test_primeqa_hybrid_second_wave_retrieval_candidate_design.py: 2 passed
Select-String raw question / answer / document / snippet field patterns over Stage84 JSON: no matches
git check-ignore Stage84 JSON and SVG artifacts: ignored by .gitignore
ruff check .: passed
pytest -q: 217 passed
git diff --check: passed
```

## Decision

```text
status: primeqa_hybrid_second_wave_retrieval_candidate_design_completed
recommended_next_candidate_id: lexical_cluster_diversity_rerank_design
requires_user_confirmation_before_train_dev_run: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage85 confirmed and froze the train/dev-only protocol for
`lexical_cluster_diversity_rerank_design` as
`lexical_cluster_diversity_rerank_train_dev_v1`. No train/dev metrics were run
in Stage85.

Stage86 then ran that frozen protocol after user confirmation. It selected
`lcdr_penalty_0_06_title_query_cluster` on train, but the selected config had
`dev hit@10 delta = 0.0000`. Because the Stage84 target metric contract
requires dev hit@10 to improve over BM25 baseline, lexical cluster diversity
does not advance to runtime.

Stage87 stopped lexical cluster diversity as a retrieval-recall route and
blocked defaultization for that route. The next candidate in the Stage84 queue
is `structured_query_keyphrase_compaction_design`.

Stage88 confirmed and froze the train/dev-only protocol for
`structured_query_keyphrase_compaction_design` as
`structured_query_keyphrase_compaction_train_dev_v1`. Stage88 did not run
retrieval metrics and did not change runtime defaults.

Stage89 then ran that frozen protocol after user confirmation. It selected
`sqkc_title_guarded_action_error_v1` on train, but the selected config had
`dev hit@10 delta = -0.0527`, with `1` dev top10 improvement and `5` dev top10
regressions. Because the Stage88 target metric contract requires train-selected
dev hit@10 to improve over BM25 baseline, structured query compaction does not
advance to runtime.

Stage90 stopped structured query keyphrase compaction as a retrieval-recall
route and left runtime defaults unchanged. The remaining Stage84 queue is:

```text
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

The current next step is Stage91: confirm and freeze the train/dev-only
protocol for `section_signal_guarded_expansion_design`. The frozen test split
remains locked, final metrics must not be run, source `DOC_IDS` must not be used
as runtime retrieval evidence, and runtime defaults remain unchanged.
