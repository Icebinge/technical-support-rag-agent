# PrimeQA Hybrid Structured Query Stop Decision

This document records Stage 90.

## Scope

Stage 90 stops the Stage84 structured query keyphrase compaction route after
the Stage89 train/dev comparison failed the train-selected dev hit@10
improvement gate and the top10 regression guard.

This is a decision checkpoint. It reads the public-safe Stage84 and Stage89
reports, does not load train/dev/test split files, does not run new retrieval
metrics, does not run final metrics, does not use source `DOC_IDS` as runtime
retrieval evidence, and does not change runtime defaults.

The stop decision was explicitly confirmed for this run:

```text
route_id: structured_query_stop_decision
confirmed: true
confirmation_note: user confirmed Stage90 stop decision in current turn
```

## Command

```text
python scripts\decide_primeqa_hybrid_structured_query_stop.py ^
  --user-confirmed-stop ^
  --confirmation-note "user confirmed Stage90 stop decision in current turn" ^
  --output artifacts\primeqa_hybrid_structured_query_stop_decision_stage90.json ^
  --visualization-dir artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals
```

The run completed in `0.002s`.

## Evidence Used

Stage89 selected this config on train:

```text
sqkc_title_guarded_action_error_v1
```

Selected query view:

```text
title_guarded_action_error_product_terms
```

Train evidence:

```text
train hit@10 delta: -0.0027
train top10 improvements: 14
train top10 regressions: 15
```

Dev evidence:

```text
dev hit@10 delta: -0.0527
dev top10 improvements: 1
dev top10 regressions: 5
dev rank-up within top10: 5
dev rank-down within top10: 5
dev not-found@50 delta: -1
dev rank 11-50 delta: +5
average compacted query token count delta: -36.0921
```

Stage84 target metric contract for this route:

```text
primary: train-selected dev hit@10 must improve over BM25 baseline
secondary: top10 regression count must be lower than improvement count
guard: no query view may be selected by dev-only performance
```

## Decision

```text
status: primeqa_hybrid_structured_query_route_stopped
stopped_candidate_id: structured_query_keyphrase_compaction_design
stopped_protocol_id: structured_query_keyphrase_compaction_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: section_signal_guarded_expansion_design
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

The route is stopped because the train-selected structured-query config
reduced query length, but dev hit@10 decreased and dev top10 regressions
outnumbered improvements. This fails the Stage84/Stage88 target metric
contract and provides no justification for runtime defaultization or final-test
gate opening.

## Candidate Queue

Original Stage84 execution order:

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Prior stopped candidate:

```text
lexical_cluster_diversity_rerank_design
```

Remaining execution order after Stage90:

```text
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Next candidate:

```text
section_signal_guarded_expansion_design
```

This is a next-protocol candidate only. Stage90 does not freeze the protocol for
it and does not run any metrics for it.

## Guard Checks

All `22 / 22` guard checks passed:

```text
source_stage84_report_is_stage84: passed
source_stage89_report_is_stage89: passed
user_confirmed_stage90_stop_decision: passed
stage89_comparison_completed: passed
stage89_candidate_matches_structured_query: passed
stage89_protocol_matches_structured_query: passed
stage84_candidate_metric_contract_requires_train_selected_dev_hit10_gain: passed
stage89_primary_contract_failed: passed
stage89_secondary_contract_failed: passed
stage89_train_selected_config_has_dev_hit10_loss: passed
stage89_dev_top10_net_negative: passed
stage89_final_test_metrics_locked: passed
stage89_final_test_gate_closed: passed
stage89_forbids_test_tuning: passed
stage89_default_runtime_policy_unchanged: passed
stage84_execution_order_contains_stopped_candidate: passed
stage84_next_candidate_available_after_structured_query_stop: passed
prior_lcdr_route_removed_from_remaining_queue: passed
source_doc_ids_not_selected_as_next_candidate: passed
stage90_no_new_retrieval_metrics_run: passed
stage90_final_test_metrics_not_run: passed
stage90_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_structured_query_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_structured_query_dev_change_counts.svg
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_second_wave_remaining_candidate_priority.svg
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_structured_query_stop_decision_flags.svg
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_structured_query_stop_guard_check_status.svg
```

Stage90 JSON SHA256:

```text
E7A5B120B709E0802B352773ECE3561442943CAF6357C2F2075101A1765997B6
```

Visualization SHA256:

```text
stage90_second_wave_remaining_candidate_priority.svg: 246DC08BC22076751014FEC62C5A14DC1EBE3241C85C998CE0EC5B1CCF13AC35
stage90_structured_query_dev_change_counts.svg: B45AF0F1E99202563CDCF085217A592AE55A93222B7F04AC3EF6706546F851B6
stage90_structured_query_stop_decision_flags.svg: 15F5CC5415B8327F8B814441B9040BA13A0440D9C46744E7079E3B2BF6D18E22
stage90_structured_query_stop_guard_check_status.svg: 7B39D1C0E041374C99F8467170CE0AC21BA45A042C2BD69DA4D6CBA31F9111BA
stage90_structured_query_train_dev_hit10_delta.svg: D6A2093E2353AF37EEE5B3B4F228375136CBA7AAD5AB604E22BD6F2C1C98D012
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_structured_query_stop_decision.py scripts\decide_primeqa_hybrid_structured_query_stop.py tests\test_primeqa_hybrid_structured_query_stop_decision.py: passed
pytest -q tests\test_primeqa_hybrid_structured_query_stop_decision.py: 3 passed
Select-String raw question / answer / document / snippet / query-term field patterns over Stage90 JSON: no matches
git check-ignore Stage90 JSON and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 235 passed
git diff --check: passed
```

## Next Step

Stage91 confirmed and froze the train/dev-only protocol for
`section_signal_guarded_expansion_design` as
`section_signal_guarded_expansion_train_dev_v1`. Stage91 did not run retrieval
metrics and did not change runtime defaults.

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

Stage96 stopped score-margin BM25 normalization as a retrieval-recall route and
left runtime defaults unchanged. The remaining Stage84 queue is:

```text
selective_dense_sparse_low_overlap_gate_design
```

The current next step is Stage97: confirm and freeze the train/dev-only
protocol for `selective_dense_sparse_low_overlap_gate_design`. The frozen test
split remains locked, final metrics must not be run, source `DOC_IDS` must not
be used as runtime retrieval evidence, dev-only observations must not select
runtime rules, and runtime defaults remain unchanged.
