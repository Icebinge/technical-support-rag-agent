# PrimeQA Hybrid Section Signal Stop Decision

This document records Stage 93.

## Scope

Stage 93 stops the Stage84 section signal guarded expansion route after the
Stage92 train/dev comparison failed the dev hit@10 improvement gate and the
search-depth improvement gate.

This is a decision checkpoint. It reads the public-safe Stage84 and Stage92
reports, does not load train/dev/test split files, does not run new retrieval
metrics, does not run final metrics, does not use source `DOC_IDS` as runtime
retrieval evidence, and does not change runtime defaults.

The stop decision was explicitly confirmed for this run:

```text
route_id: section_signal_stop_decision
confirmed: true
confirmation_note: user confirmed Stage93 stop decision in current turn
```

## Command

```text
python scripts\decide_primeqa_hybrid_section_signal_stop.py ^
  --user-confirmed-stop ^
  --confirmation-note "user confirmed Stage93 stop decision in current turn" ^
  --output artifacts\primeqa_hybrid_section_signal_stop_decision_stage93.json ^
  --visualization-dir artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals
```

The run completed in `0.007s`.

## Evidence Used

Stage92 selected this config on train:

```text
ssgx_section_top50_injection_guard_v1
```

Train evidence:

```text
train hit@10 delta: +0.0000
train search-depth net improvement: +1
train top10 improvements: 1
train top10 regressions: 1
train section-signal promotions: 92
```

Dev evidence:

```text
dev hit@10 delta: +0.0000
dev search-depth net improvement: 0
dev top10 improvements: 0
dev top10 regressions: 0
dev not-found@50 delta: 0
dev rank 11-50 delta: 0
dev section-signal promotions: 22
dev protected top10 demotion actions: 22
```

Stage84 target metric contract for this route:

```text
primary: dev hit@10 must improve over BM25 baseline
secondary: search-depth improvements must exceed regressions
guard: section signal must not demote existing BM25 top10 hits by default
```

## Decision

```text
status: primeqa_hybrid_section_signal_route_stopped
stopped_candidate_id: section_signal_guarded_expansion_design
stopped_protocol_id: section_signal_guarded_expansion_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: score_margin_bm25_normalization_gate_design
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

The route is stopped because the train-selected section-signal config executed
promotion actions, but dev hit@10 did not improve and dev search-depth net
improvement remained zero. This fails the Stage84/Stage91 target metric
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

Prior stopped candidates:

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
```

Remaining execution order after Stage93:

```text
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Next candidate:

```text
score_margin_bm25_normalization_gate_design
```

This is a next-protocol candidate only. Stage93 does not freeze the protocol for
it and does not run any metrics for it.

## Guard Checks

All `26 / 26` guard checks passed:

```text
source_stage84_report_is_stage84: passed
source_stage92_report_is_stage92: passed
user_confirmed_stage93_stop_decision: passed
stage92_comparison_completed: passed
stage92_candidate_matches_section_signal: passed
stage92_protocol_matches_section_signal: passed
stage84_candidate_metric_contract_requires_dev_hit10_gain: passed
stage84_candidate_metric_contract_requires_search_depth_gain: passed
stage92_primary_contract_failed: passed
stage92_secondary_contract_failed: passed
stage92_guard_contract_passed: passed
stage92_train_selected_config_has_no_dev_hit10_gain: passed
stage92_dev_search_depth_net_not_positive: passed
stage92_dev_top10_net_not_positive: passed
stage92_final_test_metrics_locked: passed
stage92_final_test_gate_closed: passed
stage92_forbids_test_tuning: passed
stage92_default_runtime_policy_unchanged: passed
stage84_execution_order_contains_stopped_candidate: passed
stage84_next_candidate_available_after_section_signal_stop: passed
prior_lcdr_route_removed_from_remaining_queue: passed
prior_structured_query_route_removed_from_remaining_queue: passed
source_doc_ids_not_selected_as_next_candidate: passed
stage93_no_new_retrieval_metrics_run: passed
stage93_final_test_metrics_not_run: passed
stage93_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_section_signal_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_section_signal_dev_change_counts.svg
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_second_wave_remaining_candidate_priority.svg
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_section_signal_stop_decision_flags.svg
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_section_signal_stop_guard_check_status.svg
```

Stage93 JSON SHA256:

```text
F409A99EA7DCF0823C140EFA6C69AED512A9B7BAE23FE63442602C21B59BF90A
```

Visualization SHA256:

```text
stage93_second_wave_remaining_candidate_priority.svg: E29F4679863809EEC1969A0D079BCABF0169A16F44AC1B6F93BDCE655C0A3332
stage93_section_signal_dev_change_counts.svg: 883D57F772EDFA60FFFE1E4C1471ACA946D6490C695EE2DCCD47DDD67EBBA481
stage93_section_signal_stop_decision_flags.svg: 4CE969467572C3EC9A63C8F607C8E25D736E93F6AE1DFA13447BD010C1E67E23
stage93_section_signal_stop_guard_check_status.svg: 2A3FCCEDE15EB338F6C0289451644A07D02C397ADAE8B64141DF08A0B9F338F3
stage93_section_signal_train_dev_hit10_delta.svg: DA5152C754A0246D8C0316440574A97EFFACEC17FF3C7AD0B38050AA7C6FA120
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_section_signal_stop_decision.py scripts\decide_primeqa_hybrid_section_signal_stop.py tests\test_primeqa_hybrid_section_signal_stop_decision.py: passed
pytest -q tests\test_primeqa_hybrid_section_signal_stop_decision.py: 3 passed
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage93 JSON: no matches
git check-ignore Stage93 JSON and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 244 passed
git diff --check: passed
```

## Next Step

Stage94 confirmed and froze the train/dev-only protocol for
`score_margin_bm25_normalization_gate_design` as
`score_margin_bm25_normalization_gate_train_dev_v1`. Stage94 did not run
retrieval metrics and did not change runtime defaults.

The current next step is Stage95: run the frozen train/dev-only score-margin
BM25 normalization gate comparison after user confirmation. The frozen test
split remains locked, final metrics must not be run, source `DOC_IDS` must not
be used as runtime retrieval evidence, Stage82 dev-only `b=0.95` observations
must not select the runtime rule, and runtime defaults remain unchanged.
