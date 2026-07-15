# PrimeQA Hybrid Score-Margin BM25 Stop Decision

This document records Stage 96.

## Scope

Stage 96 stops the Stage84 score-margin BM25 normalization gate route after the
Stage95 train/dev comparison failed the train-selected dev hit@10 improvement
gate and the rank 11-50 near-miss reduction gate.

This is a decision checkpoint. It reads the public-safe Stage84 and Stage95
reports, does not load train/dev/test split files, does not run new retrieval
metrics, does not run final metrics, does not use source `DOC_IDS` as runtime
retrieval evidence, and does not change runtime defaults.

The stop decision was explicitly confirmed for this run:

```text
route_id: score_margin_bm25_stop_decision
confirmed: true
confirmation_note: user confirmed Stage96 score-margin BM25 stop decision in current turn
```

## Command

```text
python scripts\decide_primeqa_hybrid_score_margin_bm25_stop.py ^
  --user-confirmed-stop ^
  --confirmation-note "user confirmed Stage96 score-margin BM25 stop decision in current turn" ^
  --output artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96.json ^
  --visualization-dir artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals
```

The run completed successfully and wrote console output to:

```text
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96.console.txt
```

## Evidence Used

Stage95 selected this config on train:

```text
smbn_rank11_20_long_doc_b095_margin_v1
```

Train evidence:

```text
train hit@10 delta: +0.0000
train rank 11-50 delta: 0
train top10 improvements: 0
train top10 regressions: 0
train score-margin gate promotions: 1
train length-band gate count: 1
```

Dev evidence:

```text
dev hit@10 delta: +0.0000
dev rank 11-50 delta: 0
dev top10 improvements: 0
dev top10 regressions: 0
dev rank-up within top10: 0
dev rank-down within top10: 0
dev not-found@50 delta: 0
dev score-margin gate promotions: 0
dev length-band gate count: 0
```

Stage84 target metric contract for this route:

```text
primary: train-selected rule must improve dev hit@10
secondary: rank 11-50 near misses should decrease
guard: dev-only b=0.95 observations cannot select a runtime rule
```

## Decision

```text
status: primeqa_hybrid_score_margin_bm25_route_stopped
stopped_candidate_id: score_margin_bm25_normalization_gate_design
stopped_protocol_id: score_margin_bm25_normalization_gate_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: selective_dense_sparse_low_overlap_gate_design
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

The route is stopped because the train-selected score-margin BM25 config had
no dev hit@10 gain, did not reduce dev rank 11-50 near misses, and produced no
dev score-margin gate promotions. This fails the Stage84/Stage94 target metric
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
section_signal_guarded_expansion_design
```

Remaining execution order after Stage96:

```text
selective_dense_sparse_low_overlap_gate_design
```

Next candidate:

```text
selective_dense_sparse_low_overlap_gate_design
```

This is a next-protocol candidate only. Stage96 does not freeze the protocol
for it and does not run any metrics for it.

## Guard Checks

All `30 / 30` guard checks passed:

```text
source_stage84_report_is_stage84: passed
source_stage95_report_is_stage95: passed
user_confirmed_stage96_stop_decision: passed
stage95_comparison_completed: passed
stage95_candidate_matches_score_margin_bm25: passed
stage95_protocol_matches_score_margin_bm25: passed
stage84_candidate_metric_contract_requires_train_selected_dev_hit10_gain: passed
stage84_candidate_metric_contract_requires_rank_11_to_50_decrease: passed
stage84_candidate_guard_blocks_dev_only_b95_runtime_selection: passed
stage95_primary_contract_failed: passed
stage95_secondary_contract_failed: passed
stage95_guard_contract_passed: passed
stage95_train_selected_config_has_no_dev_hit10_gain: passed
stage95_dev_rank_11_to_50_not_reduced: passed
stage95_dev_top10_net_not_positive: passed
stage95_selected_config_has_no_dev_score_margin_promotions: passed
stage95_final_test_metrics_locked: passed
stage95_final_test_gate_closed: passed
stage95_forbids_test_tuning: passed
stage95_default_runtime_policy_unchanged: passed
stage95_raw_question_answer_document_or_query_text_not_written: passed
stage84_execution_order_contains_stopped_candidate: passed
stage84_next_candidate_available_after_score_margin_stop: passed
prior_lcdr_route_removed_from_remaining_queue: passed
prior_structured_query_route_removed_from_remaining_queue: passed
prior_section_signal_route_removed_from_remaining_queue: passed
source_doc_ids_not_selected_as_next_candidate: passed
stage96_no_new_retrieval_metrics_run: passed
stage96_final_test_metrics_not_run: passed
stage96_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_score_margin_bm25_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_score_margin_bm25_dev_change_counts.svg
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_second_wave_remaining_candidate_priority.svg
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_score_margin_bm25_stop_decision_flags.svg
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_score_margin_bm25_stop_guard_check_status.svg
```

Stage96 JSON SHA256:

```text
1EF1180676209A5A311E2AB22A4745E07E12FBCBCAE594D1774CD9CD58634C87
```

Visualization SHA256:

```text
stage96_score_margin_bm25_dev_change_counts.svg: A8F65CFF117762A7B65F29A61C0177E0785ABEEB677309205A3413E54258C9A1
stage96_score_margin_bm25_stop_decision_flags.svg: 437EEBD0F4508236D974B469BD8CC6ADE37F59EB4AC767230EDFC6248AB505DB
stage96_score_margin_bm25_stop_guard_check_status.svg: D9A2EC65EAE7216D499BB5C7E0343E180BABE1C099354E5D6363BE84DF8C5BCC
stage96_score_margin_bm25_train_dev_hit10_delta.svg: 072D4F52F9DE8F77FF69F8F7E98D5C4CE050A9E6DC339643E4EC81538128CF75
stage96_second_wave_remaining_candidate_priority.svg: 6606310CDC836D6C433542C63C7B94F0F5A73C5F39BA96D8E390218A206B71D5
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_score_margin_bm25_stop_decision.py scripts\decide_primeqa_hybrid_score_margin_bm25_stop.py tests\test_primeqa_hybrid_score_margin_bm25_stop_decision.py: passed
pytest -q tests\test_primeqa_hybrid_score_margin_bm25_stop_decision.py: 3 passed
Select-String raw/private field patterns over Stage96 JSON: no matches
git check-ignore Stage96 JSON, console, and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 253 passed
git diff --check: passed
```

## Next Step

Stage97 should confirm and freeze the train/dev-only protocol for
`selective_dense_sparse_low_overlap_gate_design`. The frozen test split remains
locked, final metrics must not be run, source `DOC_IDS` must not be used as
runtime retrieval evidence, dev-only observations must not select runtime
rules, and runtime defaults remain unchanged.
