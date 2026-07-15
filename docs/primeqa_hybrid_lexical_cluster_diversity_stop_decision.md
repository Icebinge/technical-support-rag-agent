# PrimeQA Hybrid Lexical Cluster Diversity Stop Decision

This document records Stage 87.

## Scope

Stage 87 stops the Stage84 lexical cluster diversity rerank route after the
Stage86 train/dev comparison failed the dev hit@10 improvement gate.

This is a decision checkpoint. It reads the public-safe Stage84 and Stage86
reports, does not load train/dev/test split files, does not run new retrieval
metrics, does not run final metrics, does not use source `DOC_IDS` as runtime
retrieval evidence, and does not change runtime defaults.

The stop decision was explicitly confirmed for this run:

```text
route_id: lexical_cluster_diversity_stop_decision
confirmed: true
confirmation_note: user confirmed Stage87 stop decision in current turn
```

## Command

```text
python scripts\decide_primeqa_hybrid_lexical_cluster_diversity_stop.py ^
  --user-confirmed-stop ^
  --confirmation-note "user confirmed Stage87 stop decision in current turn" ^
  --output artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87.json ^
  --visualization-dir artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals
```

The run completed in `0.001s`.

## Evidence Used

Stage86 selected this config on train:

```text
lcdr_penalty_0_06_title_query_cluster
```

Train evidence:

```text
train hit@10 delta: +0.0054
train top10 improvements: 4
train top10 regressions: 2
```

Dev evidence:

```text
dev hit@10 delta: +0.0000
dev top10 improvements: 0
dev top10 regressions: 0
dev rank-up within top10: 0
dev rank-down within top10: 0
dev not-found@50 delta: 0
dev rank 11-50 delta: 0
```

Stage84 target metric contract for this route:

```text
primary: dev hit@10 must improve over BM25 baseline
secondary: top1-overlap decoy misses should decrease
guard: no title/body text should be written to reports
```

## Decision

```text
status: primeqa_hybrid_lexical_cluster_diversity_route_stopped
stopped_candidate_id: lexical_cluster_diversity_rerank_design
stopped_protocol_id: lexical_cluster_diversity_rerank_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: structured_query_keyphrase_compaction_design
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

The route is stopped because the train-selected LCDR config improved train
hit@10 but did not improve dev hit@10. This fails the Stage84 primary metric
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

Remaining execution order after Stage87:

```text
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Next candidate:

```text
structured_query_keyphrase_compaction_design
```

This is a next-protocol candidate only. Stage87 does not freeze the protocol for
it and does not run any metrics for it.

## Guard Checks

All `19 / 19` guard checks passed:

```text
source_stage84_report_is_stage84: passed
source_stage86_report_is_stage86: passed
user_confirmed_stage87_stop_decision: passed
stage86_comparison_completed: passed
stage86_candidate_matches_lcdr: passed
stage86_protocol_matches_lcdr: passed
stage84_candidate_metric_contract_requires_dev_hit10_gain: passed
stage86_train_selected_config_has_no_dev_hit10_gain: passed
stage86_dev_top10_net_not_positive: passed
stage86_final_test_metrics_locked: passed
stage86_final_test_gate_closed: passed
stage86_forbids_test_tuning: passed
stage86_default_runtime_policy_unchanged: passed
stage84_execution_order_contains_stopped_candidate: passed
stage84_next_candidate_available_after_lcdr_stop: passed
source_doc_ids_not_selected_as_next_candidate: passed
stage87_no_new_retrieval_metrics_run: passed
stage87_final_test_metrics_not_run: passed
stage87_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_lcdr_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_lcdr_dev_change_counts.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_second_wave_remaining_candidate_priority.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_lcdr_stop_decision_flags.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_lcdr_stop_guard_check_status.svg
```

Stage87 JSON SHA256:

```text
F5C9BC614F8210B20DD6B099C36126D1078FE1262169BD874AE32136A7F9FD79
```

Visualization SHA256:

```text
stage87_lcdr_dev_change_counts.svg: C00034DD885257891031C95D2DFEC9FE6F7E10459CDB076377D56A7B8C6042F1
stage87_lcdr_stop_decision_flags.svg: 4283DAA01223C72FBA86049B299BB89405781F6BFBEA798364D2A25BECFD9762
stage87_lcdr_stop_guard_check_status.svg: B21CFCE72854F0AEF7EF46EF361CEBAB96C90FF25B4AC429ED2E1C09C1F5B1B0
stage87_lcdr_train_dev_hit10_delta.svg: 46427B4BAE399D2084202B54DDB13BD1C91F7F82AF03AC17971B61F275E6187F
stage87_second_wave_remaining_candidate_priority.svg: E0D20D4DB853F04AB4CCADC75D63BA3365079689F9825851BDCDCFB3F449CD78
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_lexical_cluster_diversity_stop_decision.py scripts\decide_primeqa_hybrid_lexical_cluster_diversity_stop.py tests\test_primeqa_hybrid_lexical_cluster_diversity_stop_decision.py: passed
pytest -q tests\test_primeqa_hybrid_lexical_cluster_diversity_stop_decision.py: 3 passed
Select-String raw question / answer / document / snippet field patterns over Stage87 JSON: no matches
git check-ignore Stage87 JSON and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 226 passed
git diff --check: passed
```

## Next Step

Stage88 confirmed and froze the train/dev-only protocol for
`structured_query_keyphrase_compaction_design` as
`structured_query_keyphrase_compaction_train_dev_v1`. Stage88 did not run
retrieval metrics and did not change runtime defaults.

The current next step is Stage89: run the frozen train/dev-only structured query
keyphrase compaction comparison after user confirmation. The frozen test split
remains locked, final metrics must not be run, source `DOC_IDS` must not be used
as runtime retrieval evidence, and runtime defaults remain unchanged.
