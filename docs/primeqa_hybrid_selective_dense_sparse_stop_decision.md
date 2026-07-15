# PrimeQA Hybrid Selective Dense+Sparse Stop Decision

This document records Stage 99.

## Scope

Stage 99 stops the Stage84 selective dense+sparse low-overlap gate route after
the Stage98 train/dev comparison failed the frozen Stage97 metric contract.

This is a decision checkpoint. It reads the public-safe Stage84, Stage97, and
Stage98 reports. It does not load train/dev/test split files, does not run new
retrieval metrics, does not run final metrics, does not use source `DOC_IDS` as
runtime retrieval evidence, does not tune dev thresholds, and does not change
runtime defaults.

The stop decision was explicitly confirmed for this run:

```text
route_id: selective_dense_sparse_stop_decision
confirmed: true
confirmation_note: user confirmed Stage99 selective dense sparse stop decision in current turn
```

## Command

```text
python scripts\decide_primeqa_hybrid_selective_dense_sparse_stop.py ^
  --user-confirmed-stop ^
  --confirmation-note "user confirmed Stage99 selective dense sparse stop decision in current turn" ^
  --output artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99.json ^
  --visualization-dir artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals
```

The run completed successfully and wrote console output to:

```text
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99.console.txt
```

## Evidence Used

Stage98 selected this policy on train:

```text
sdsl_minilm_low_overlap_conservative_v1
```

Train evidence:

```text
train hit@10 delta: +0.0000
train not-found@50 delta: 0
train gate activations: 2
train promotions: 2
```

Dev evidence:

```text
dev hit@10 delta: +0.0000
dev hit@1 delta: +0.0000
dev not-found@50 delta: 0
dev top10 improvements: 0
dev top10 regressions: 0
dev gate activations: 0
dev promotions: 0
```

Stage97/Stage84 metric contract for this route:

```text
primary: train-selected gated policy must improve dev hit@10
secondary: dev not-found@50 should decrease without hit@1 collapse
guard: no downloads and no dev-selected gate thresholds
```

## Decision

```text
status: primeqa_hybrid_selective_dense_sparse_route_stopped
stopped_candidate_id: selective_dense_sparse_low_overlap_gate_design
stopped_protocol_id: selective_dense_sparse_low_overlap_gate_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: null
remaining_actionable_candidate_count: 0
route_family_exhausted: true
can_continue_train_dev_development: false
requires_user_confirmation_before_next_experiment: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

The route is stopped because the train-selected policy had no dev hit@10 gain,
no dev hit@1 gain, no dev not-found@50 reduction, no dev top10 improvements, no
dev gate activations, and no dev promotions. This fails the frozen Stage97
primary and secondary metric contracts. It provides no justification for runtime
defaultization or final-test gate opening.

## Route Queue

Original Stage84 execution order:

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Stopped candidates after Stage99:

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Remaining actionable candidates after Stage99:

```text
none
```

The only remaining non-stopped Stage84 diagnostic is:

```text
source_doc_ids_oracle_union_blocked
```

That route remains blocked from train/dev experimentation and runtime
defaultization because source `DOC_IDS` are not runtime retrieval evidence.

## Guard Checks

All `37 / 37` guard checks passed:

```text
source_stage84_report_is_stage84: passed
source_stage97_report_is_stage97: passed
source_stage98_report_is_stage98: passed
user_confirmed_stage99_stop_decision: passed
stage97_protocol_frozen: passed
stage97_candidate_matches_selective_dense_sparse: passed
stage97_protocol_matches_selective_dense_sparse: passed
stage97_downloads_forbidden: passed
stage97_train_selection_rule_forbids_dev_selection: passed
stage98_comparison_completed: passed
stage98_protocol_matches_stage97: passed
stage84_candidate_metric_contract_requires_train_selected_dev_hit10_gain: passed
stage84_candidate_metric_contract_requires_not_found_decrease: passed
stage84_candidate_guard_blocks_downloads_and_dev_thresholds: passed
stage98_primary_contract_failed: passed
stage98_secondary_contract_failed: passed
stage98_guard_contract_passed: passed
stage98_train_selected_policy_has_no_dev_hit10_gain: passed
stage98_dev_not_found_not_reduced: passed
stage98_dev_hit1_not_improved: passed
stage98_dev_top10_net_not_positive: passed
stage98_selected_policy_has_no_dev_gate_activation: passed
stage98_selected_policy_has_no_dev_promotions: passed
stage98_test_split_not_loaded: passed
stage98_final_metrics_not_run: passed
stage98_final_test_gate_closed: passed
stage98_forbids_test_tuning: passed
stage98_default_runtime_policy_unchanged: passed
stage98_artifact_safety_flags_false: passed
stage84_execution_order_contains_stopped_candidate: passed
prior_second_wave_routes_removed_from_remaining_queue: passed
selective_dense_sparse_removed_from_remaining_queue: passed
source_doc_ids_candidate_is_blocked_not_actionable: passed
no_remaining_second_wave_actionable_candidates: passed
stage99_no_new_retrieval_metrics_run: passed
stage99_final_test_metrics_not_run: passed
stage99_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_dev_contract_deltas.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_gate_actions.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_second_wave_route_status.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_stop_decision_flags.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_stop_guard_check_status.svg
```

Stage99 JSON SHA256:

```text
4BBAECBC7C6AFD45564E02829B142BA1A733316736CA92F4CB7B95C0450800E3
```

Visualization SHA256:

```text
stage99_second_wave_route_status.svg: E127E7CE92C6B6119467413E833DEC21FCA36DBA33858C4111CC8C9B7DDFC86B
stage99_selective_dense_sparse_dev_contract_deltas.svg: 35C35A8F511ED56EBE40EB2BA3B3550C71C244418C7C65A3E9A4810A14339910
stage99_selective_dense_sparse_gate_actions.svg: E11B623B6F8AE7DB8D0AB99EDEA6C2E4BC6EC9443B848FC9095392D52431641E
stage99_selective_dense_sparse_stop_decision_flags.svg: 176E059FF6610EF77EA5E5BA53B9D473C5A6E541EDA1493C6FD27E9A291C4FEA
stage99_selective_dense_sparse_stop_guard_check_status.svg: D8D38E26E551337AEE5DED500397384BC1393FDD3F84F8830C40A00DA98B4C24
stage99_selective_dense_sparse_train_dev_hit10_delta.svg: D1B81D1CADFAF22A026DAC6C507A347DB191A6D766665AF673A7A537D18DCF3E
```

## Validation

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_selective_dense_sparse_stop_decision.py scripts\decide_primeqa_hybrid_selective_dense_sparse_stop.py tests\test_primeqa_hybrid_selective_dense_sparse_stop_decision.py
pytest -q tests\test_primeqa_hybrid_selective_dense_sparse_stop_decision.py
python scripts\decide_primeqa_hybrid_selective_dense_sparse_stop.py ...: exit code 0
```

Result:

```text
ruff: passed
pytest: 3 passed
Stage99 run: passed
```

Full validation:

```text
ruff check .: passed
pytest -q: 263 passed
git diff --check: passed
```

Artifact safety note:

```text
Stage99 JSON contains prohibited feature names only inside policy text and safety flags.
Actual raw/private fixture snippets scan: no matches.
PrimeQA raw row fields such as QUESTION_TEXT/QUESTION_TITLE/ANSWERABLE/START_OFFSET/END_OFFSET: no matches.
```

## Conclusion

- Stage99 stopped `selective_dense_sparse_low_overlap_gate_design`.
- Stage99 did not load split files and did not run retrieval metrics.
- Stage99 did not run final metrics.
- Stage99 did not tune dev thresholds.
- Stage99 did not change runtime defaults.
- Stage84 second-wave actionable retrieval candidates are exhausted.
- The next stage should summarize second-wave route exhaustion and decide the
  next research direction from existing train/dev evidence while keeping test
  locked.
