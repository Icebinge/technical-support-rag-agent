# PrimeQA Hybrid Second-Wave Route Exhaustion Summary

This document records Stage 100.

## Scope

Stage 100 summarizes the exhausted Stage84 second-wave retrieval routes and
selects the next research direction from saved train/dev evidence.

This stage reads only public-safe Stage83, Stage84, Stage87, Stage90, Stage93,
Stage96, and Stage99 reports. It does not load train/dev/test split files, does
not run new retrieval metrics, does not run final metrics, does not use source
`DOC_IDS` as runtime retrieval evidence, does not tune dev thresholds, and does
not change runtime defaults.

The summary was explicitly confirmed for this run:

```text
route_id: second_wave_retrieval_route_exhaustion_summary
confirmed: true
confirmation_note: user confirmed Stage100 second wave route exhaustion summary in current turn
```

## Command

```text
python scripts\summarize_primeqa_hybrid_second_wave_route_exhaustion.py ^
  --user-confirmed-summary ^
  --confirmation-note "user confirmed Stage100 second wave route exhaustion summary in current turn" ^
  --output artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100.json ^
  --visualization-dir artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals
```

The run completed successfully and wrote console output to:

```text
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100.console.txt
```

## Route Outcomes

| Candidate | Stop stage | Selected ID | Train hit@10 delta | Dev hit@10 delta | Dev top10 net | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| lexical_cluster_diversity_rerank_design | Stage 87 | lcdr_penalty_0_06_title_query_cluster | +0.0054 | +0.0000 | 0 | stopped |
| structured_query_keyphrase_compaction_design | Stage 90 | sqkc_title_guarded_action_error_v1 | -0.0027 | -0.0527 | -4 | stopped |
| section_signal_guarded_expansion_design | Stage 93 | ssgx_section_top50_injection_guard_v1 | +0.0000 | +0.0000 | 0 | stopped |
| score_margin_bm25_normalization_gate_design | Stage 96 | smbn_rank11_20_long_doc_b095_margin_v1 | +0.0000 | +0.0000 | 0 | stopped |
| selective_dense_sparse_low_overlap_gate_design | Stage 99 | sdsl_minilm_low_overlap_conservative_v1 | +0.0000 | +0.0000 | 0 | stopped |

Aggregate summary:

```text
first_wave_retrieval_candidates_exhausted: true
second_wave_expected_candidate_count: 5
second_wave_stopped_candidate_count: 5
second_wave_all_expected_candidates_stopped: true
runtime_advancing_second_wave_candidate_count: 0
best_second_wave_dev_hit10_delta: 0.0000
best_second_wave_top10_net: 0
stage99_route_family_exhausted: true
remaining_actionable_candidate_count: 0
blocked_source_doc_ids_diagnostic_status: blocked_from_train_dev_experiment
second_wave_retrieval_route_family_exhausted: true
```

The blocked diagnostic remains blocked:

```text
candidate_id: source_doc_ids_oracle_union_blocked
status: blocked_from_train_dev_experiment
eligible_for_train_dev_experiment: false
eligible_for_runtime_defaultization: false
```

## Next Direction Options

| Option | Recommended | Readiness | Reason |
| --- | --- | ---: | --- |
| answer_pipeline_error_decomposition | yes | 0.92 | Retrieval-route invention is exhausted; decompose remaining failures into retrieval, evidence selection, citation, and answer composition buckets before another intervention. |
| third_wave_retrieval_design | no | 0.42 | Needs a new deployable diagnostic signal not already covered by Stage76 or Stage84. |
| final_test_gate_review | no | 0.00 | No retrieval route produced a train-selected dev contract pass or runtime-default candidate. |
| source_doc_ids_oracle_union | no | 0.00 | Source `DOC_IDS` are dataset metadata, not runtime retrieval evidence. |

## Decision

```text
status: primeqa_hybrid_second_wave_route_exhaustion_summary_completed
first_wave_retrieval_candidates_exhausted: true
second_wave_retrieval_route_family_exhausted: true
runtime_advancing_second_wave_candidate_count: 0
remaining_actionable_candidate_count: 0
recommended_next_direction: answer_pipeline_error_decomposition
requires_user_confirmation_before_next_protocol: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
recommended_next_stage: Stage101: design a train/dev-only answer-pipeline error decomposition protocol from existing public-safe artifacts.
```

Stage100 does not start Stage101 and does not run answer-pipeline metrics. It
only records that the next research direction should move away from retrieval
candidate invention and toward a public-safe answer-pipeline error
decomposition protocol.

## Guard Checks

All `17 / 17` guard checks passed:

```text
source_reports_are_expected_stages: passed
user_confirmed_stage100_summary: passed
stage83_first_wave_exhausted: passed
stage84_second_wave_order_matches_expected: passed
all_second_wave_candidates_have_stop_reports: passed
no_second_wave_candidate_advanced_to_runtime: passed
best_second_wave_dev_hit10_delta_not_positive: passed
stage99_route_family_exhausted: passed
no_remaining_actionable_retrieval_candidates: passed
source_doc_ids_diagnostic_remains_blocked: passed
all_stop_reports_have_passing_guards: passed
all_source_decisions_keep_final_test_locked: passed
all_source_decisions_forbid_test_tuning: passed
all_source_decisions_keep_runtime_defaults_unchanged: passed
stage100_runs_summary_only_no_new_retrieval_metrics: passed
stage100_final_test_metrics_not_run: passed
stage100_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_second_wave_dev_hit10_deltas.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_second_wave_top10_net_changes.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_second_wave_route_outcomes.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_next_direction_readiness.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_route_exhaustion_decision_flags.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_route_exhaustion_guard_check_status.svg
```

Stage100 JSON SHA256:

```text
664E88514DDE230942642243DC901EE1A8685DCD3E1FCE46CA88386E45777CA3
```

Visualization SHA256:

```text
stage100_next_direction_readiness.svg: C880D069A6D57AD39B2AC2651F182A7807E09722144EEFA98164345CC5D13F98
stage100_route_exhaustion_decision_flags.svg: 75AD7FA89F4B8AF7C440727DEC4E14ACA07F4A8CAD666662805344B3656A6195
stage100_route_exhaustion_guard_check_status.svg: 831E5491B8D02A9F74C50186D6F4D5DFBC11974F12AAC0FA5F6AE68EF2A573EB
stage100_second_wave_dev_hit10_deltas.svg: 00B2D0A78D346D7DCC1EDF924861F4A25C9D24C79FF484AEBAE3EAB17A792993
stage100_second_wave_route_outcomes.svg: 274E97DE700E61DED8E5F49B1DAA2EB4108B5A608A7554023AED3E7276D75CE1
stage100_second_wave_top10_net_changes.svg: 571F0B0646CCEEDB10D1E9F7708CF071150A9B2BB6C112D310C11DEA4BCBF39B
```

## Validation

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_second_wave_route_exhaustion_summary.py scripts\summarize_primeqa_hybrid_second_wave_route_exhaustion.py tests\test_primeqa_hybrid_second_wave_route_exhaustion_summary.py
pytest -q tests\test_primeqa_hybrid_second_wave_route_exhaustion_summary.py
python scripts\summarize_primeqa_hybrid_second_wave_route_exhaustion.py ...: exit code 0
```

Result:

```text
ruff: passed
pytest: 3 passed
Stage100 run: passed
guard checks: 17 / 17 passed
```

Full validation:

```text
ruff check .: passed
pytest -q: 266 passed
git diff --check: passed
```

Artifact safety scan:

```text
Select-String Stage100 JSON for private fixture snippets and PrimeQA raw row fields: no matches
```

## Conclusion

- Stage100 completed the second-wave route exhaustion summary.
- Stage100 did not load split files and did not run retrieval metrics.
- Stage100 did not run final metrics.
- Stage100 did not tune dev thresholds.
- Stage100 did not change runtime defaults.
- Both first-wave and second-wave retrieval-route families are exhausted.
- The next recommended direction is `answer_pipeline_error_decomposition`.
- Stage101 should design that protocol from existing public-safe train/dev
  evidence while keeping test locked.
