# PrimeQA Hybrid Evidence/Answerability Candidate Protocol

This document records Stage 103.

## Scope

Stage 103 freezes a train/dev-only candidate protocol for the shared
answerability and evidence-selection bottlenecks found in Stage102.

This stage reads only the saved public-safe Stage102 aggregate report. It does
not load train/dev/test split files, does not load corpus documents, does not
run retrieval metrics, does not run answer metrics, does not run final metrics,
does not use oracle document identifiers or gold answers as runtime evidence,
does not add fallback strategies, and does not change runtime defaults.

The protocol was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: Stage103 user-confirmed design freeze after Stage102 decomposition
```

## Command

```text
python scripts\design_primeqa_hybrid_evidence_answerability_candidates.py ^
  --user-confirmed-protocol ^
  --confirmation-note "Stage103 user-confirmed design freeze after Stage102 decomposition"
```

The run completed successfully:

```text
stage103_exit_code=0
```

Console output was written to:

```text
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103.console.txt
```

## Stage102 Premise

```text
decision_status: primeqa_hybrid_answer_pipeline_error_decomposition_completed
analysis_id: answer_pipeline_error_decomposition_train_dev_analysis_v1
recommended_next_direction: evidence_selection_and_answerability_candidate_design
train_top_bucket: answerability_false_answer
dev_top_bucket: answerability_false_answer
train_verified_average_token_f1: 0.2017
dev_verified_average_token_f1: 0.2040
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Bottlenecks

| Bucket | Role | Train count | Train rate | Dev count | Dev rate | Combined count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| answerability_false_answer | primary_candidate_target | 180 | 0.3203 | 41 | 0.3388 | 221 |
| gold_span_beats_selected_answer | primary_candidate_target | 174 | 0.3096 | 41 | 0.3388 | 215 |
| retrieval_context_miss | secondary_retrieval_context_not_a_new_route | 125 | 0.2224 | 23 | 0.1901 | 148 |
| evidence_selection_miss | secondary_evidence_context | 67 | 0.1192 | 12 | 0.0992 | 79 |

Retrieval-context misses are tracked as secondary context only. Stage103 does
not reopen a new retrieval-route family because earlier retrieval routes were
already stopped as non-actionable.

## Frozen Candidate Protocol

```text
design_id: evidence_selection_and_answerability_candidate_design_v1
protocol_status: frozen_requires_user_confirmation_before_train_dev_run
design_mode: protocol_freeze_only
recommended_direction: evidence_answerability_train_dev_candidate_comparison
fallback_strategies_enabled: false
```

Candidate policies:

| Candidate | Focus | Target buckets | Target cases | Priority score | Risk |
| --- | --- | --- | ---: | ---: | --- |
| answerability_margin_gate_candidate_v1 | answerability_pre_generation_gate | answerability_false_answer | 221 | 271.2996 | medium |
| evidence_window_reselector_candidate_v1 | evidence_selection_and_composition_ordering | gold_span_beats_selected_answer, evidence_selection_miss | 294 | 313.1271 | medium |
| joint_gate_then_window_candidate_v1 | answerability_gate_plus_evidence_window_ordering | answerability_false_answer, gold_span_beats_selected_answer, evidence_selection_miss | 515 | 366.6990 | medium_high |

Recommended Stage104 execution order:

```text
1. joint_gate_then_window_candidate_v1
2. evidence_window_reselector_candidate_v1
3. answerability_margin_gate_candidate_v1
```

This order is a design-priority order, not a runtime default and not a final
promotion decision.

## Runtime Feature Boundary

Allowed runtime feature groups:

```text
question_observables
retrieval_observables
evidence_observables
composition_observables
```

Prohibited runtime inputs:

```text
gold answers
gold spans
answer document identifiers
source DOC_IDS
test split labels
dev-selected thresholds
raw private question or answer strings written to reports
```

Blocked items:

```text
source_doc_id_oracle_candidate_blocked
gold_span_oracle_selector_blocked
test_tuned_threshold_candidate_blocked
```

## Stage104 Contract

Stage104 may run only after user confirmation. Its selection contract is:

```text
candidate_thresholds_selected_on: train_only
dev_threshold_tuning_allowed: false
test_access_allowed: false
dev_used_for: single validation of train-selected candidate
dev_retuning_allowed: false
runtime_default_change_allowed_in_stage104: false
final_test_gate_remains_closed: true
```

Primary metrics:

```text
answerability_false_answer count/rate
gold_span_beats_selected_answer count/rate
combined target priority score
```

Secondary and guard metrics:

```text
evidence_selection_miss count/rate
verified average token F1
verified gold document citation rate
answerable refusal rate
unanswerable refusal rate
changed answer count
public-safe changed-case bucket summary
```

## Decision

```text
status: primeqa_hybrid_evidence_answerability_candidate_protocol_frozen
design_id: evidence_selection_and_answerability_candidate_design_v1
recommended_direction: evidence_answerability_train_dev_candidate_comparison
requires_user_confirmation_before_train_dev_run: true
can_continue_train_dev_development: true
can_run_train_dev_candidate_comparison_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_stage: Stage104: after user confirmation, run the frozen train/dev-only evidence-answerability candidate comparison against the Stage102 verified baseline; keep test locked, do not retune on dev, and do not change runtime defaults.
```

## Guard Checks

All `31 / 31` guard checks passed:

```text
stage102_source_is_expected_stage: passed
user_confirmed_stage103_protocol: passed
stage102_analysis_id_is_expected: passed
stage102_analysis_completed: passed
stage102_recommends_evidence_answerability_design: passed
stage102_train_top_bucket_is_answerability_false_answer: passed
stage102_dev_top_bucket_is_answerability_false_answer: passed
shared_answerability_false_answer_observed: passed
shared_gold_span_gap_observed: passed
shared_evidence_selection_miss_observed: passed
retrieval_context_miss_is_secondary_context_only: passed
stage102_all_guard_checks_passed: passed
stage102_can_continue_train_dev: passed
stage102_final_test_gate_locked: passed
stage102_test_not_available_for_tuning: passed
stage102_fallback_disabled: passed
stage102_runtime_defaults_unchanged: passed
stage103_protocol_status_frozen: passed
stage103_candidate_count_is_three: passed
stage103_candidate_ids_are_unique: passed
stage103_candidate_execution_order_complete: passed
stage103_candidates_are_train_dev_protocol_candidates: passed
stage103_candidate_runtime_features_are_declared: passed
stage103_candidates_do_not_use_forbidden_runtime_inputs: passed
stage103_oracle_items_are_blocked: passed
stage104_train_selection_forbids_dev_threshold_tuning: passed
stage104_dev_validation_forbids_retuning: passed
stage104_test_access_forbidden: passed
stage103_public_safe_output_contract_has_no_forbidden_fields: passed
stage103_exclusions_lock_test_runtime_fallback: passed
stage103_fallback_policy_disabled: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_shared_bottleneck_counts.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_shared_bottleneck_rates.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_candidate_priority_scores.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_candidate_target_case_counts.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_candidate_feature_group_counts.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_protocol_decision_flags.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_guard_check_status.svg
```

Stage103 JSON SHA256:

```text
DA15C416DAA79BE2EBC25749923DB51B5B648894D01DAB95FF500ECEF9D97062
```

Visualization SHA256:

```text
stage103_candidate_feature_group_counts.svg: 7C4E690F32BC6766E9BBF3179A649A38DB451753088DD255FA03B14F779F07E0
stage103_candidate_priority_scores.svg: 24C6C3C4B6AD620B8700B4896524B5047C695248685C52DBF573BAEA0C278797
stage103_candidate_target_case_counts.svg: F69D77ED5C537C7CA804F1CB782D004F3D4B99CE6C36500D7F08AEE35A658ACB
stage103_guard_check_status.svg: 77BA0695B9C98A895DB934DEE6DDC36B3BEE1E204770341118CD345CE884637F
stage103_protocol_decision_flags.svg: 40036E546FB9A03D6482C9A17DFEEDA700D18C6283305463B7D7E32564E471DA
stage103_shared_bottleneck_counts.svg: 07B8C53DA25DC01C1DF604C7E72FD51BB705D9C0DE8866E0A124DB93F26C42F4
stage103_shared_bottleneck_rates.svg: 6ED2660487FF300A83DDE53A5E1DF131422204A79E23981477B10C627482E424
```

## Validation

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_candidate_protocol.py scripts\design_primeqa_hybrid_evidence_answerability_candidates.py tests\test_primeqa_hybrid_evidence_answerability_candidate_protocol.py
pytest -q tests\test_primeqa_hybrid_evidence_answerability_candidate_protocol.py
python scripts\design_primeqa_hybrid_evidence_answerability_candidates.py ...: stage103_exit_code=0
```

Result:

```text
ruff: passed
pytest: 5 passed
Stage103 run: passed
guard checks: 31 / 31 passed
```

Full validation:

```text
ruff check .: passed
pytest -q: 279 passed
git diff --check: passed
```

Artifact safety checks:

```text
Allowed output fields intersect forbidden fields: []
Source file recorded by Stage103: artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json
No split-file or corpus-document paths are recorded in Stage103 source_files.
```

The string scan for forbidden field names finds expected matches only inside
the explicit `forbidden_fields` and `explicit_exclusions` lists, not in allowed
case or aggregate output fields and not as raw private values.

```text
question_text / raw_answer_text / answer_doc_id / source_doc_ids patterns:
expected matches in forbidden/exclusion lists only
```

## Conclusion

Stage103 can advance to Stage104 after user confirmation. The next stage should
run the frozen train/dev-only evidence-answerability candidate comparison
against the Stage102 verified baseline. The test split remains locked, dev
threshold tuning is forbidden, fallback strategies remain disabled, and runtime
defaults remain unchanged.
