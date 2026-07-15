# PrimeQA Hybrid Evidence/Answerability Stop Decision

This document records Stage 106.

## Scope

Stage 106 stops the Stage103/104 evidence-answerability candidate family after
the Stage105 train/dev comparison failed dev validation.

This is a decision checkpoint. It reads the public-safe Stage104 and Stage105
reports. It does not load train/dev/test split files, does not run retrieval or
answer metrics, does not run final metrics, does not select from dev-only
observations, does not add fallback strategies, and does not change runtime
defaults.

The stop decision was explicitly confirmed for this run:

```text
route_id: evidence_answerability_stop_decision
confirmed: true
confirmation_note: user confirmed Stage106 evidence-answerability stop decision on 2026-07-15 after Stage105 dev validation failed; test locked; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\decide_primeqa_hybrid_evidence_answerability_stop.py ^
  --user-confirmed-stop ^
  --confirmation-note "user confirmed Stage106 evidence-answerability stop decision on 2026-07-15 after Stage105 dev validation failed; test locked; runtime defaults unchanged; no fallback strategies"
```

The run completed successfully and wrote console output to:

```text
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106.console.txt
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106.json
```

## Evidence Used

Stage105 selected this config on train:

```text
selected_config_id: amg_bm25_evidence8_rank3_v1
selected_candidate_id: answerability_margin_gate_candidate_v1
selectable_config_count: 2 / 9
selected_train_weighted_target_delta: 0.0
```

Dev validation for the train-selected config:

```text
dev_validation_passed: false
dev_weighted_target_delta: 0.0
dev_changed_answer_count: 0
answerability_false_answer delta: 0
gold_span_beats_selected_answer delta: 0
evidence_selection_miss delta: 0
answerable_refusal_rate delta: 0.0
gold_doc_citation_rate delta: 0.0
average_token_f1 delta: 0.0
```

Stage105 had `25 / 25` guard checks passed. Stage106 records this exact count;
an earlier manual note had incorrectly summarized it as `29 / 29`, and the
project docs were corrected during Stage106.

## Dev-Better But Non-Selectable Configs

The following configs had better dev weighted target deltas than the
train-selected config, but all were train-nonselectable under the frozen
Stage104 guards. They cannot be selected from dev.

| Config | Candidate | Dev delta | Train delta | Failed train guards |
| --- | --- | ---: | ---: | --- |
| jgw_answer_window_mcpd5_evidence8_rank2_v1 | joint_gate_then_window_candidate_v1 | -29.15 | -133.90 | answerable refusal, gold citation |
| ewr_answer_window_mcpd3_evidence7_rank3_v1 | evidence_window_reselector_candidate_v1 | -17.05 | -89.25 | answerable refusal, gold citation |
| ewr_answer_window_mcpd5_evidence7_rank3_v1 | evidence_window_reselector_candidate_v1 | -17.05 | -89.25 | answerable refusal, gold citation |
| jgw_answer_window_mcpd3_evidence8_rank3_v1 | joint_gate_then_window_candidate_v1 | -17.05 | -89.25 | answerable refusal, gold citation |
| ewr_hybrid_window_mcpd3_evidence7_rank3_v1 | evidence_window_reselector_candidate_v1 | -3.90 | -16.60 | answerable refusal |
| jgw_hybrid_window_mcpd3_evidence8_rank3_v1 | joint_gate_then_window_candidate_v1 | -3.90 | -16.60 | answerable refusal |
| amg_bm25_evidence8_rank2_v1 | answerability_margin_gate_candidate_v1 | -3.10 | -29.25 | answerable refusal |

## Family Summary

```text
answerability_margin_gate_candidate_v1:
  configs: 3
  train_selectable: 2
  best_train_delta: -29.25 via amg_bm25_evidence8_rank2_v1
  best_dev_delta: -3.10 via amg_bm25_evidence8_rank2_v1
  train_guard_failures:
    answerable_refusal_rate_delta_within_guard: 1

evidence_window_reselector_candidate_v1:
  configs: 3
  train_selectable: 0
  best_train_delta: -89.25 via ewr_answer_window_mcpd3_evidence7_rank3_v1
  best_dev_delta: -17.05 via ewr_answer_window_mcpd3_evidence7_rank3_v1
  train_guard_failures:
    answerable_refusal_rate_delta_within_guard: 3
    gold_doc_citation_rate_drop_within_guard: 2

joint_gate_then_window_candidate_v1:
  configs: 3
  train_selectable: 0
  best_train_delta: -133.90 via jgw_answer_window_mcpd5_evidence8_rank2_v1
  best_dev_delta: -29.15 via jgw_answer_window_mcpd5_evidence8_rank2_v1
  train_guard_failures:
    answerable_refusal_rate_delta_within_guard: 3
    gold_doc_citation_rate_drop_within_guard: 2
```

## Decision

```text
status: primeqa_hybrid_evidence_answerability_candidate_family_stopped
stopped_family_id: evidence_answerability_candidate_family
stopped_protocol_id: evidence_answerability_candidate_train_dev_comparison_v1
current_route_defaultization: blocked
redesign_required_before_any_runtime_or_test_gate: true
recommended_next_direction: evidence_answerability_redesign_decision
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

The family is stopped because the train-selected Stage105 config did not
improve the train weighted target objective and failed dev validation with zero
dev weighted target improvement and zero dev changed answers. The configs with
better dev target deltas were not train selectable under the frozen Stage104
guards, so selecting them from dev would violate the protocol.

Stage106 therefore provides no justification for runtime defaultization or
final-test gate opening.

## Guard Checks

All `30 / 30` guard checks passed:

```text
source_stage104_report_is_stage104: passed
source_stage105_report_is_stage105: passed
user_confirmed_stage106_stop_decision: passed
stage104_protocol_is_frozen: passed
stage104_protocol_id_matches: passed
stage104_candidate_grid_has_nine_configs: passed
stage104_train_selection_is_train_only: passed
stage104_dev_validation_forbids_dev_selection: passed
stage104_final_test_gate_locked: passed
stage104_runtime_defaults_unchanged: passed
stage104_fallback_disabled: passed
stage105_analysis_id_matches: passed
stage105_completed_with_dev_guard_failed: passed
stage105_recommends_stop_decision: passed
stage105_all_guard_checks_passed: passed
stage105_selection_uses_train_split: passed
stage105_selected_config_did_not_improve_train_target: passed
stage105_selected_config_failed_dev_validation: passed
stage105_selected_config_did_not_improve_dev_target: passed
stage105_selected_config_changed_no_dev_answers: passed
stage105_dev_better_configs_are_train_nonselectable: passed
stage105_final_test_gate_locked: passed
stage105_forbids_test_tuning: passed
stage105_default_runtime_policy_unchanged: passed
stage105_fallback_strategies_disabled: passed
stage106_no_new_train_dev_metrics_run: passed
stage106_test_split_not_loaded: passed
stage106_final_test_metrics_not_run: passed
stage106_default_runtime_policy_unchanged: passed
stage106_fallback_strategies_not_added: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_evidence_answerability_target_deltas.svg
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_train_selectability_by_family.svg
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_train_guard_failure_reasons.svg
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_stop_decision_flags.svg
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_stop_guard_check_status.svg
```

## Validation

Targeted validation:

```text
ruff check --fix src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_stop_decision.py scripts\decide_primeqa_hybrid_evidence_answerability_stop.py tests\test_primeqa_hybrid_evidence_answerability_stop_decision.py
pytest -q tests\test_primeqa_hybrid_evidence_answerability_stop_decision.py
python scripts\decide_primeqa_hybrid_evidence_answerability_stop.py ...: exit code 0
```

Result:

```text
ruff: passed
pytest: 3 passed
Stage106 run: passed
```

Full validation is recorded in `docs\learning_journal.md`.

## Next Step

Stage107 should happen only after user confirmation. It should either freeze a
new train/dev-only evidence-answerability redesign protocol or explicitly move
to another research direction. Stage107 must not select from dev-only
observations, must keep test locked, must keep runtime defaults unchanged, and
must not add fallback strategies.
