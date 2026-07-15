# PrimeQA Hybrid Failure-Pattern Redesign Stop Decision

This document records Stage 110.

## Scope

Stage 110 stops the frozen Stage108 failure-pattern redesign family after the
Stage109 train grouped-CV/dev validation comparison selected no config.

This is a decision checkpoint. It reads only the public-safe Stage109 report.
It does not load train/dev/test split files, does not load corpus documents,
does not run retrieval or answer metrics, does not run final metrics, does not
select from dev-only observations, does not add fallback strategies, and does
not change runtime defaults.

The stop decision was explicitly confirmed for this run:

```text
route_id: failure_pattern_redesign_stop_decision
confirmed: true
confirmation_note: user confirmed Stage110 failure-pattern redesign stop decision on 2026-07-16 after Stage109 selected no train-CV config; test locked; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\decide_primeqa_hybrid_failure_pattern_redesign_stop.py ^
  --user-confirmed-stop ^
  --confirmation-note "user confirmed Stage110 failure-pattern redesign stop decision on 2026-07-16 after Stage109 selected no train-CV config; test locked; runtime defaults unchanged; no fallback strategies"
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110.json
```

The run completed successfully in `0.002s` according to the Stage110 timing
block.

## Evidence Used

Stage109 selected no config:

```text
stage109_status: primeqa_hybrid_failure_pattern_redesign_completed_no_train_cv_selectable_config
selection_split: train
selection_mode: train_grouped_cross_validation_then_full_train_refit
selected_config_id: null
selectable_config_count: 0 / 7
dev_validation_status: no_train_cv_selectable_config
dev_validation_passed: false
stage109_guard_checks: 20 / 20 passed
```

Stage108 frozen selection rules still apply:

```text
requires_negative_train_cv_weighted_delta: true
no_op_candidate_selectable: false
max_train_cv_answerable_refusal_rate_delta: 0.02
max_train_cv_average_token_f1_drop: 0.005
max_train_cv_gold_doc_citation_rate_drop: 0.015
max_train_cv_retrieval_context_miss_delta: 0
dev_selection_allowed: false
dev_retuning_allowed: false
dev_threshold_tuning_allowed: false
test_access_allowed: false
```

## Family Summary

| Family | Configs | Train-CV selectable | Best train-CV delta | Best dev delta | Max answerable refusal delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint_support_gate_span_composer_candidate_v1 | 2 | 0 | -199.65 | -48.25 | +0.3784 |
| context_present_span_composer_candidate_v1 | 3 | 0 | -123.05 | -47.15 | +0.3027 |
| support_aware_answerability_gate_candidate_v1 | 2 | 0 | -111.15 | -24.75 | +0.1919 |

The target-score improvements are therefore real but blocked. They trade too
much answerable coverage for lower failure-bucket counts.

## Dev-Improved But Non-Selectable Configs

The following configs improved dev weighted target score, but every one was
train-CV nonselectable. They cannot be selected from dev.

| Config | Family | Dev delta | Train-CV delta | Failed train-CV guards |
| --- | --- | ---: | ---: | --- |
| jsgc_support2_evidence7_anchor_top2_v1 | joint_support_gate_span_composer_candidate_v1 | -48.25 | -199.65 | answerable_refusal_rate_delta |
| cpsc_title_query_anchor_top2_mcpd3_rank3_v1 | context_present_span_composer_candidate_v1 | -47.15 | -123.05 | answerable_refusal_rate_delta |
| jsgc_support2_evidence6_title_anchor_top2_v1 | joint_support_gate_span_composer_candidate_v1 | -34.95 | -115.90 | answerable_refusal_rate_delta |
| saag_support2_evidence7_rank3_v1 | support_aware_answerability_gate_candidate_v1 | -24.75 | -111.15 | answerable_refusal_rate_delta |
| saag_support2_evidence6_rank5_v1 | support_aware_answerability_gate_candidate_v1 | -1.75 | -31.65 | answerable_refusal_rate_delta |
| cpsc_anchor_top2_mcpd3_rank3_v1 | context_present_span_composer_candidate_v1 | -0.30 | -28.50 | answerable_refusal_rate_delta, average_token_f1_drop, gold_doc_citation_rate_drop |

## No-Op Block

One config did not change the baseline behavior and was blocked by the frozen
no-op rule:

```text
config_id: cpsc_anchor_top3_mcpd3_rank3_v1
train_cv_weighted_target_delta: 0.0
train_cv_changed_answer_count: 0
failed_train_cv_guard: train_cv_weighted_target_delta_negative
```

## Decision

```text
status: primeqa_hybrid_failure_pattern_redesign_family_stopped
stopped_family_id: failure_pattern_redesign_candidate_family
stopped_protocol_id: primeqa_hybrid_failure_pattern_redesign_protocol_v1
stopped_analysis_id: primeqa_hybrid_failure_pattern_redesign_train_cv_dev_validation_v1
current_route_defaultization: blocked
redesign_required_before_any_runtime_or_test_gate: true
recommended_next_direction: user_confirmed_next_research_direction_required
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Stage110 therefore provides no justification for runtime defaultization or
final-test gate opening.

## Guard Checks

All `27 / 27` guard checks passed.

Key checks:

```text
source_stage109_report_is_stage109: passed
user_confirmed_stage110_stop_decision: passed
stage109_completed_with_no_train_cv_selectable_config: passed
stage109_recommends_stop_decision: passed
stage109_all_guard_checks_passed: passed
stage109_split_contract_is_train_dev_only: passed
stage108_train_cv_selection_rule_frozen: passed
stage108_dev_selection_forbidden: passed
stage109_selected_no_config: passed
all_stage109_configs_are_train_cv_nonselectable: passed
stage109_negative_deltas_are_blocked_not_selected: passed
stage109_noop_config_blocked: passed
stage110_split_files_not_loaded: passed
stage110_corpus_documents_not_loaded: passed
stage110_final_test_metrics_not_run: passed
stage110_default_runtime_policy_unchanged: passed
stage110_fallback_strategies_not_added: passed
stage110_public_outputs_have_no_forbidden_keys: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_train_cv_weighted_target_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_dev_weighted_target_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_answerable_refusal_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_selectability_by_family.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_train_cv_guard_failure_reasons.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_stop_decision_flags.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_stop_guard_check_status.svg
```

## Validation

Targeted validation:

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_stop_decision.py scripts\decide_primeqa_hybrid_failure_pattern_redesign_stop.py tests\test_primeqa_hybrid_failure_pattern_redesign_stop_decision.py
python -m pytest tests\test_primeqa_hybrid_failure_pattern_redesign_stop_decision.py -q
python scripts\decide_primeqa_hybrid_failure_pattern_redesign_stop.py --user-confirmed-stop ...
```

Result:

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage110 run: passed
guard checks: 27 / 27 passed
full ruff: passed
full pytest: 300 passed
git diff --check: passed
```

Full validation is recorded in `docs\learning_journal.md`.

## Next Step

Stage111 should happen only after user confirmation. It should choose the next
train/dev-only research direction before any new protocol or experiment. It
must not select from dev-only observations, must keep test locked, must keep
runtime defaults unchanged, and must not add fallback strategies.
