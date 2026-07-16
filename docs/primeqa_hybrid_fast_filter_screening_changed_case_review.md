# PrimeQA Hybrid Fast-Filter Screening Changed-Case Review

Stage: Stage 122

Status: completed

Artifact:

```text
artifacts\primeqa_hybrid_fast_filter_screening_changed_case_review_stage122.json
```

## Scope

Stage122 reviews changed cases for the Stage121 selected safe config and the
stronger but guard-blocked logistic config:

```text
selected safe config:
  special_token_exact_window40_rule_selector_v1

blocked signal config:
  top10_locked_route_vote_window50_pairwise_logistic_v1
```

This review uses only train grouped cross-validation and dev report-only data.
The final test split remains locked. No final test metrics were run, no runtime
defaults were changed, and no fallback strategies were added.

## Command

```text
python scripts\review_primeqa_hybrid_fast_filter_screening_changed_cases.py --user-confirmed-review --confirmation-note "user confirmed Stage122 changed-case review after Stage121 selected special-token config and blocked logistic hit@20 signal; train/dev only; test locked; dev report-only; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Result

```text
status: primeqa_hybrid_fast_filter_screening_changed_case_review_completed
recommended_next_direction: design_first_stage_recall_expansion_protocol
can_continue_train_dev_development: true
runtime_defaultization_supported: false
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Changed-Case Summary

```text
special_token_exact_window40_rule_selector_v1:
  interpretation: safe_but_weak
  train-CV changed cases: 40
  train-CV improved/regressed: 4 / 36
  train-CV hit@20 recoveries/regressions: 4 / 3
  dev changed cases: 8
  dev improved/regressed: 0 / 8
  dev hit@20 recoveries/regressions: 0 / 0

top10_locked_route_vote_window50_pairwise_logistic_v1:
  interpretation: positive_signal_but_guard_risky
  train-CV changed cases: 40
  train-CV improved/regressed: 11 / 29
  train-CV hit@20 recoveries/regressions: 11 / 7
  dev changed cases: 8
  dev improved/regressed: 2 / 6
  dev hit@20 recoveries/regressions: 2 / 1
```

## Cross-Config Findings

```text
blocked_signal_has_real_hit20_recoveries: true
blocked_signal_has_guard_relevant_regressions: true
selected_config_is_low_change: false
```

The blocked logistic config is not a false signal. It recovers additional
hit@20 cases on both train-CV and dev. However, it also creates hit@20
regressions, including one on dev, so the Stage121 guard block remains
justified.

The selected special-token config remains conservative but weak. It has no dev
hit@20 recoveries and all dev changed cases are rank regressions that do not
change hit@20.

## Guard Checks

```text
guard checks: 14 / 14 passed
test_split_loaded: false
dev_report_only: true
runtime_defaults_unchanged: true
fallback_strategies_added: false
public_safe_contract.forbidden_keys_found: []
raw_candidate_rows_written: false
```

The report uses anonymized `case_hash` values for changed-case samples. It does
not write raw question text, raw answer text, raw document text, raw document
IDs, raw sample IDs, or raw candidate rows.

## Visualizations

```text
artifacts\primeqa_hybrid_fast_filter_screening_changed_case_review_stage122_visuals\stage122_changed_case_outcomes.svg
artifacts\primeqa_hybrid_fast_filter_screening_changed_case_review_stage122_visuals\stage122_hit20_transitions.svg
artifacts\primeqa_hybrid_fast_filter_screening_changed_case_review_stage122_visuals\stage122_changed_case_rank_delta.svg
artifacts\primeqa_hybrid_fast_filter_screening_changed_case_review_stage122_visuals\stage122_guard_risk_summary.svg
artifacts\primeqa_hybrid_fast_filter_screening_changed_case_review_stage122_visuals\stage122_decision_flags.svg
artifacts\primeqa_hybrid_fast_filter_screening_changed_case_review_stage122_visuals\stage122_guard_check_status.svg
```

## Interpretation

Stage122 confirms that second-stage fast screening is not the right place to
seek large recall gains. The stronger logistic screener has useful recall
signal, but it pays for that signal with guard-relevant regressions. The safe
selected screener does not materially improve dev recall.

Stage123 froze the first-stage recall expansion protocol. The result is
recorded in:

```text
docs/primeqa_hybrid_first_stage_recall_expansion_protocol.md
```

The target is a broader, simple, fast candidate generator that raises
candidate-pool recall before any precise second-stage selection. Test remains
locked.
