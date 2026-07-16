# PrimeQA Hybrid Stage121 Fast-Filter Screening Validation

Stage121 ran the Stage120 fast-filter plus alternate-screening protocol on
train grouped cross-validation and dev report-only validation.

This stage is not a runtime defaultization step and does not run final test
metrics.

## Command

The final recorded run was a rerun after fixing a protected-prefix duplicate
segment bug:

```text
python scripts\run_primeqa_hybrid_fast_filter_screening_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage121 train-CV/dev fast-filter plus alternate-screening validation after Stage120 protocol freeze; rerun after protected-prefix duplicate segment fix; test locked; dev report-only; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Scope

```text
stage: Stage 121
analysis_id: primeqa_hybrid_fast_filter_screening_train_cv_dev_validation_v1
source protocol: primeqa_hybrid_fast_filter_screening_protocol_v1
candidate pool: fixed Stage116 top200
candidate configs: 6
```

The stage rebuilt train/dev candidate records in memory only:

```text
train candidate records in memory: 74,000
dev candidate records in memory: 15,200
raw candidate rows written: false
```

## Baseline

The Stage116 fixed top200 pool was reproduced:

```text
train hit@10: 255 / 370 = 0.6892
train hit@20: 279 / 370 = 0.7541
train hit@200: 345 / 370 = 0.9324
train mrr@20: 0.5062

dev hit@10: 55 / 76 = 0.7237
dev hit@20: 61 / 76 = 0.8026
dev hit@200: 69 / 76 = 0.9079
dev mrr@20: 0.5588
```

This confirms again that Stage121 is screening within a fixed pool; it cannot
recover the 25 train and 7 dev examples whose gold document is absent from the
top200 pool.

## Selected Train-CV Config

```text
selected_config_id: special_token_exact_window40_rule_selector_v1
selected_family_id: evidence_density_fast_filter_family_v1
guard_passed_config_count: 2 / 6
selectable_config_count: 1 / 6
status: train_cv_selected_positive_config
```

Train-CV comparison to baseline:

```text
hit@10 delta: +0.0000
hit@20 delta: +0.0027
hit@20 count delta: +1
hit@200 delta: +0.0000
mrr@20 delta: +0.0000
average promoted tail docs into top10: 0.0
```

Dev report-only comparison:

```text
hit@10 delta: +0.0000
hit@20 delta: +0.0000
hit@200 delta: +0.0000
mrr@20 delta: -0.0008
average promoted tail docs into top10: 0.0
```

Dev was not used for selection or retuning.

## Config Summary

```text
top10_locked_route_vote_window50_pairwise_logistic_v1:
  train-CV objective: +0.0261
  train hit@20 delta: +0.0108
  dev hit@20 delta: +0.0132
  blocked by hit@20 regression rate: 0.0189 > 0.01

top5_locked_strong_consensus_window80_pairwise_gbdt_v1:
  train-CV objective: -8.0391
  blocked by top10 regressions, hit@20 regression rate, and negative mrr@20

top20_locked_low_confidence_tail_screen_v1:
  train-CV objective: +0.0000
  passed guards but not selected because it has no positive objective

evidence_density_window40_pairwise_logistic_v1:
  train-CV objective: -8.0118
  blocked by top10 regressions and negative train hit@10/mrr@20

special_token_exact_window40_rule_selector_v1:
  train-CV objective: +0.0054
  selected

hybrid_filter_window80_pairwise_gbdt_v1:
  train-CV objective: -8.0425
  blocked by top10 regressions, hit@20 regression rate, and negative metrics
```

## Bug Fixed Before Final Run

An initial Stage121 run exposed a protected-prefix assembly bug:

```text
when protected_prefix_depth > 10, ranks 11-20 were appended again
```

This duplicated already-protected records in one config and produced an invalid
hit@200 loss. The bug was fixed before the final recorded run. The final
artifact and this document reflect the rerun after the fix.

## Guard Checks

```text
14 / 14 passed
public_safe_contract.forbidden_keys_found: []
test_split_loaded: false
final_test_metrics_run: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Visualizations

```text
artifacts\primeqa_hybrid_fast_filter_screening_validation_stage121_visuals\stage121_train_cv_objective_scores.svg
artifacts\primeqa_hybrid_fast_filter_screening_validation_stage121_visuals\stage121_train_cv_mrr_at_20_delta.svg
artifacts\primeqa_hybrid_fast_filter_screening_validation_stage121_visuals\stage121_train_cv_hit_at_10_delta.svg
artifacts\primeqa_hybrid_fast_filter_screening_validation_stage121_visuals\stage121_train_cv_guard_pass_counts.svg
artifacts\primeqa_hybrid_fast_filter_screening_validation_stage121_visuals\stage121_train_cv_top10_tail_promotions.svg
artifacts\primeqa_hybrid_fast_filter_screening_validation_stage121_visuals\stage121_dev_selected_config_deltas.svg
artifacts\primeqa_hybrid_fast_filter_screening_validation_stage121_visuals\stage121_selection_decision_flags.svg
artifacts\primeqa_hybrid_fast_filter_screening_validation_stage121_visuals\stage121_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_fast_filter_screening_completed_train_cv_selected_dev_reported
recommended_next_direction: review_fast_filter_screening_changed_cases
selected_config_id: special_token_exact_window40_rule_selector_v1
can_run_final_test_metrics_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Interpretation

The fast-filter screening route is safer than the stopped full-top200 reranking
route, but the improvement is weak:

- train-CV gains only one extra hit@20 case;
- dev hit@10, hit@20, and hit@200 are unchanged;
- dev mrr@20 decreases slightly by 0.0008;
- the stronger logistic route has better train/dev hit@20 deltas but violates
  the frozen hit@20 regression guard.

The result supports changed-case review, not runtime adoption.

## Next Step

Stage122 reviewed changed cases for:

```text
selected safe config:
  special_token_exact_window40_rule_selector_v1

strong but blocked config:
  top10_locked_route_vote_window50_pairwise_logistic_v1
```

The result is recorded in:

```text
docs/primeqa_hybrid_fast_filter_screening_changed_case_review.md
```

The blocked logistic config's extra hit@20 gains are real, but they come with
guard-relevant hit@20 regressions. Test remains locked. The next direction is a
first-stage recall expansion protocol.
