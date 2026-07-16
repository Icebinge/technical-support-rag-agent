# PrimeQA Hybrid First-Stage Recall Expansion Protocol

Stage: Stage 123

Status: frozen

Artifact:

```text
artifacts\primeqa_hybrid_first_stage_recall_expansion_protocol_stage123.json
```

## Scope

Stage123 freezes the next train/dev-only protocol after Stage122 showed that
second-stage screening has real but guard-risky hit@20 signal. The new route
returns to first-stage candidate generation: broaden the candidate pool with
simple, fast, runtime-visible retrieval routes before any precise second-stage
selection.

This stage reads only the public-safe Stage122 report. It does not load split
files, does not load corpus documents, does not build candidate rows, does not
run retrieval, does not run final metrics, does not change runtime defaults,
and does not add fallback strategies.

## Command

```text
python scripts\freeze_primeqa_hybrid_first_stage_recall_expansion_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage123 first-stage recall expansion protocol after Stage122 changed-case review; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source Signal

Stage122 source result:

```text
status: primeqa_hybrid_fast_filter_screening_changed_case_review_completed
recommended_next_direction: design_first_stage_recall_expansion_protocol
blocked_signal_has_real_hit20_recoveries: true
blocked_signal_has_guard_relevant_regressions: true
```

Changed-case signal carried into Stage123:

```text
special_token_exact_window40_rule_selector_v1:
  interpretation: safe_but_weak
  train hit@20 recoveries/regressions: 4 / 3
  dev hit@20 recoveries/regressions: 0 / 0

top10_locked_route_vote_window50_pairwise_logistic_v1:
  interpretation: positive_signal_but_guard_risky
  train hit@20 recoveries/regressions: 11 / 7
  dev hit@20 recoveries/regressions: 2 / 1
```

## Baseline Boundary

```text
baseline pool: stage116_multi_route_union_candidate_pool
baseline pool depth: 200
baseline train hit@200: 0.9324
baseline dev hit@200: 0.9079
baseline uncapped train hit: 0.9676
baseline uncapped dev hit: 0.9474
```

The uncapped Stage116 union has better recall but is too large to become a
default runtime pool. Stage123 therefore freezes bounded 300/400-depth
candidate-pool experiments.

## Candidate Families

```text
rrf_depth_expansion_family_v1
route_balanced_union_family_v1
query_variant_lexical_family_v1
existing_dense_cache_union_family_v1
```

## Candidate Configs

```text
rrf_same_routes_top300_k60_v1:
  family: rrf_depth_expansion_family_v1
  channel_top_k: 300
  target_pool_depth: 300

rrf_same_routes_top400_k60_v1:
  family: rrf_depth_expansion_family_v1
  channel_top_k: 400
  target_pool_depth: 400

rrf_lexical_priority_top300_k80_v1:
  family: rrf_depth_expansion_family_v1
  channel_top_k: 300
  target_pool_depth: 300

route_balanced_round_robin_top300_v1:
  family: route_balanced_union_family_v1
  channel_top_k: 300
  target_pool_depth: 300

route_balanced_round_robin_top400_v1:
  family: route_balanced_union_family_v1
  channel_top_k: 400
  target_pool_depth: 400

query_variant_title_special_token_top300_v1:
  family: query_variant_lexical_family_v1
  channel_top_k: 250
  target_pool_depth: 300

existing_dense_cache_broad_union_top400_v1:
  family: existing_dense_cache_union_family_v1
  channel_top_k: 400
  target_pool_depth: 400
```

## Selection Rules

```text
selection split: train
selection mode: train_grouped_cross_validation_candidate_pool_selection
minimum train folds: 5
dev mode: single_pass_report_only_no_retuning
test access: false
runtime defaultization: false
fallback strategies: false
```

Primary metrics:

```text
hit_at_200_delta_vs_stage116_order
target_depth_hit_count_gain_vs_stage116_top200
target_depth_missing_count_reduction_vs_stage116_top200
train_fold_stability_at_target_depth
```

Guard thresholds:

```text
maximum_train_cv_hit_at_200_loss_count: 0
minimum_train_cv_target_depth_hit_count_gain: 1
maximum_channel_top_k: 400
maximum_output_pool_depth: 400
maximum_model_download_attempts: 0
maximum_raw_candidate_rows_written: 0
minimum_train_fold_count: 5
```

## Guard Checks

```text
guard checks: 16 / 16 passed
test split loaded: false
final test metrics run: false
runtime defaults changed: false
fallback strategies added: false
model downloads allowed: false
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_first_stage_recall_expansion_protocol_stage123_visuals\stage123_stage122_signal_summary.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_protocol_stage123_visuals\stage123_candidate_family_counts.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_protocol_stage123_visuals\stage123_target_pool_depths.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_protocol_stage123_visuals\stage123_channel_top_k_budgets.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_protocol_stage123_visuals\stage123_guard_thresholds.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_protocol_stage123_visuals\stage123_protocol_decision_flags.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_protocol_stage123_visuals\stage123_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_first_stage_recall_expansion_protocol_frozen
recommended_next_direction: run_first_stage_recall_expansion_train_cv_dev_validation
can_run_first_stage_recall_expansion_now: true
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Next Step

Stage124 should run the frozen first-stage recall expansion protocol on train
grouped cross-validation and dev report-only validation. Test remains locked.

