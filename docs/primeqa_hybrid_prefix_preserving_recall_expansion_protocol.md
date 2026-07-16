# PrimeQA Hybrid Stage116 Prefix-Preserving Recall Expansion Protocol

Stage: Stage 125

Status: frozen

Artifact:

```text
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json
```

## Scope

Stage125 freezes the next train/dev-only protocol after Stage124 showed that
bounded 300/400 candidate pools contain extra recall signal but can damage the
Stage116 top200 boundary.

The Stage125 design is append-only:

```text
ranks 1-200: exactly the Stage116 top200 order
ranks 201-300/400: newly appended candidates only
```

This stage reads only the public-safe Stage124 validation report. It does not
load split files, does not load corpus documents, does not build candidate rows,
does not run retrieval, does not run final metrics, does not change runtime
defaults, and does not add fallback strategies.

## Command

```text
python scripts\freeze_primeqa_hybrid_prefix_preserving_recall_expansion_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage125 Stage116 prefix-preserving recall expansion protocol after Stage124 no-selection validation; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source Signal

Stage124 source result:

```text
status: primeqa_hybrid_first_stage_recall_expansion_validation_completed_no_selection
recommended_next_direction: design_stage116_prefix_preserving_recall_expansion_protocol
eligible_config_count: 0 / 7
selected_config_id: null
positive_target_depth_signal_blocked_by_hit_at_200_loss: true
```

Best blocked signals carried into Stage125:

```text
existing_dense_cache_broad_union_top400_v1:
  train target-depth gain: +9
  train hit@200 delta/losses: +1 / 1
  dev target-depth gain: +1

rrf_same_routes_top400_k60_v1:
  train target-depth gain: +9
  train hit@200 delta/losses: +1 / 1
  dev target-depth gain: +1

rrf_same_routes_top300_k60_v1:
  train target-depth gain: +7
  train hit@200 delta/losses: -1 / 1
  dev target-depth gain: +0
```

## Baseline Prefix Contract

```text
baseline config: stage116_fixed_rrf_top200_baseline
prefix depth: 200
train baseline hit@200: 345 / 370 = 0.9324
dev baseline hit@200: 69 / 76 = 0.9079
ranks 1-200 must remain identical: true
prefix documents may be reordered: false
prefix documents may be dropped: false
prefix duplicate in append region allowed: false
hit@200 loss count must be zero by construction: true
```

## Candidate Families

```text
stage116_prefix_rrf_append_family_v1
stage116_prefix_existing_dense_append_family_v1
stage116_prefix_query_variant_append_family_v1
stage116_prefix_route_balanced_append_family_v1
```

## Candidate Configs

```text
prefix_rrf_same_routes_append100_k60_v1:
  source Stage124 config: rrf_same_routes_top300_k60_v1
  append budget: 100
  target pool depth: 300

prefix_rrf_same_routes_append200_k60_v1:
  source Stage124 config: rrf_same_routes_top400_k60_v1
  append budget: 200
  target pool depth: 400

prefix_existing_dense_broad_append200_v1:
  source Stage124 config: existing_dense_cache_broad_union_top400_v1
  append budget: 200
  target pool depth: 400

prefix_rrf_same_routes_append100_k80_v1:
  source Stage124 config: rrf_lexical_priority_top300_k80_v1
  append budget: 100
  target pool depth: 300

prefix_query_variant_append100_v1:
  source Stage124 config: query_variant_title_special_token_top300_v1
  append budget: 100
  target pool depth: 300

prefix_route_balanced_append200_v1:
  source Stage124 config: route_balanced_round_robin_top400_v1
  append budget: 200
  target pool depth: 400
```

All configs:

```text
append_start_rank: 201
deduplicate_against_prefix: true
deduplicate_within_append_region: true
may_reorder_prefix: false
may_drop_prefix_documents: false
may_insert_before_rank_201: false
requires_model_download: false
uses_test_membership: false
runtime_defaultization_allowed: false
fallback_strategies_enabled: false
```

## Selection Rules

```text
selection split: train
selection mode: train_grouped_cross_validation_prefix_preserving_candidate_selection
minimum train folds: 5
dev mode: single_pass_report_only_no_retuning
test access: false
runtime defaultization: false
fallback strategies: false
```

Primary metrics:

```text
prefix_identity_violation_count
hit_at_200_delta_vs_stage116_prefix
target_depth_hit_count_gain_vs_stage116_top200
appended_gold_recovery_count
train_fold_stability_at_target_depth
```

Guard thresholds:

```text
maximum_train_cv_prefix_identity_violation_count: 0
maximum_train_cv_hit_at_200_loss_count: 0
minimum_train_cv_hit_at_200_delta: 0
minimum_train_cv_target_depth_hit_count_gain: 1
maximum_channel_top_k: 400
maximum_output_pool_depth: 400
maximum_append_budget: 200
maximum_model_download_attempts: 0
maximum_raw_candidate_rows_written: 0
minimum_train_fold_count: 5
```

## Guard Checks

```text
guard checks: 18 / 18 passed
test split loaded: false
final test metrics run: false
runtime defaults changed: false
fallback strategies added: false
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125_visuals\stage125_stage124_blocked_signal_summary.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125_visuals\stage125_candidate_family_counts.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125_visuals\stage125_append_budgets.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125_visuals\stage125_target_pool_depths.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125_visuals\stage125_guard_thresholds.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125_visuals\stage125_protocol_decision_flags.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125_visuals\stage125_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_frozen
recommended_next_direction: run_stage116_prefix_preserving_recall_expansion_train_cv_dev_validation
can_run_prefix_preserving_recall_expansion_now: true
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Follow-Up

Stage126 ran the frozen prefix-preserving recall expansion protocol on train
grouped cross-validation and dev report-only validation. The result is recorded
in:

```text
docs/primeqa_hybrid_prefix_preserving_recall_expansion_validation.md
```

The selected train-CV config is `prefix_existing_dense_broad_append200_v1`.
Test remains locked.
