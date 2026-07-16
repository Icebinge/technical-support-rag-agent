# PrimeQA Hybrid Stage116 Prefix-Preserving Recall Expansion Validation

Stage: Stage 126

Status: completed, config selected on train-CV

Artifact:

```text
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126.json
```

## Scope

Stage126 runs the frozen Stage125 append-only protocol on train grouped
cross-validation and dev report-only validation.

This stage keeps the Stage116 top200 prefix exactly unchanged and appends only
deduplicated new candidates after rank 200. It uses only train/dev data. The
final test split remains locked. No final metrics were run, no runtime defaults
were changed, no fallback strategies were added, no model download was attempted,
and raw candidate rows were not written.

## Command

```text
python scripts\run_primeqa_hybrid_prefix_preserving_recall_expansion_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage126 Stage116 prefix-preserving recall expansion train-CV/dev validation after Stage125 protocol freeze; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Baseline Reproduction

```text
train evaluated answerable questions: 370
train Stage116 top200 hits: 345 / 370 = 0.9324

dev evaluated answerable questions: 76
dev Stage116 top200 hits: 69 / 76 = 0.9079
```

The baseline matches the Stage125 frozen prefix contract.

## Config Results

```text
prefix_existing_dense_broad_append200_v1:
  train target-depth gain: +9
  train appended gold recovery: 9
  train hit@200 losses: 0
  train prefix identity violations: 0
  dev target-depth gain: +1
  dev hit@200 losses: 0
  train guard passed: true

prefix_rrf_same_routes_append200_k60_v1:
  train target-depth gain: +9
  train appended gold recovery: 9
  train hit@200 losses: 0
  train prefix identity violations: 0
  dev target-depth gain: +1
  dev hit@200 losses: 0
  train guard passed: true

prefix_query_variant_append100_v1:
  train target-depth gain: +7
  train appended gold recovery: 7
  train hit@200 losses: 0
  train prefix identity violations: 0
  dev target-depth gain: +5
  dev hit@200 losses: 0
  train guard passed: true

prefix_rrf_same_routes_append100_k60_v1:
  train target-depth gain: +7
  train appended gold recovery: 7
  train hit@200 losses: 0
  train prefix identity violations: 0
  dev target-depth gain: +0
  dev hit@200 losses: 0
  train guard passed: true

prefix_route_balanced_append200_v1:
  train target-depth gain: +7
  train appended gold recovery: 7
  train hit@200 losses: 0
  train prefix identity violations: 0
  dev target-depth gain: +2
  dev hit@200 losses: 0
  train guard passed: true

prefix_rrf_same_routes_append100_k80_v1:
  train target-depth gain: +4
  train appended gold recovery: 4
  train hit@200 losses: 0
  train prefix identity violations: 0
  dev target-depth gain: +0
  dev hit@200 losses: 0
  train guard passed: true
```

## Selected Config

```text
selected_config_id: prefix_existing_dense_broad_append200_v1
selected_family_id: stage116_prefix_existing_dense_append_family_v1
source Stage124 config: existing_dense_cache_broad_union_top400_v1
append source algorithm: cached_dense_plus_lexical_rrf
route set: stage116_lexical_routes_plus_existing_dense_cache_routes
channel_top_k: 400
append_budget: 200
target_pool_depth: 400
```

Selected train summary:

```text
baseline hit@200: 345 / 370 = 0.9324
target-depth hit@400: 354 / 370 = 0.9568
target-depth gain: +9
appended gold recovery: 9
hit@200 delta vs Stage116 prefix: 0
hit@200 loss count: 0
prefix identity violation count: 0
append count average/median/p95/max: 200 / 200 / 200 / 200
```

Selected dev report-only summary:

```text
baseline hit@200: 69 / 76 = 0.9079
target-depth hit@400: 70 / 76 = 0.9211
target-depth gain: +1
appended gold recovery: 1
hit@200 delta vs Stage116 prefix: 0
hit@200 loss count: 0
prefix identity violation count: 0
```

## Fold Stability

For the selected config, train fold target-depth gains were:

```text
fold_1: +2
fold_2: +3
fold_3: +2
fold_4: +0
fold_5: +2
```

The selected config recovered additional train gold documents in 4 of 5 folds.
Fold 4 was neutral, not regressive.

## Decision

```text
status: primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_completed
selected_config_id: prefix_existing_dense_broad_append200_v1
eligible_config_count: 6 / 6
recommended_next_direction: review_stage116_prefix_preserving_recall_expansion_selected_config
can_run_final_test_metrics_now: false
can_open_final_test_gate_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Interpretation

The Stage125 hypothesis is validated on train/dev: preserving Stage116 ranks
1-200 removes the hit@200 regression problem seen in Stage124 while keeping the
extra target-depth recall signal.

The selected train config adds 9 train gold-document recoveries at depth 400 and
the dev report-only pass adds 1. The stronger dev-only signal appears in
`prefix_query_variant_append100_v1` with +5 dev gain, but dev was not used for
selection or retuning, so the train-selected config remains
`prefix_existing_dense_broad_append200_v1`.

This is still not a runtime default. It is a validated train/dev candidate for
review before deciding the next agent-facing retrieval design.

## Guard Checks

```text
guard checks: 21 / 21 passed
test split loaded: false
dev used for selection: false
dev used for retuning: false
model download attempted: false
raw candidate rows written: false
runtime defaults changed: false
fallback strategies added: false
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_train_target_depth_gain.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_dev_target_depth_gain.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_train_appended_gold_recovery.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_dev_appended_gold_recovery.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_train_hit200_loss.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_prefix_identity_violations.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_selected_append_count_summary.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_selection_decision_flags.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126_visuals\stage126_guard_check_status.svg
```

## Next Step

Stage127 should review the selected config against the broader agent design
before any runtime defaultization. The obvious question is whether the agent
should consume a fixed top200 retrieval context plus an optional 201-400
candidate expansion for second-stage evidence selection. Test remains locked.
