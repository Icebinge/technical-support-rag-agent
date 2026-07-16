# PrimeQA Hybrid First-Stage Recall Expansion Validation

Stage: Stage 124

Status: completed, no config selected

Artifact:

```text
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124.json
```

## Scope

Stage124 runs the frozen Stage123 first-stage recall expansion protocol on train
grouped cross-validation and dev report-only validation. It evaluates bounded
300/400-depth candidate pools against the Stage116 top200 baseline.

This stage uses only train/dev data. The final test split remains locked. No
final metrics were run, no runtime defaults were changed, no fallback strategies
were added, no model download was attempted, and raw candidate rows were not
written.

## Command

```text
python scripts\run_primeqa_hybrid_first_stage_recall_expansion_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage124 first-stage recall expansion train-CV/dev validation after Stage123 protocol freeze; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Baseline Reproduction

```text
train evaluated answerable questions: 370
train Stage116 top200 hits: 345 / 370 = 0.9324

dev evaluated answerable questions: 76
dev Stage116 top200 hits: 69 / 76 = 0.9079
```

The baseline matches the Stage116 recorded top200 boundary.

## Config Results

```text
rrf_same_routes_top300_k60_v1:
  train target-depth gain: +7
  train hit@200 delta/losses: -1 / 1
  dev target-depth gain: +0
  dev hit@200 delta/losses: +0 / 0
  train guard passed: false

rrf_same_routes_top400_k60_v1:
  train target-depth gain: +9
  train hit@200 delta/losses: +1 / 1
  dev target-depth gain: +1
  dev hit@200 delta/losses: +0 / 0
  train guard passed: false

rrf_lexical_priority_top300_k80_v1:
  train target-depth gain: +3
  train hit@200 delta/losses: -6 / 7
  dev target-depth gain: +0
  dev hit@200 delta/losses: -1 / 1
  train guard passed: false

route_balanced_round_robin_top300_v1:
  train target-depth gain: +3
  train hit@200 delta/losses: -7 / 11
  dev target-depth gain: -1
  dev hit@200 delta/losses: -2 / 2
  train guard passed: false

route_balanced_round_robin_top400_v1:
  train target-depth gain: +6
  train hit@200 delta/losses: -7 / 11
  dev target-depth gain: +2
  dev hit@200 delta/losses: -2 / 2
  train guard passed: false

query_variant_title_special_token_top300_v1:
  train target-depth gain: +0
  train hit@200 delta/losses: -11 / 14
  dev target-depth gain: +3
  dev hit@200 delta/losses: +1 / 3
  train guard passed: false

existing_dense_cache_broad_union_top400_v1:
  train target-depth gain: +9
  train hit@200 delta/losses: +1 / 1
  dev target-depth gain: +1
  dev hit@200 delta/losses: +0 / 0
  train guard passed: false
```

No config passed the frozen train guard because every positive train recall
signal also introduced at least one Stage116 hit@200 loss.

## Decision

```text
status: primeqa_hybrid_first_stage_recall_expansion_validation_completed_no_selection
selected_config_id: null
eligible_config_count: 0 / 7
positive_target_depth_signal_blocked_by_hit_at_200_loss: true
recommended_next_direction: design_stage116_prefix_preserving_recall_expansion_protocol
can_run_final_test_metrics_now: false
can_open_final_test_gate_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Interpretation

The frozen Stage123 configs confirm a useful but blocked signal:

- broader 300/400 candidate pools can recover extra target-depth hits;
- the strongest train gains are +9 at target depth;
- dev gains are small but positive for several configs;
- all train-positive configs violate the no-loss Stage116 hit@200 guard.

The practical next direction is not to default any current 300/400 reranked
pool. It is to design a prefix-preserving expansion: keep the Stage116 top200
unchanged, then append additional candidates into positions 201-400. That should
preserve the existing hit@200 boundary by construction while testing whether the
extra target-depth recall signal is still real.

## Guard Checks

```text
guard checks: 16 / 16 passed
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
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124_visuals\stage124_train_target_depth_gain.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124_visuals\stage124_dev_target_depth_gain.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124_visuals\stage124_train_hit200_delta.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124_visuals\stage124_dev_hit200_delta.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124_visuals\stage124_train_fold_target_hit_summary.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124_visuals\stage124_candidate_pool_size.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124_visuals\stage124_selection_decision_flags.svg
artifacts\primeqa_hybrid_first_stage_recall_expansion_validation_stage124_visuals\stage124_guard_check_status.svg
```

## Next Step

Stage125 should freeze a Stage116 prefix-preserving recall expansion protocol.
It should keep positions 1-200 exactly unchanged and evaluate only appended
201-300/400 candidates on train-CV plus dev report-only data. Test remains
locked.

