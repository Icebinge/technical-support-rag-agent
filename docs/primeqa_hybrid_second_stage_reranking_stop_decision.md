# PrimeQA Hybrid Second-Stage Reranking Stop Decision

## Scope

Stage119 records the stop decision for the frozen Stage117 second-stage
reranking family after Stage118 selected no train-CV-safe config.

This stage reads only the public-safe Stage118 report. It does not load
train/dev/test split files, does not load corpus documents, does not rebuild
candidate rows, does not run retrieval, reranking, answer, or final metrics,
does not select from dev-only observations, does not add fallback strategies,
and does not change runtime defaults.

## Command

```text
python scripts\decide_primeqa_hybrid_second_stage_reranking_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage119 second-stage reranking stop decision in current turn after Stage118 selected 0 of 8 configs; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source

```text
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118.json
```

Stage118 source facts:

```text
status: primeqa_hybrid_second_stage_reranking_completed_no_train_cv_selectable_config
recommended_next_direction: record_second_stage_reranking_stop_decision
selectable_config_count: 0 / 8
guard checks: 16 / 16 passed
public_safe_contract.forbidden_keys_found: []
```

Stage118 reproduced the fixed Stage116 candidate pool:

```text
train top200 gold present: 0.9324
dev top200 gold present: 0.9079
train in-memory candidate rows: 74,000
dev in-memory candidate rows: 15,200
raw candidate rows written: false
```

## Stop Evidence

The stopped family is:

```text
family_id: second_stage_reranking_candidate_family
source_protocol_id: primeqa_hybrid_second_stage_reranking_protocol_v1
source_analysis_id: primeqa_hybrid_second_stage_reranking_train_cv_dev_validation_v1
```

All Stage118 configs were nonselectable:

```text
channel_rank_feature_reranker_family_v1: 0 / 2 selectable
lexical_document_feature_reranker_family_v1: 0 / 3 selectable
supervised_lightweight_reranker_family_v1: 0 / 3 selectable
```

The best train-CV signal was still blocked:

```text
crf_lexical_routes_first_v1:
  train mrr@20 delta: +0.0102
  train hit@10 delta: -0.0162
  train hit@20 delta: -0.0190
  train hit@200 delta: +0.0000
  failed guards:
    train_cv_hit_at_20_regression_rate_within_guard
    train_cv_top10_regression_count_within_guard

crf_route_agreement_best_rank_v1:
  train mrr@20 delta: +0.0018
  train hit@10 delta: +0.0000
  train hit@20 delta: +0.0081
  train hit@200 delta: +0.0000
  failed guards:
    train_cv_bm25_top10_gold_demotions_to_below_50_within_guard
    train_cv_hit_at_20_regression_rate_within_guard
    train_cv_top10_regression_count_within_guard
```

The core failure pattern is ranking regression, not top200 recall loss. All
configs preserved hit@200, but the rerankers moved too many already-good top10
or top20 cases downward.

## Dev Observation

Dev remained report-only. It was not used for selection or retuning:

```text
dev_validation.status: no_train_cv_selectable_config
dev_used_for_selection: false
dev_used_for_retuning: false
dev_observations_are_non_adoptable: true
best dev mrr@20 delta: +0.0085 from crf_lexical_routes_first_v1
```

The dev observation does not override the train-CV stop decision.

## Guard Checks

```text
27 / 27 passed
```

Important boundary flags:

```text
current_route_defaultization: blocked
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_train_cv_objective_scores.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_train_cv_mrr_at_20_deltas.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_train_cv_hit_at_10_deltas.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_train_cv_guard_failure_reasons.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_selectability_by_family.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_stop_decision_flags.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_stop_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_second_stage_reranking_family_stopped
stopped_family_id: second_stage_reranking_candidate_family
recommended_next_direction: user_confirmed_next_research_direction_required
```

Stage119 does not justify runtime defaultization and does not open the final
test gate.

## Next Step

Stage120 requires user confirmation before choosing the next train/dev-only
research direction.

Continue to preserve:

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```
