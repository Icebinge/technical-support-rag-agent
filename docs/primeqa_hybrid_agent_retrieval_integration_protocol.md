---
Stage: Stage 128
Title: PrimeQA hybrid agent retrieval integration protocol
Status: frozen
---

# PrimeQA Hybrid Agent Retrieval Integration Protocol

Stage128 freezes a train/dev-only protocol for integrating the Stage126 selected
prefix-preserving recall expansion into the agent retrieval design.

This stage does not run retrieval validation, answer validation, final test
metrics, or runtime defaultization. It reads only the public-safe Stage127
selected-config review report and writes only public-safe protocol summaries.

## Command

```text
python scripts\freeze_primeqa_hybrid_agent_retrieval_integration_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage128 agent retrieval integration protocol freeze after Stage127 selected-config review; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source Review

```text
source stage: Stage 127
source review id: primeqa_hybrid_stage116_prefix_preserving_recall_expansion_selected_config_review_v1
source status: primeqa_hybrid_stage116_prefix_preserving_recall_expansion_selected_config_review_completed
source next direction: freeze_agent_retrieval_integration_protocol_for_selected_prefix_expansion
```

Selected Stage127 facts carried into Stage128:

```text
selected config: prefix_existing_dense_broad_append200_v1
selected family: stage116_prefix_existing_dense_append_family_v1
train incremental recall gain: +9 / 370 = 0.0243
dev incremental recall gain: +1 / 76 = 0.0132
train/dev hit@200 losses: 0 / 0
train/dev prefix identity violations: 0 / 0
target pool depth: 400
append budget: 200
candidate depth multiplier vs Stage116: 2.0
channel count: 7
```

## Frozen Protocol

```text
protocol id: primeqa_hybrid_agent_retrieval_integration_protocol_v1
route name: stage116_prefix_expansion_agent_candidate_pool_integration
selected retrieval config: prefix_existing_dense_broad_append200_v1
selection source: Stage126 train grouped cross-validation
dev role: report_only_no_retuning
append source algorithm: cached_dense_plus_lexical_rrf
route set: stage116_lexical_routes_plus_existing_dense_cache_routes
channel_top_k: 400
```

## Agent Retrieval Contract

```text
candidate_pool_output_depth: 400
candidate_pool_is_not_automatic_answer_context: true
answer_context_policy: unchanged_until_stage129_validation
```

Rank-region contract:

```text
ranks 1-200:
  region: stage116_immutable_prefix
  source: Stage116 fixed top200 order
  purpose: preserve the validated hit@200 retrieval boundary
  may_reorder: false
  may_drop: false
  may_insert_expansion_candidate: false

ranks 201-400:
  region: stage128_append_expansion
  source: prefix_existing_dense_broad_append200_v1
  purpose: provide additional recall candidates for evidence selection
  append_budget: 200
  deduplicate_against_prefix: true
  deduplicate_within_region: true
  may_insert_before_rank_201: false
```

## Agent Candidate Interface

Candidate record fields to validate in Stage129:

```text
runtime_document_key
candidate_rank
rank_region_id
region_rank
retrieval_route_family
retrieval_score_summary
runtime_content_handle
```

Allowed runtime-visible signals:

```text
runtime query text
runtime corpus title/body/section content
Stage116 immutable prefix rank
Stage128 append region rank
BM25 and section BM25 route ranks
exact special-token route rank
existing local dense-cache route ranks
```

Forbidden runtime signals:

```text
test membership
gold labels
answer document labels
source-provided candidate labels
dev-selected thresholds
raw private rows in public artifacts
```

## Agent Consumer Policy

Allowed only after Stage129 validation:

```text
evidence_selection
answerability_estimation
citation_validation
```

Blocked in Stage128:

```text
direct_answer_context_all_400
runtime_default_retrieval_route
fallback_strategy_route
```

The extra 200 candidates are recall candidates only. They are not approved as
final answer context.

## Stage129 Validation Plan

```text
next stage: Stage129
action: run_agent_retrieval_integration_train_cv_dev_validation
selection split: train
selection mode: train_grouped_cross_validation_agent_integration_validation
minimum train folds: 5
validation split: dev
dev mode: single_pass_report_only_no_retuning
```

Primary checks:

```text
Stage116 ranks 1-200 remain identical
hit@200 loss count remains zero
candidate-pool target-depth recall does not regress
agent evidence selection does not reduce verified answer quality on train-CV
dev remains report-only
```

Metrics to report:

```text
retrieval hit@200 preservation
target-depth gold-document recall
selected evidence count and route mix
answer F1 delta
gold citation preservation
changed answer count
```

Rules:

```text
test access allowed: false
final test metrics allowed: false
test tuning allowed: false
default runtime policy: unchanged
runtime defaultization allowed in Stage128: false
fallback strategies enabled: false
```

## Risk Controls

```text
dev gain is smaller than train gain: true
best dev config differs from train-selected config: true
best dev config: prefix_query_variant_append100_v1
best dev target-depth gain: +5
answer quality measured for integration: false
final test run: false
runtime default changed: false
```

The best dev-only observation remains non-adoptable. Dev was report-only in
Stage126 and cannot be used to retune or replace the train-selected config.

## Guard Checks

```text
guard checks: 17 / 17 passed
test split loaded: false
final test metrics run: false
runtime defaultization allowed now: false
fallback strategies enabled: false
default runtime policy: unchanged
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_agent_retrieval_integration_protocol_stage128_visuals\stage128_selected_config_value.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_protocol_stage128_visuals\stage128_candidate_pool_contract.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_protocol_stage128_visuals\stage128_agent_consumer_policy.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_protocol_stage128_visuals\stage128_risk_review_flags.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_protocol_stage128_visuals\stage128_protocol_decision_flags.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_protocol_stage128_visuals\stage128_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_agent_retrieval_integration_protocol_frozen
recommended_next_direction: run_agent_retrieval_integration_train_cv_dev_validation
selected_config_id: prefix_existing_dense_broad_append200_v1
selected_family_id: stage116_prefix_existing_dense_append_family_v1
can_run_agent_retrieval_integration_validation_now: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
runtime_defaultization_allowed_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Next Step

Stage129 should run train grouped-CV plus dev report-only validation for the
frozen agent retrieval integration protocol. Test remains locked, runtime
defaults remain unchanged, and fallback strategies remain disabled.

Stage129 completed this validation in:

```text
docs/primeqa_hybrid_agent_retrieval_integration_validation.md
```

The validation is blocked/failed for runtime approval: Stage128 top400 preserved
the recall gain but lost 1 train-CV gold citation versus the Stage116 top200
agent-pool control.
