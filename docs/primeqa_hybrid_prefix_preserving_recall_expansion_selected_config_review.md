# PrimeQA Hybrid Stage116 Prefix-Preserving Recall Expansion Selected-Config Review

Stage: Stage 127

Status: completed

Artifact:

```text
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_stage127.json
```

## Scope

Stage127 reviews the Stage126 train-selected prefix-preserving recall expansion
config before any agent-facing retrieval integration work.

This stage reads only the public-safe Stage126 validation report. It does not
load train/dev/test split files, does not load corpus documents, does not build
candidate rows, does not run retrieval, reranking, answering, or final metrics,
does not select from dev-only observations, does not add fallback strategies,
and does not change runtime defaults.

## Command

```text
python scripts\review_primeqa_hybrid_prefix_preserving_recall_expansion_selected_config.py --user-confirmed-review --confirmation-note "user confirmed Stage127 selected-config review after Stage126 selected prefix_existing_dense_broad_append200_v1; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source

```text
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_validation_stage126.json
```

Stage126 source facts:

```text
status: primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_completed
recommended_next_direction: review_stage116_prefix_preserving_recall_expansion_selected_config
selected_config_id: prefix_existing_dense_broad_append200_v1
selected_family_id: stage116_prefix_existing_dense_append_family_v1
eligible_config_count: 6 / 6
guard checks: 21 / 21 passed
public_safe_contract.forbidden_keys_found: []
```

Baseline reproduced by Stage126:

```text
train Stage116 top200: 345 / 370 = 0.9324
dev Stage116 top200: 69 / 76 = 0.9079
```

## Selected Config

```text
config_id: prefix_existing_dense_broad_append200_v1
family_id: stage116_prefix_existing_dense_append_family_v1
source Stage124 config: existing_dense_cache_broad_union_top400_v1
append source algorithm: cached_dense_plus_lexical_rrf
route set: stage116_lexical_routes_plus_existing_dense_cache_routes
channel_top_k: 400
append_budget: 200
target_pool_depth: 400
```

Observed selected-config value:

```text
train incremental recall gain: +9 / 370 = 0.0243
dev incremental recall gain: +1 / 76 = 0.0132
train hit@200 loss count: 0
dev hit@200 loss count: 0
train prefix identity violation count: 0
dev prefix identity violation count: 0
```

Train fold target-depth gains:

```text
fold_1: +2
fold_2: +3
fold_3: +2
fold_4: +0
fold_5: +2
```

## Agent Retrieval Contract

Stage127 does not implement the agent integration, but it defines the contract
the next protocol should preserve:

```text
ranks 1-200:
  region: stage116_immutable_prefix
  role: preserve the validated Stage116 top200 boundary
  may_reorder: false
  may_drop: false

ranks 201-400:
  region: stage126_append_expansion
  role: add recall candidates for downstream evidence selection
  deduplicate_against_prefix: true
  may_insert_before_rank_201: false
```

Cost profile:

```text
baseline candidate depth: 200
target candidate depth: 400
candidate depth multiplier vs Stage116: 2.0
additional candidates per query: 200
selected train average append count: 200.0
selected dev average append count: 200.0
channel count: 7
channel families:
  dense_cache: 2
  lexical_bm25: 1
  lexical_exact_token: 1
  lexical_section_rollup: 1
  lexical_weighted_document: 2
```

## Risk Review

```text
dev gain is smaller than train gain: true
best dev config differs from train-selected config: true
best dev config: prefix_query_variant_append100_v1
best dev target-depth gain: +5
answer quality measured: false
final test run: false
runtime default changed: false
```

The best dev-only observation does not override the train-CV selection. It is
recorded as a risk and a possible future research signal, not as a reason to
retune Stage126 after seeing dev.

## Guard Checks

```text
guard checks: 15 / 15 passed
test split loaded: false
final test metrics run: false
runtime defaultization allowed now: false
runtime defaults changed: false
fallback strategies added: false
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_stage127_visuals\stage127_selected_incremental_recall.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_stage127_visuals\stage127_config_train_dev_gain.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_stage127_visuals\stage127_boundary_safety.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_stage127_visuals\stage127_candidate_pool_shape.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_stage127_visuals\stage127_decision_flags.svg
artifacts\primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_stage127_visuals\stage127_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_stage116_prefix_preserving_recall_expansion_selected_config_review_completed
recommended_next_direction: freeze_agent_retrieval_integration_protocol_for_selected_prefix_expansion
selected_config_id: prefix_existing_dense_broad_append200_v1
selected_family_id: stage116_prefix_existing_dense_append_family_v1
selected_config_supported_for_agent_protocol_design: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
runtime_defaultization_allowed_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Next Step

Stage128 should freeze an agent retrieval integration protocol for the selected
prefix-preserving expansion. The protocol should treat the 400-depth output as a
candidate pool for evidence selection, not as an automatic answer context. Test
remains locked.

Stage128 completed this protocol freeze in:

```text
docs/primeqa_hybrid_agent_retrieval_integration_protocol.md
```

The frozen protocol keeps ranks 1-200 immutable, exposes ranks 201-400 only as
additional agent candidate-pool entries, blocks direct all-400 answer context,
and keeps runtime defaults unchanged.
