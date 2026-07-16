# PrimeQA Hybrid Stage120 Fast-Filter Screening Protocol

Stage120 freezes a train/dev-only protocol for the next research direction:
use a cheap first-pass filter over the fixed Stage116 top200 candidate pool,
then use a more constrained alternate screening selector on the filtered
subset.

This stage is a protocol freeze, not an effectiveness result.

## Command

```text
python scripts\freeze_primeqa_hybrid_fast_filter_screening_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage120 fast-filter plus alternate screening protocol after Stage119 stopped full top200 reranking; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source

Stage120 reads only:

```text
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119.json
```

It does not load split files, corpus documents, candidate rows, raw question
text, raw answer text, or raw document text.

Stage119 stopped the previous second-stage reranking family:

```text
status: primeqa_hybrid_second_stage_reranking_family_stopped
selected Stage118 configs: 0 / 8
train top200 gold present: 0.9324
dev top200 gold present: 0.9079
```

The main Stage119 lesson was that the problem was not top200 recall loss.
Full top200 reranking preserved hit@200 but caused too many top10/top20 ranking
regressions.

## Frozen Protocol

```text
protocol_id: primeqa_hybrid_fast_filter_screening_protocol_v1
route_name: conservative_fast_filter_plus_alternate_screening
candidate_pool_depth: 200
status: primeqa_hybrid_fast_filter_screening_protocol_frozen
recommended_next_direction: run_fast_filter_screening_train_cv_dev_validation
```

The fixed pool contract is:

```text
source_pool_id: stage116_multi_route_union_candidate_pool
candidate_pool_depth: 200
screening_may_reorder_entire_top200: false
screening_may_add_documents: false
screening_may_use_uncapped_union: false
```

The fast filter may use cheap runtime-visible signals only:

```text
Stage116 baseline rank
route hit counts
best route rank
BM25 top10/top20 membership
title/heading/body lexical overlap
special-token exact match counts
locally cached dense-route ranks
```

It may not use labels or split membership at runtime:

```text
answer_doc_id
gold_answer
question_id
source_doc_ids
test_membership
```

## Candidate Families

```text
protected_prefix_fast_filter_family_v1
evidence_density_fast_filter_family_v1
pairwise_screening_selector_family_v1
```

Stage120 freezes 6 candidate configs:

```text
top10_locked_route_vote_window50_pairwise_logistic_v1
top5_locked_strong_consensus_window80_pairwise_gbdt_v1
top20_locked_low_confidence_tail_screen_v1
evidence_density_window40_pairwise_logistic_v1
special_token_exact_window40_rule_selector_v1
hybrid_filter_window80_pairwise_gbdt_v1
```

All configs preserve a protected Stage116 prefix and limit top10 promotion:

```text
minimum protected_prefix_depth: 5
maximum promotion_budget_top10: 1
full_top200_rerank_allowed: false
```

## Train/Dev Rules

Selection remains train-only:

```text
selection_split: train
selection_mode: train_grouped_cross_validation_then_full_train_refit
minimum_train_folds: 5
```

Dev remains report-only:

```text
validation_split: dev
dev_selection_allowed: false
dev_retuning_allowed: false
dev_threshold_tuning_allowed: false
```

Test remains locked:

```text
test_access_allowed: false
final_test_metrics_allowed: false
test_tuning_allowed: false
```

Runtime remains unchanged:

```text
default_runtime_policy: unchanged
fallback_strategies_enabled: false
runtime_defaultization_allowed_in_stage120: false
```

## Guard Thresholds

The next train-CV/dev validation must satisfy the frozen Stage120 guard
thresholds before any later runtime discussion:

```text
maximum_train_cv_hit_at_200_loss_count: 0
maximum_train_cv_top10_regression_count: 0
maximum_train_cv_hit_at_20_regression_rate: 0.01
maximum_train_cv_bm25_top10_gold_demotions_to_below_50: 0
minimum_train_cv_hit_at_10_delta: 0.0
minimum_train_cv_mrr_at_20_delta: 0.0
maximum_train_cv_promoted_tail_docs_into_top10_average: 1.0
```

## Guard Checks

```text
16 / 16 passed
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_stage119_stop_summary.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_candidate_family_counts.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_fast_filter_window_sizes.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_promotion_budgets.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_guard_thresholds.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_protocol_decision_flags.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_fast_filter_screening_protocol_frozen
recommended_next_direction: run_fast_filter_screening_train_cv_dev_validation
can_run_fast_filter_screening_now: true
requires_user_confirmation_before_train_dev_run: true
can_run_final_test_metrics_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Next Step

Stage121 has now run the frozen fast-filter plus alternate-screening protocol on
train grouped cross-validation and dev report-only validation:

```text
docs/primeqa_hybrid_fast_filter_screening_validation.md
```

The selected train-CV-safe config is:

```text
special_token_exact_window40_rule_selector_v1
```

The result supports changed-case review, not runtime defaultization.

Continue to preserve:

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```
