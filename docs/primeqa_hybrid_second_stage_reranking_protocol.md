# PrimeQA Hybrid Second-Stage Reranking Protocol

## Scope

Stage117 freezes the protocol for second-stage precision/reranking over the
fixed Stage116 ranked top200 candidate pool.

This stage reads only the saved public-safe Stage116 report. It does not load
train/dev/test split files, does not load corpus documents, does not build
candidate rows, does not run reranking or answer metrics, does not run final
metrics, does not select from dev-only observations, does not add fallback
strategies, and does not change runtime defaults.

## Command

```text
python scripts\freeze_primeqa_hybrid_second_stage_reranking_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage117 second-stage precision/reranking protocol over fixed Stage116 top200 candidate pool on 2026-07-16; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source

```text
artifacts\primeqa_hybrid_high_recall_union_stage116.json
```

Stage116 source facts used by this protocol:

```text
status: primeqa_hybrid_high_recall_union_candidate_pool_completed
candidate pool: stage116_multi_route_union_candidate_pool
fixed pool depth: top200
dev union hit@100: 0.8684
dev union hit@200: 0.9079
dev hit@200 delta vs BM25: +3
dev uncapped union not found: 4
guard checks: 11 / 11 passed
```

## Fixed Candidate Pool

Stage117 freezes this boundary:

```text
source_pool_id: stage116_multi_route_union_candidate_pool
candidate_pool_depth: 200
source_stage116_rrf_k: 60
source_stage116_channel_top_k: 200
reranker_may_reorder_pool: true
reranker_may_add_documents: false
reranker_may_drop_documents_before_top200_metric: false
uncapped_union_is_not_runtime_input: true
```

The untruncated Stage116 union is explicitly blocked as an answer input because
it averages hundreds of documents.

## Candidate Families

Stage117 freezes three second-stage candidate families and eight configs:

```text
channel_rank_feature_reranker_family_v1: 2 configs
lexical_document_feature_reranker_family_v1: 3 configs
supervised_lightweight_reranker_family_v1: 3 configs
```

Configs:

```text
crf_route_agreement_best_rank_v1
crf_lexical_routes_first_v1
ldf_title_heading_overlap_v1
ldf_title_heading_body_coverage_v1
ldf_special_token_title_heading_v1
slr_logistic_balanced_v1
slr_logistic_hard_negative_v1
slr_ridge_rank_proxy_v1
```

No config may use source `DOC_IDS`, test membership, gold answer text, or answer
document IDs as runtime features. Train labels are allowed only for train-CV
fitting.

## Selection Rules

```text
selection_split: train
selection_mode: train_grouped_cross_validation_then_full_train_refit
minimum_train_folds: 5
train_group_key: normalized_question_plus_answer_document
eligible_training_rows: answerable train rows whose gold document is present in fixed Stage116 top200 pool
baseline_order: stage116_fixed_rrf_pool_order
validation_split: dev
dev_validation_mode: single_pass_report_only_no_retuning
```

Primary metrics:

```text
mrr_at_20_delta_vs_stage116_order
hit_at_10_delta_vs_stage116_order
hit_at_20_delta_vs_stage116_order
```

Guard thresholds:

```text
maximum_train_cv_hit_at_200_loss_count: 0
maximum_train_cv_bm25_top10_gold_demotions_to_below_50: 0
maximum_train_cv_hit_at_20_regression_rate: 0.02
maximum_train_cv_top10_regression_count: 3
minimum_train_cv_mrr_at_20_delta: 0.0
```

## Blocked Options

```text
uncapped_union_as_answer_input_blocked
source_doc_ids_oracle_reranker_blocked
dev_selected_threshold_blocked
final_test_reranking_metrics_blocked
runtime_defaultization_in_stage117_blocked
```

## Guard Checks

```text
19 / 19 passed
```

Important boundary flags:

```text
can_continue_train_dev_development: true
can_run_second_stage_reranking_now: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_stage116_candidate_pool_recall.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_candidate_family_priorities.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_candidate_config_counts.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_objective_weights.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_guard_thresholds.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_protocol_decision_flags.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_second_stage_reranking_protocol_frozen
recommended_next_direction: run_second_stage_reranking_train_cv_dev_validation
can_run_second_stage_reranking_now: true
```

## Stage118 Result

Stage118 ran the frozen second-stage reranking train-CV/dev validation over the
fixed Stage116 top200 candidate pool:

```text
report: artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118.json
doc: docs\primeqa_hybrid_second_stage_reranking_validation.md
selectable_config_count: 0 / 8
status: primeqa_hybrid_second_stage_reranking_completed_no_train_cv_selectable_config
recommended_next_direction: record_second_stage_reranking_stop_decision
```

Stage118 kept test locked, did not use dev for selection or retuning, did not
change runtime defaults, and did not add fallback strategies.

## Next Step

Stage119 should record the second-stage reranking stop decision. It should keep
test locked, avoid final metrics, avoid runtime/default changes, and avoid
fallback strategies.
