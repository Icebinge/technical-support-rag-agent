# PrimeQA Hybrid Second-Stage Reranking Validation

## Scope

Stage118 runs the frozen Stage117 second-stage reranking validation over the
fixed Stage116 top200 candidate pool.

This stage rebuilds train/dev candidate records in memory, evaluates the eight
frozen reranking configs with train grouped-CV, reports dev only for a train-CV
selected config, keeps the frozen test split locked, does not run answer
metrics, does not change runtime defaults, and does not add fallback
strategies.

## Command

```text
python scripts\run_primeqa_hybrid_second_stage_reranking_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage118 train-CV/dev second-stage reranking validation in current turn; test locked; dev report-only; runtime defaults unchanged; no fallback strategies"
```

## Source Artifacts

```text
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117.json
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_train.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_dev.jsonl
data\raw\primeqa_techqa\TechQA\training_and_dev\training_dev_technotes.sections.json
artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json
```

No test split path is accepted by the Stage118 runner.

## Candidate Pool Rebuild

Stage118 reproduced the Stage116 top200 candidate-pool recall exactly:

```text
train answerable rows: 370
train top200 gold present: 345 / 370 = 0.9324

dev answerable rows: 76
dev top200 gold present: 69 / 76 = 0.9079
```

In-memory candidate rows:

```text
train: 74,000
dev: 15,200
raw candidate rows written: false
```

Dense routes used the existing local Stage80-compatible caches:

```text
dense_channel_preflight.status: dense_channels_ready
can_run_without_download: true
no_model_download_attempted: true
```

## Baseline Order

The comparison baseline is the fixed Stage116 RRF pool order:

```text
train hit@10:  0.6892
train hit@20:  0.7541
train hit@50:  0.8189
train hit@100: 0.8973
train hit@200: 0.9324
train mrr@20:  0.5062

dev hit@10:  0.7237
dev hit@20:  0.8026
dev hit@50:  0.8421
dev hit@100: 0.8684
dev hit@200: 0.9079
dev mrr@20:  0.5588
```

## Feature Alias Correction

Stage117 contained a config feature name:

```text
bm25_top10_non_gold_indicator
```

Stage118 does not use gold labels as runtime features. It records the following
explicit alias:

```text
bm25_top10_non_gold_indicator -> bm25_top10_indicator
```

The runtime-visible feature only means that the candidate appears in the
full-document BM25 top10. Train labels are still used only for supervised
train-CV fitting.

## Train-CV Results

No config passed the frozen Stage117 train-CV selection guards:

```text
selected_config_id: null
selectable_config_count: 0 / 8
status: no_train_cv_selectable_config
```

Top configs by objective:

```text
crf_lexical_routes_first_v1:
  objective: -0.0229
  mrr@20 delta: +0.0102
  hit@10 delta: -0.0162
  hit@20 delta: -0.0190
  hit@200 delta: +0.0000
  failed guards:
    train_cv_hit_at_20_regression_rate_within_guard: 0.0324 > 0.02
    train_cv_top10_regression_count_within_guard: 15 > 3

crf_route_agreement_best_rank_v1:
  objective: -5.9883
  mrr@20 delta: +0.0018
  hit@10 delta: +0.0000
  hit@20 delta: +0.0081
  hit@200 delta: +0.0000
  failed guards:
    train_cv_bm25_top10_gold_demotions_to_below_50_within_guard: 3 > 0
    train_cv_hit_at_20_regression_rate_within_guard: 0.0243 > 0.02
    train_cv_top10_regression_count_within_guard: 21 > 3

ldf_special_token_title_heading_v1:
  objective: -16.2573
  mrr@20 delta: -0.0428
  hit@10 delta: -0.0730
  hit@20 delta: -0.0622
  hit@200 delta: +0.0000
```

The main failure pattern is not top200 recall loss. All configs preserved
hit@200, but reranking moved too many already-good top10/top20 cases down.

## Dev Validation

Dev remains report-only and was not used for selection or retuning.

Because train-CV selected no config, Stage118 did not run a selected-config dev
comparison:

```text
dev_validation.status: no_train_cv_selectable_config
dev_used_for_selection: false
dev_used_for_retuning: false
```

## Guard Checks

```text
16 / 16 passed
public_safe_contract.forbidden_keys_found: []
```

Important boundary flags:

```text
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Visualizations

```text
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_train_cv_objective_scores.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_train_cv_mrr_at_20_delta.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_train_cv_hit_at_10_delta.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_train_cv_guard_pass_counts.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_dev_selected_config_deltas.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_selection_decision_flags.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_second_stage_reranking_completed_no_train_cv_selectable_config
recommended_next_direction: record_second_stage_reranking_stop_decision
```

Stage118 should not be defaultized and should not open the final test gate.

## Next Step

Stage119 recorded the second-stage reranking stop decision:

```text
report: artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119.json
doc: docs\primeqa_hybrid_second_stage_reranking_stop_decision.md
status: primeqa_hybrid_second_stage_reranking_family_stopped
recommended_next_direction: user_confirmed_next_research_direction_required
```

Stage119 preserved the current boundary:

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```

The next train/dev-only research direction requires explicit user confirmation.
