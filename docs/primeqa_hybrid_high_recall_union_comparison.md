# PrimeQA Hybrid High-Recall Union Candidate Pool

## Scope

Stage 116 evaluates a first-stage high-recall candidate pool for the PrimeQA
hybrid train/dev split. It unions several simple retrieval routes, deduplicates
the returned documents, ranks the pool with fixed reciprocal-rank scoring, and
reports recall-only gold-document coverage.

This is not an answer-generation experiment and not a runtime policy. It keeps
the frozen test split locked, does not run final metrics, does not use source
`DOC_IDS` as a runtime retrieval route, does not add fallback strategies, and
does not change runtime defaults.

## Command

```text
python scripts\run_primeqa_hybrid_high_recall_union_comparison.py --user-confirmed-direction --confirmation-note "user confirmed Stage116 first-stage high-recall multi-route union candidate-pool experiment on 2026-07-16; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Candidate Routes

Stage116 used seven fixed routes:

```text
full_document_bm25
section_bm25_max_section_rollup
title_heading_weighted_bm25
title_heading_only_bm25
special_token_boosted_bm25
dense_cache__intfloat_e5_small_v2__512_passage
dense_cache__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
```

Dense routes used only the existing local Stage80-compatible caches. No model
download was attempted.

## Results

Answerable rows:

```text
train: 370
dev: 76
```

Union candidate-pool recall:

```text
train union hit@10:  255 / 370 = 0.6892
train union hit@50:  303 / 370 = 0.8189
train union hit@100: 332 / 370 = 0.8973
train union hit@200: 345 / 370 = 0.9324
train uncapped union: 358 / 370 = 0.9676

dev union hit@10:  55 / 76 = 0.7237
dev union hit@50:  64 / 76 = 0.8421
dev union hit@100: 66 / 76 = 0.8684
dev union hit@200: 69 / 76 = 0.9079
dev uncapped union: 72 / 76 = 0.9474
```

Compared with full-document BM25:

```text
train hit@100: +20 hits, +0.0541
train hit@200: +14 hits, +0.0378

dev hit@100: +3 hits, +0.0395
dev hit@200: +3 hits, +0.0395
```

The untruncated route union is large:

```text
train average uncapped union size: 643.6135 documents
dev average uncapped union size: 662.2632 documents
dev p95 uncapped union size: 806 documents
```

Therefore the practical first-stage boundary should be the ranked top200 pool,
not the untruncated union.

## Train-Fold Stability

Stage116 does not tune a model, but it reports 5-fold train stability for the
fixed candidate-pool design:

```text
train-fold union hit@100 average: 0.8965
train-fold union hit@100 min/max: 0.8592 / 0.9200
train-fold union hit@100 spread: 0.0608

train-fold union hit@200 average: 0.9326
train-fold union hit@200 min/max: 0.9067 / 0.9565
train-fold union hit@200 spread: 0.0498
```

No raw group values were written.

## Channel Contribution

Most answerable dev cases are already found by full-document BM25 within each
route top200, but the union adds a small positive tail:

```text
dev first-new hits by channel order:
full_document_bm25: 66
section_bm25_max_section_rollup: 2
title_heading_only_bm25: 2
dense_cache__intfloat_e5_small_v2__512_passage: 1
dense_cache__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix: 1
```

This is enough to justify a second-stage precision/reranking design over the
fixed Stage116 top200 pool, but not enough to change runtime defaults.

## Guard Checks

```text
11 / 11 passed
```

Important boundary flags:

```text
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_dev_channel_hit_at_100.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_dev_union_recall_by_pool_depth.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_dev_union_delta_vs_baseline.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_dev_marginal_hits_by_channel.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_train_fold_union_hit_at_100.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_candidate_pool_size_summary.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_high_recall_union_candidate_pool_completed
can_continue_train_dev_development: true
can_continue_second_stage_precision_experiment: true
recommended_next_direction: design_second_stage_precision_reranking_protocol_over_stage116_pool
```

## Next Step

Stage117 should design the second-stage precision/reranking protocol over the
fixed Stage116 top200 candidate pool. It should remain train/dev-only, keep test
locked, avoid runtime/default changes, and avoid fallback strategies.
