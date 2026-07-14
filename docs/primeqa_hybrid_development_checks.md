# PrimeQA Hybrid Development Checks

This document records Stage 70: train/dev-only development checks for the frozen
PrimeQA/TechQA hybrid split.

Stage 70 uses `primeqa_hybrid_stage68_v1`. It runs BM25 retrieval baselines on
train/dev, audits the Stage 69 train/dev candidate artifact, keeps the frozen
test split locked, does not run final metrics, and does not change the default
runtime.

## Input Contract

```text
split_name: primeqa_hybrid_stage68_v1
protocol_version: primeqa_hybrid_split_v1
development_splits: train, dev
forbidden_final_splits: test
```

Source files:

```text
artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_train.jsonl
sha256: cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155

artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_dev.jsonl
sha256: 071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f

data/raw/primeqa_techqa/TechQA/training_and_dev/training_dev_technotes.sections.json
sha256: f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b

artifacts/primeqa_hybrid_rebuild_stage69_candidates.jsonl
sha256: d379d59f5172394a40bcd1852aa8188f2dec18d4abcae20d08acd992a802da4d

artifacts/primeqa_hybrid_rebuild_stage69_candidates.summary.json
sha256: a753848fe2f6c111e2a376c53522ce5ca67536d0203d5addd135f86beaa6332d
```

## Command

```powershell
python scripts\run_primeqa_hybrid_development_checks.py `
  --output artifacts\primeqa_hybrid_development_checks_stage70.json `
  --visualization-dir artifacts\primeqa_hybrid_development_checks_stage70_visuals `
  --top-k 1,5,10 `
  --bm25-k1 1.5 `
  --bm25-b 0.75
```

## Loaded Development Splits

| Split | Rows | Answerable | Unanswerable | Unique answer docs | Unique candidate docs |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 562 | 370 | 192 | 309 | 19,602 |
| dev | 121 | 76 | 45 | 71 | 5,373 |

No test split file is loaded by the Stage 70 BM25 baseline path.

## BM25 Train/Dev Baseline

Configuration:

```text
top_k_values: 1, 5, 10
k1: 1.5
b: 0.75
```

| Split | Total questions | Evaluated questions | Hit@1 | Hit@5 | Hit@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 562 | 370 | 0.4243 | 0.6054 | 0.6622 | 0.5023 |
| dev | 121 | 76 | 0.4342 | 0.6579 | 0.6974 | 0.5331 |

These are development baselines only. They are not final held-out metrics.

## Candidate Artifact Audit

Candidate artifact scan:

```text
row_count: 5993
train rows: 5006
dev rows: 987
train questions: 370
dev questions: 76
rows_with_runtime_features: 5993
rows_with_gold_labels: 5993
rows_with_test_split: 0
rows_with_forbidden_runtime_gold_keys: 0
```

Candidate checks:

```text
candidate_dataset_row_count_matches_summary: passed
candidate_rows_by_split_match_summary: passed
candidate_questions_by_split_match_summary: passed
candidate_dataset_contains_runtime_features: passed
candidate_dataset_contains_offline_gold_labels: passed
runtime_features_exclude_gold_label_keys: passed
```

The candidate JSONL contains offline gold labels for train/dev development, but
the runtime features exclude gold-label keys.

## Guard Checks

```text
development_baseline_splits_are_train_dev_only: passed
candidate_artifact_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
candidate_artifact_checks_passed: passed
final_test_metrics_not_run: passed
```

## Artifacts

These are local ignored artifacts and are not committed by git policy.

Report:

```text
artifacts/primeqa_hybrid_development_checks_stage70.json
sha256: 3d85033ac4b831fac4d2978af255d7e28c301e638fb02e0b87f97e4d2ea3e92d
```

Visualizations:

```text
artifacts/primeqa_hybrid_development_checks_stage70_visuals/stage70_primeqa_bm25_hit_at_k.svg
sha256: a2122ccc07f44621832af61d6900b17a0beee3651d5f84a06f1637c04effcd29

artifacts/primeqa_hybrid_development_checks_stage70_visuals/stage70_primeqa_bm25_mrr.svg
sha256: a8439d71ebc95fdab64f4a7a96b3924701ea0804d6a0598fcc24b76f52f7b44b

artifacts/primeqa_hybrid_development_checks_stage70_visuals/stage70_primeqa_candidate_questions_by_split.svg
sha256: 679c550f4c44254f267766f000fc2aae0678d80d398def731fab4fd172b60f67

artifacts/primeqa_hybrid_development_checks_stage70_visuals/stage70_primeqa_candidate_rows_by_split.svg
sha256: 8b4665863e52697a6d2cb80ee1a91392dd6aeb51c9ea95767b0452d5916f8bea
```

## Runtime Boundary

Stage 70 does not train a reranker, does not tune against the frozen test split,
does not run final test metrics, and does not change the default runtime policy.

## Decision

```text
status: primeqa_hybrid_train_dev_development_checks_ready
can_continue_train_dev_development: true
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 71 completed train/dev candidate-reranker policy development on
`primeqa_hybrid_stage68_v1`. The current follow-up is Stage 72: review Stage 71
train/dev candidate-reranker changed cases before considering any final-test
evaluation gate.

Stage 71 is recorded in:

```text
docs/primeqa_hybrid_candidate_reranker_development.md
```
