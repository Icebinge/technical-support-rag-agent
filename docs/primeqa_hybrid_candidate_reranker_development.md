# PrimeQA Hybrid Candidate Reranker Development

This document records Stage 71: train/dev-only candidate-reranker development
for the frozen PrimeQA/TechQA hybrid split.

Stage 71 uses `primeqa_hybrid_stage68_v1`. It runs train-only grouped
cross-validation over candidate-reranker models, then train-to-dev guarded
policy validation for both candidate models. It keeps the frozen test split
locked, does not run final metrics, and does not change the default runtime.

## Input Contract

```text
split_name: primeqa_hybrid_stage68_v1
protocol_version: primeqa_hybrid_split_v1
train_split: train
development_evaluation_split: dev
forbidden_final_splits: test
```

Source files:

```text
artifacts/primeqa_hybrid_rebuild_stage69_candidates.jsonl
sha256: d379d59f5172394a40bcd1852aa8188f2dec18d4abcae20d08acd992a802da4d

artifacts/primeqa_hybrid_rebuild_stage69_candidates.summary.json
sha256: a753848fe2f6c111e2a376c53522ce5ca67536d0203d5addd135f86beaa6332d

artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_train.jsonl
sha256: cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155

artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_dev.jsonl
sha256: 071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f
```

## Command

```powershell
python scripts\run_primeqa_hybrid_candidate_reranker_development.py `
  --output artifacts\primeqa_hybrid_candidate_reranker_development_stage71.json `
  --visualization-dir artifacts\primeqa_hybrid_candidate_reranker_development_stage71_visuals `
  --fold-count 5 `
  --models logistic_best_candidate,ridge_candidate_token_f1 `
  --max-answer-candidates 3
```

## Loaded Data

Candidate rows:

```text
total rows: 5993
train rows: 5006
dev rows: 987
train questions: 370
dev questions: 76
test rows: 0
```

Gold answers are loaded from the frozen Stage 68 train/dev split JSONL files and
keyed as `split::sample_id`, matching the Stage 69 candidate `question_id`
contract.

```text
gold train answers: 370
gold dev answers: 76
```

No test split file is loaded by the Stage 71 run.

## Train-Only Model CV

Configuration:

```text
fold_count: 5
models: logistic_best_candidate, ridge_candidate_token_f1
```

| Model | Baseline F1 | Selected F1 | Delta | Oracle gap closed | Improved | Regressed | Tied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_best_candidate | 0.2269 | 0.2523 | +0.0254 | 0.1344 | 120 | 88 | 162 |
| ridge_candidate_token_f1 | 0.2269 | 0.2652 | +0.0383 | 0.2025 | 120 | 87 | 163 |

The train-only CV best model is `ridge_candidate_token_f1`. This is still
development evidence only.

## Train-To-Dev Policy Validation

Stage 71 validates fixed guarded policies for both models. The answer proxy uses
two modes:

```text
single_candidate_answer
top3_leading_candidate_rewrite
```

The top3 rewrite proxy recomputes answer token F1 from local gold answers and
candidate metadata sentences. It is not a verified RAG runtime metric.

### Dev Holdout Top3 Proxy

| Model | Best top3 policy | Best top3 delta | Best top3 regressions | Best top3 citation delta |
| --- | --- | ---: | ---: | ---: |
| logistic_best_candidate | candidate_score_gte_60 | +0.0004 | 0 | +0 |
| ridge_candidate_token_f1 | stage36_main | +0.0003 | 1 | +0 |

The dev top3 proxy improvements are very small. This supports another
changed-case review stage, not final-test evaluation or runtime defaultization.

### Dev Holdout Single-Candidate Proxy

| Model | Policy | Delta | Replacements | Regressions | Gold citation delta |
| --- | --- | ---: | ---: | ---: | ---: |
| logistic_best_candidate | stage36_main | +0.0436 | 21 | 4 | +7 |
| logistic_best_candidate | candidate_score_gte_60 | +0.0337 | 17 | 4 | +6 |
| ridge_candidate_token_f1 | stage36_main | +0.0144 | 8 | 0 | +3 |
| ridge_candidate_token_f1 | candidate_score_gte_60 | +0.0027 | 3 | 0 | +2 |

The single-candidate proxy shows larger gains than the top3 rewrite proxy, so it
should not be treated as sufficient by itself.

## Guard Checks

```text
candidate_artifact_splits_are_train_dev_only: passed
candidate_summary_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
gold_answer_splits_are_train_dev_only: passed
train_cv_uses_train_only: passed
split_validations_are_train_to_dev: passed
candidate_artifact_checks_passed: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

## Artifacts

These are local ignored artifacts and are not committed by git policy.

Report:

```text
artifacts/primeqa_hybrid_candidate_reranker_development_stage71.json
sha256: bb5c665295a7cd0768e8c69d805c0dd60c5fdfb5839aba4cd77f7161c35a4573
```

Visualizations:

```text
artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/candidate_reranker_model_delta.svg
sha256: ef1bdf5408b008360ba1beeeb9015dbad1d289a62cdd62998b21d9887ed0e19f

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/candidate_reranker_model_gap_closed.svg
sha256: 1ea1fca9730233a86aecd321840b11c788332a41b88039fb55c063769f6bce97

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/candidate_reranker_best_model_route_delta.svg
sha256: c4ab78565e140f4d1aa3c09dd930d6077bf3f6196e88d09b449dc7b46e4e9bb8

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/candidate_reranker_best_model_selected_rank.svg
sha256: bdd243122e5482eb2cb3c8825432d0ce29f1321363b7731df89a33d69bd4f241

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/stage71_logistic_best_candidate_dev_holdout_top3_policy_delta.svg
sha256: 56b1fb622bf6ca7ad82e0071442241c3b193bae44d11d6fa2075f11e7eedb689

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/stage71_logistic_best_candidate_dev_holdout_top3_policy_regressions.svg
sha256: 821a4a7efdce42fe5445bc6c7f992a405c78cdcf867efc49ec47a9fa56b67352

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/stage71_logistic_best_candidate_dev_holdout_top3_policy_citation_exchange.svg
sha256: cd96f849c5a7d0cd866e9a062db4d794823c6be7366106cdf33a52d65d6dcf97

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/stage71_logistic_best_candidate_dev_holdout_single_candidate_policy_delta.svg
sha256: fb05746271d50301e47a3afdd4cb4cf227fb64952f43876b397473760da36bdd

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/stage71_ridge_candidate_token_f1_dev_holdout_top3_policy_delta.svg
sha256: 79bb7896fe2d7df4ebe01e77d25baeb1c7a7a12f5b1d6c97bb9bdfd7aa8f23a8

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/stage71_ridge_candidate_token_f1_dev_holdout_top3_policy_regressions.svg
sha256: 7146cdf182a278b25db5ced413dc81c13132a8d2befb536de8ffb120bf3c2cb7

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/stage71_ridge_candidate_token_f1_dev_holdout_top3_policy_citation_exchange.svg
sha256: b46c0db91f5948a36744b6a88765cfb781e5822704ce8076e7c31adf5f138e51

artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/stage71_ridge_candidate_token_f1_dev_holdout_single_candidate_policy_delta.svg
sha256: 2e816744537cea4f6ff262f603790c6c0b8fb05765ff91276c86e54599044c40
```

The report contains the complete visualization manifest for all 20 SVG files.

## Runtime Boundary

Stage 71 does not train a production model, does not tune against the frozen test
split, does not run final test metrics, and does not change the default runtime
policy.

## Decision

```text
status: primeqa_hybrid_candidate_reranker_development_ready
can_continue_train_dev_development: true
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 72 should review Stage 71 train/dev candidate-reranker policy changed
cases before considering any final-test evaluation gate.
