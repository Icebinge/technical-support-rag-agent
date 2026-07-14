# PrimeQA Hybrid Candidate Reranker Changed-Case Review

This document records Stage 72.

## Scope

Stage 72 reviews Stage 71 train/dev candidate-reranker changed cases. It uses
the frozen Stage 68 train/dev split and the Stage 69 train/dev candidate
artifact.

Stage 72 does not load the frozen test split, does not run final test metrics,
does not tune on test, and does not change the default runtime policy.

## Command

```powershell
python scripts\review_primeqa_hybrid_candidate_reranker_changed_cases.py `
  --output artifacts\primeqa_hybrid_candidate_reranker_changed_case_review_stage72.json `
  --visualization-dir artifacts\primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals `
  --models logistic_best_candidate,ridge_candidate_token_f1 `
  --max-answer-candidates 3 `
  --sample-limit 20
```

## Inputs

```text
artifacts/primeqa_hybrid_candidate_reranker_development_stage71.json
artifacts/primeqa_hybrid_rebuild_stage69_candidates.jsonl
artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_train.jsonl
artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_dev.jsonl
```

The Stage 68 test split is not an input.

## Dev Top3 Changed-Case Summary

| Model | Policy | Dev top3 delta | Regressions | Changed cases | Gold citation delta |
| --- | --- | ---: | ---: | ---: | ---: |
| logistic_best_candidate | stage36_main | +0.0001 | 1 | 21 | +0 |
| logistic_best_candidate | candidate_score_gte_60 | +0.0004 | 0 | 17 | +0 |
| ridge_candidate_token_f1 | stage36_main | +0.0003 | 1 | 8 | +0 |
| ridge_candidate_token_f1 | candidate_score_gte_60 | +0.0000 | 0 | 3 | +0 |

The best train/dev-only dev top3 proxy row remains
`logistic_best_candidate` with `candidate_score_gte_60`: delta `+0.0004`, zero
regressions, and no gold-citation loss.

## Policy-Vs-Main Review

| Model | Changed vs stage36_main | Changed rate | Candidate better/tied/worse | Avg delta vs main on changed |
| --- | ---: | ---: | --- | ---: |
| logistic_best_candidate | 4 / 76 | 0.0526 | 1 / 2 / 1 | +0.0059 |
| ridge_candidate_token_f1 | 5 / 76 | 0.0658 | 1 / 2 / 2 | -0.0039 |

The candidate-score threshold reduces changed-case count and removes residual
top3 regressions in both models, but the aggregate dev top3 gain is still very
small.

## Guard Checks

```text
candidate_artifact_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
gold_answer_splits_are_train_dev_only: passed
stage71_final_test_metrics_not_run: passed
stage72_review_uses_dev_holdout_only: passed
stage72_report_is_public_safe_no_raw_answer_text: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

The report is public-safe: it stores case IDs, route labels, ranks, scores,
document IDs, and metric deltas. It omits raw answer text and raw candidate
sentence text.

## Artifacts

These are local ignored artifacts and are not committed by git policy.

Report:

```text
artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72.json
sha256: 97e5d6d94a3ec1305e4d928168cf18c9c4e43ee63c79af9f8c8e7895e6d7577d
```

Visualizations:

```text
artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/stage72_candidate_score_vs_main_outcomes.svg
sha256: 7cafee12d9f39b9d630699fcbe613318e155178a583a629096a90c61ae0d4056

artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/stage72_changed_cases_by_policy.svg
sha256: b662fb0c62b0943ccf9f322006203453d3cf020a37444d17e0683b02c9970360

artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/stage72_dev_top3_delta_by_policy.svg
sha256: 97207e76b0e96a4829b21ea8143f5d9995fdbaace37a83de608b4ff7ea22039e

artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/stage72_residual_regressions_by_policy.svg
sha256: 76b70cbec1ca9af6dc09a3ff63f86e84eda3c8b1c5840792adeae3c9c8b4b186
```

## Decision

```text
status: primeqa_hybrid_candidate_reranker_changed_case_review_completed
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
observed_best_dev_top3_delta: +0.0004
observed_min_regressed_count: 0
```

Stage 72 does not itself open the final-test gate. The next stage must decide
whether to refine train/dev reranker gates further or explicitly approve one
one-time final-test evaluation gate. The test split still cannot be used for
tuning.

## Next Step

Stage 73 ran the requested train/dev top10 diagnostic in:

```text
docs/primeqa_hybrid_candidate_reranker_top10_diagnostic.md
```

The current next step is Stage 74: choose whether to stop reranker-policy
development as non-actionable for now, or refine train/dev reranker gates using
the top3/top10 diagnostics. The test split still cannot be used for evaluation
or tuning.
