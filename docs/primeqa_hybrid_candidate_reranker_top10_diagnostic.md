# PrimeQA Hybrid Candidate Reranker Top10 Diagnostic

This document records Stage 73.

## Scope

Stage 73 runs a train/dev-only top10 answer proxy diagnostic for the
PrimeQA/TechQA hybrid candidate-reranker policies.

This stage uses the Stage 69 train/dev candidate artifact and the Stage 68
train/dev split files. It does not load the frozen test split, does not run
final test metrics, does not tune on test, and does not change the default
runtime policy.

## Command

```powershell
python scripts\run_primeqa_hybrid_candidate_reranker_top10_diagnostic.py `
  --output artifacts\primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73.json `
  --visualization-dir artifacts\primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals `
  --models logistic_best_candidate,ridge_candidate_token_f1 `
  --max-answer-candidates 10
```

## Inputs

```text
artifacts/primeqa_hybrid_rebuild_stage69_candidates.jsonl
artifacts/primeqa_hybrid_rebuild_stage69_candidates.summary.json
artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_train.jsonl
artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_dev.jsonl
```

The Stage 68 test split is not an input.

## Train-Only CV Top10 Proxy

Selected rows:

| Model | Policy | Delta | Policy F1 | Baseline F1 | Oracle F1 | Replacements | Improved | Regressed | Tied | Citation delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_best_candidate | stage36_main | +0.0000 | 0.1845 | 0.1845 | 0.1893 | 105 | 0 | 0 | 370 | +0 |
| logistic_best_candidate | candidate_score_gte_60 | +0.0000 | 0.1845 | 0.1845 | 0.1893 | 77 | 0 | 0 | 370 | +0 |
| ridge_candidate_token_f1 | stage36_main | +0.0000 | 0.1845 | 0.1845 | 0.1893 | 40 | 0 | 0 | 370 | +0 |
| ridge_candidate_token_f1 | candidate_score_gte_60 | +0.0000 | 0.1845 | 0.1845 | 0.1893 | 25 | 0 | 0 | 370 | +0 |

In train-only CV top10 proxy, policy replacements do not change answer token F1.

## Dev Holdout Top10 Proxy

Selected rows:

| Model | Policy | Delta | Policy F1 | Baseline F1 | Oracle F1 | Replacements | Improved | Regressed | Tied | Citation delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_best_candidate | stage36_main | +0.0000 | 0.1933 | 0.1933 | 0.1947 | 21 | 0 | 0 | 76 | +0 |
| logistic_best_candidate | candidate_score_gte_60 | +0.0000 | 0.1933 | 0.1933 | 0.1947 | 17 | 0 | 0 | 76 | +0 |
| ridge_candidate_token_f1 | stage36_main | +0.0000 | 0.1933 | 0.1933 | 0.1947 | 8 | 0 | 0 | 76 | +0 |
| ridge_candidate_token_f1 | candidate_score_gte_60 | +0.0000 | 0.1933 | 0.1933 | 0.1947 | 3 | 0 | 0 | 76 | +0 |

In dev holdout top10 proxy, all policies tie the baseline. The observed best
dev top10 delta is `+0.0000`, with zero regressions and zero citation delta.

## Interpretation

Stage 72 top3 showed a tiny dev proxy signal for
`logistic_best_candidate / candidate_score_gte_60`:

```text
top3 dev delta: +0.0004
top3 regressions: 0
top3 gold citation delta: +0
```

Stage 73 top10 removes even that tiny signal:

```text
top10 dev delta: +0.0000
top10 regressions: 0
top10 gold citation delta: +0
```

This suggests the current reranker policy only affects narrow leading-candidate
ordering. When the answer proxy includes ten candidates, the composed answer
surface becomes effectively unchanged by the reranker policy.

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
stage73_topk_window_matches_requested_value: passed
stage73_split_validations_are_train_to_dev: passed
stage73_final_test_metrics_not_run: passed
stage73_default_runtime_policy_unchanged: passed
```

## Artifacts

These are local ignored artifacts and are not committed by git policy.

Report:

```text
artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73.json
sha256: 0225802984f20ec52a98726390ff17b82f4900f531ef14ca629af950ccf93822
```

Visualizations:

```text
artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals/
svg files: 20

artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals/stage73_logistic_best_candidate_dev_holdout_top10_policy_delta.svg
sha256: db0732c6f3ea80c1633717ea04cd40e373ab4fc4a72b96602a5f9f674e0e513b

artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals/stage73_logistic_best_candidate_dev_holdout_top10_policy_regressions.svg
sha256: dcad8d375c3299e8376e35eed269141b4c1241d45cf182987156ada0a46fbf93

artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals/stage73_ridge_candidate_token_f1_dev_holdout_top10_policy_delta.svg
sha256: 35f0b0da41389c715659e7dfaa7ed1e9196b8f7eefd00e6f3f83d6f81efef3a9

artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals/stage73_ridge_candidate_token_f1_dev_holdout_top10_policy_regressions.svg
sha256: 1349deaecd0ecab7b71070cca3eeb1b74f5c407817f22e214b565557657fd777
```

The report contains the complete visualization manifest for all 20 SVG files.

## Decision

```text
status: primeqa_hybrid_candidate_reranker_topk_diagnostic_completed
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
observed_best_dev_topk_delta: +0.0000
observed_min_regressed_count: 0
```

Stage 73 does not open the final-test gate. The top10 result weakens the case
for defaultizing the current reranker policy because the broader answer proxy
shows no train/dev F1 gain.

## Next Step

Stage 74 should choose whether to stop reranker-policy development as
non-actionable for now, or refine train/dev reranker gates using the top3/top10
diagnostics. The test split still cannot be used for evaluation or tuning.
