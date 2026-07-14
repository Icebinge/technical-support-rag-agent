# PrimeQA Hybrid Candidate Reranker Stop Decision

This document records Stage 74.

## Scope

Stage 74 stops the current PrimeQA hybrid candidate-reranker policy development
route as non-actionable for now.

This is a decision checkpoint. It does not run a new experiment, does not load
the frozen test split, does not run final test metrics, does not tune on test,
and does not change the default runtime policy.

## Evidence Used

Stage 71 completed train/dev-only candidate-reranker development:

```text
train-only CV best model: ridge_candidate_token_f1
ridge train-CV selected F1: 0.2652
ridge train-CV delta: +0.0383
logistic train-CV selected F1: 0.2523
logistic train-CV delta: +0.0254

logistic dev top3 best policy: candidate_score_gte_60
logistic dev top3 best delta: +0.0004
logistic dev top3 best regressions: 0

ridge dev top3 best policy: stage36_main
ridge dev top3 best delta: +0.0003
ridge dev top3 best regressions: 1
```

Stage 72 reviewed changed cases and found that the best dev top3 candidate was
still tiny:

```text
best dev top3 policy: logistic_best_candidate / candidate_score_gte_60
best dev top3 delta: +0.0004
best dev top3 regressions: 0
best dev top3 gold citation delta: +0
```

Stage 73 ran the requested top10 diagnostic on train/dev only:

```text
train-only CV top10:
  logistic stage36_main delta: +0.0000, regressions: 0
  logistic candidate_score_gte_60 delta: +0.0000, regressions: 0
  ridge stage36_main delta: +0.0000, regressions: 0
  ridge candidate_score_gte_60 delta: +0.0000, regressions: 0

dev holdout top10:
  logistic stage36_main delta: +0.0000, regressions: 0
  logistic candidate_score_gte_60 delta: +0.0000, regressions: 0
  ridge stage36_main delta: +0.0000, regressions: 0
  ridge candidate_score_gte_60 delta: +0.0000, regressions: 0
```

## Decision

```text
status: candidate_reranker_policy_route_stopped_as_non_actionable
default_runtime_policy: unchanged
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
current_reranker_policy_defaultization: blocked
```

The current reranker policy route is stopped because the only positive answer
proxy signal is the Stage 72 dev top3 delta of `+0.0004`, and that signal
disappears under the Stage 73 top10 diagnostic. This is not enough evidence to
justify runtime defaultization, more tuning on the same path, or a final-test
gate.

## Boundary After Stop

- Do not defaultize the current reranker policy.
- Do not run final test metrics for the current reranker policy.
- Do not tune the current reranker policy against the frozen test split.
- Do not treat Stage 71 train-CV gains as answer-level runtime evidence.
- Do not treat the Stage 72 top3 delta as enough to open a final-test gate.
- Keep the default runtime policy unchanged.

## If This Route Is Reopened

Reopening this route requires a new user-confirmed train/dev-only plan. That
plan should define the target surface before implementation, for example
retrieval ranking, answer composition, citation preservation, or another
specific behavior. It should also define train/dev success criteria before any
new experiment is run.

The frozen test split still cannot be used for evaluation or tuning during
development.

## Next Step

Stage 75 should select a new non-reranker-policy direction only after explicit
user confirmation. Until then, there is no active reranker-policy implementation
track.
