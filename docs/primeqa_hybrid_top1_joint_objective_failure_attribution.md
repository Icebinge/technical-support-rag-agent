# Stage 204 Top-1 Joint-Objective Failure Attribution

## Objective

Stage 204 executes the user-confirmed route A: reproduce Stage 203 once and
stream every private control-to-custom and adjacent-weight winner comparison
into public aggregate diagnostics. It does not search another model, reopen
the same weight grid, relax a constraint, or use development or test data.

## Diagnostic Population

The Stage 203 report SHA-256 matched the frozen source and all 38 source
process guards passed authorization. The rerun reproduced the complete stable
Stage 203 evidence, including 400 fits, 108,000 trees, and 983,840 private
predictions.

```text
outer contexts:                              5
outer cell contexts:                        85
custom outer cell contexts:                 80
question contexts:                       1,480
control-to-custom question comparisons: 23,680
adjacent precision comparisons:          17,760
adjacent safety comparisons:             17,760
additional diagnostic fits:                   0
persisted question or prediction rows:        0
```

Every comparison required identical question keys and identical candidate
pools. Control, custom, and transition outcome partitions were exact. All
39/39 Stage 204 process guards passed.

## Control-To-Custom Flips

Across all 16 custom objectives, `20,186 / 23,680` winners changed relative to
the Stage 196 control, a flip rate of `0.852449`:

```text
strict gains:             1,257
strict losses:            9,312
net strict:              -8,055

unsafe repairs:           5,246
unsafe regressions:       1,515
net unsafe:              -3,731

baseline additions:      12,164
baseline removals:           39
net baseline:            12,125
```

The objectives do remove unsafe winners, but they remove many more strict
winners and overwhelmingly replace changed answers with the baseline. This is
not a missing-opportunity problem: the cap-16 pool contains a mean of
`8.043919` strict actions per question context, and `1,439 / 1,480` contexts
have at least one strict opportunity.

## Precision-Weight Mechanism

Across the 60 outer-cell adjacent precision comparisons, only one had
nondecreasing strict precision and none had nondecreasing conditional strict
capture. All 60 reduced unsafe rate.

At question level, increasing precision weight produced:

```text
winner flips:             4,784 / 17,760 = 0.269369
strict gains / losses:      280 / 2,214
unsafe repairs / regressions: 1,349 / 244
baseline additions / removals: 3,251 / 97

net strict:              -1,934
net unsafe:              -1,105
net baseline:             3,154
```

The most frequent unchanged transition was `baseline -> baseline`. The
dominant outcome change was `strict_success -> baseline`, occurring 1,988
times. The next largest relevant change was `unsafe_f1_only -> baseline`, at
1,034. Precision weighting therefore abstains on both unsafe and strict
answers, but displaces substantially more strict answers.

Mean outer-cell deltas for an adjacent precision increase were:

```text
strict precision:             -0.089587
conditional strict capture:   -0.111922
unsafe rate:                  -0.062215
```

This is measured association under the Stage 203 grid, not a universal causal
claim about all grouped objectives.

## Safety-Weight Mechanism

Safety weighting has the same structural tendency. Its 17,760 adjacent
question comparisons yielded net strict `-1,382`, net unsafe `-843`, and net
baseline `+2,313`. All 60 outer-cell comparisons reduced unsafe rate, but only
one preserved strict precision and none preserved capture. A single grouped
softmax is trading answer selection against abstention instead of solving
these decisions independently.

## Target Mechanics

The precision component is mathematically normalized in every question. Its
mean target mass is `0.161709` on baseline and `0.838291` across all strict
actions. The safety component assigns mean baseline mass `0.135160`. These
component masses are valid; the failure is their joint use in one winner
distribution. Baseline is the only always-present safe action and competes
directly with every answer-bearing action.

## Decision

The formal status is
`stage204_top1_joint_objective_failure_attribution_complete`. The diagnostic
is valid and recommends freezing a separate change/abstain head plus a
conditional strict-action ranker. Stage 204 does not authorize a new model
search, same-grid search, gate relaxation, full-train selection, replacement,
runtime E2E, development, test, or default activation.

The next protocol should train the change gate and within-change ranker on
separate targets and evaluate their errors separately. Baseline must not
compete inside the conditional action-ranking softmax. Existing safety,
quality, nested-CV, and privacy constraints should remain unchanged.

## Execution Record

Formal PID `9332` ran in one PowerShell command with one `Wait-Process` call
and waited for natural completion. There was no polling, experiment timeout,
retry, fallback, OOM, or CUDA allocation. Total wall time was `639.810270`
seconds. Peak working set was `3.761 GiB`, peak private usage was `4.092 GiB`,
and minimum system-available memory was `2.222 GiB`.

After the run, a deterministic presentation correction split the ambiguous
`dominant_precision_transition` field into the most frequent overall
transition and the dominant changing transition. It recomputed only this
derived finding from persisted transition counts; no model, prediction,
population, metric, guard, or visualization was rerun or changed.

## Visual Verification

All 12 SVGs were rasterized with pinned `resvg_py==0.3.3`, project-owned
Poppins fonts, a white background, and no fallback. The manifest reports 12
nonempty PNGs with non-background pixels. Precision flips, candidate strict
losses, target mass, and all 39 process guards were opened at original
resolution and showed no clipping or overlap.

Representative views:

- [precision adjacent flips](../artifacts/primeqa_hybrid_top1_joint_objective_failure_attribution_stage204_visuals_png/stage204_precision_adjacent_flips.png)
- [candidate strict losses](../artifacts/primeqa_hybrid_top1_joint_objective_failure_attribution_stage204_visuals_png/stage204_candidate_strict_losses.png)
- [target mass](../artifacts/primeqa_hybrid_top1_joint_objective_failure_attribution_stage204_visuals_png/stage204_target_mass.png)
- [process guards](../artifacts/primeqa_hybrid_top1_joint_objective_failure_attribution_stage204_visuals_png/stage204_process_guards.png)

Artifact hashes after deterministic derived-finding correction:

```text
formal report SHA-256:
3757cc7a84a7a70beddd151228fbe39157fa546db2ce11da4996e66eefd19fe8

resvg manifest SHA-256:
5d8c1c5a06b2610dacbde666c514c079319faa853ecfb217019b4385d05b6a33
```

## Current-Source Verification

Repository-wide Ruff lint, all eight changed Python-file format checks,
`pip check`, CLI help, and `git diff --check` passed. The Stage 194-204 related
regression set passed `94 tests in 14.51s`.

The complete pytest suite used the single Python PID `27276`. Its PowerShell
command called `Wait-Process` exactly once and waited for natural completion,
without polling or a pytest timeout. The result was
`1245 passed, 1 warning in 44.82s`. The warning is the existing
FastAPI/Starlette `TestClient` deprecation.
