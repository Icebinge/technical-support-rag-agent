# Stage 203 Grouped Top-1 Joint-Objective Nested CV

## Objective

Stage 203 executes the Stage 202 train-only protocol. It tests whether a
question-grouped custom LightGBM objective can jointly improve strict-action
capture, strict precision, and unsafe selection inside the frozen cap-16
candidate pool. Development and test remain closed.

## Experiment

Each of the 20 inner partitions fits the unchanged Stage 196 control models
and 16 custom grouped objectives:

```text
safety weights:                         0, 0.5, 1, 2
precision weights:                      0, 0.5, 1, 2
custom objective cells:                16
exact Stage 196 control cells:           1
candidate cells per outer context:      17
observed model fits:                   400
observed LightGBM trees:           108,000
private predictions:               983,840
outer refits:                            0
algorithmic retry or fallback:           0
```

The exact control reproduced the published Stage 199 evidence in all five
outer contexts. The custom objective validated all 340 fitted group
contracts, and its callback was invoked 96,000 times. No private training or
prediction row was written to the public report.

## Results

None of the 17 candidates passed the complete inner eligibility contract in
any outer context. Eligible counts were `0, 0, 0, 0, 0`, so the frozen
protocol prohibited outer refits and paired outer evaluation. The zero-valued
outer aggregate therefore means "not evaluated"; it is not a measured
zero-quality result.

The top candidate in every outer context was the exact Stage 196 control:

```text
outer     capture    strict precision    unsafe rate    citation delta    mean F1 delta
fold_1    0.671280        0.667845         0.311864            5             0.016462
fold_2    0.643357        0.637011         0.338983            4             0.017257
fold_3    0.647059        0.632509         0.327759           14             0.017075
fold_4    0.689046        0.663043         0.296552            8             0.022053
fold_5    0.681507        0.674825         0.295681           12             0.020727
```

The required boundaries are capture `>= 0.68`, strict precision `>= 0.65`,
and unsafe rate `<= 0.25`, together with the unchanged quality and per-fold
constraints. The control approached the capture and precision boundaries in
some folds but remained unsafe; other folds also missed capture or precision.

Across all inner contexts, the control achieved strict success count `923`,
strict precision `0.655075`, conditional capture `0.641418`, unsafe rate
`0.314189`, citation delta `+43`, and mean F1 delta `+0.018707`. The strongest
custom cell by strict count was the unpenalized strict-only objective, with
strict count `863`, precision `0.610325`, capture `0.599722`, and unsafe rate
`0.336486`. It did not improve on the control.

## Directional Diagnosis

The safety target behaved consistently: all `12/12` adjacent safety-weight
comparisons reduced or preserved unsafe rate. The precision target behaved in
the opposite direction from the final metric: all `12/12` adjacent
precision-weight comparisons reduced strict success precision. Increasing
either penalty also reduced strict capture.

This evidence rejects another wider search over the same target mixture. The
problem is not merely that the frozen weights were too small. The current
precision distribution assigns mass to both strict actions and the baseline;
in the learned ordering, increasing that mass favors abstention and changes
the selected-outcome mix instead of improving precision among changed
answers. This is a measured association from the Stage 203 grid, not a causal
claim about every possible grouped objective.

Only `4/17` final advancement gates passed. Because no candidate reached outer
evaluation, outer quality, bootstrap, coverage, and repair gates are correctly
unavailable and fail closed. Development and test were not opened.

## Decision

The formal status is `stage203_top1_joint_objective_insufficient`. The
experiment is valid, but the candidate family is rejected. Stage 203 does not
authorize full-train policy selection, replacement, runtime E2E, development,
test, or default runtime activation.

The next stage should first attribute question-level control-to-custom ranking
flips and separate baseline abstention, strict displacement, safe-zero
selection, and unsafe selection. A redesigned objective should be frozen only
after that evidence identifies whether the next model needs a separate
change/abstain head, a conditional strict-action ranker, or a different
calibration constraint. It should not reopen the same weight grid or relax the
existing gates.

## Execution Record

The first formal PID `25956` reproduced Stage 182 and completed all four
`fold_1` inner partitions, reaching 80 fits, then failed with
`KeyError: 'top_inner_evaluation'`. The implementation had assumed a flattened
Stage 199 field that the real report does not contain. No Stage 203 report was
persisted from this failed attempt.

The correction strictly locates the unique
`source_weighted_classifier + gain_only` control inside each outer context's
published `top_inner_candidates` and requires
`control_reproduction_exact=true`. A negative contract test now rejects a
missing control candidate. All five real Stage 199 contexts passed this parser.

Corrected formal PID `17584` used one PowerShell command and one
`Wait-Process` call, then waited for natural completion. The execution did not
poll the process and used no experiment timeout, algorithmic retry, fallback,
OOM recovery, or CUDA allocation. Reported wall time was `617.379649` seconds:

```text
dependency and memory authorization:    0.950319 s
Stage 182 reproduction:                220.968605 s
Stage 203 nested CV:                   395.460724 s
```

Peak working set was `3.772 GiB`, peak private usage was `4.090 GiB`, and
minimum system-available memory was `2.380 GiB`. All `38/38` process guards
passed.

## Visual Verification

All 16 SVGs were rasterized with pinned `resvg_py==0.3.3`, project-owned
Poppins fonts, a white background, and no fallback. The deterministic manifest
reports 16 nonempty PNGs with non-background pixels. The directional-response,
top-inner unsafe-rate, and advancement-gate charts were opened at original
resolution and showed no clipping or overlap.

Representative views:

- [directional responses](../artifacts/primeqa_hybrid_top1_joint_objective_stage203_visuals_png/stage203_directional_response.png)
- [top-inner unsafe rates](../artifacts/primeqa_hybrid_top1_joint_objective_stage203_visuals_png/stage203_top_inner_unsafe.png)
- [advancement gates](../artifacts/primeqa_hybrid_top1_joint_objective_stage203_visuals_png/stage203_advancement_gates.png)

Artifact hashes before final source verification:

```text
formal report SHA-256:
b675d61a2c79d9fcd74639f6a9e4caf1de3da29205018e4b34343fae79340317

resvg manifest SHA-256:
c63b2052bd98ea2eac2542f6ec0345310974f1a2b11bf615b29c640eb667b4da
```

## Current-Source Verification

Repository-wide Ruff lint, all five changed Python-file format checks,
`pip check`, CLI help, and `git diff --check` passed. The Stage 194-203 related
regression set passed `85 tests in 15.32s`.

The complete pytest suite used the single Python PID `25668`. Its PowerShell
command called `Wait-Process` exactly once and waited for natural completion,
without polling or a pytest timeout. The result was
`1236 passed, 1 warning in 44.28s`. The warning is the existing
FastAPI/Starlette `TestClient` deprecation.
