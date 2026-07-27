# Stage 200 Joint Risk/Winner Failure-Attribution Protocol

## Objective

Stage 200 freezes the Stage 201 train-only diagnostic required after every one
of the 28 Stage 199 risk-signal and winner-rule cells failed inner eligibility
in all five outer contexts. The next run must identify the structural blockers
before another model, objective, or representation is designed.

This stage is an aggregate-only protocol freeze. It reads the public Stage 199
report and does not load split rows or documents, import LightGBM, fit a model,
materialize private predictions, evaluate a new policy, relax a constraint, or
open development or test.

## Source Evidence

The exact Stage 199 report has status
`stage199_joint_risk_winner_insufficient`, all `34/34` source process guards
passed, and all five exact controls reproduced. The source population contains
28 cells, four risk signals, seven winner rules, and 1,480 inner-OOF question
contexts. Eligible cell counts are `0, 0, 0, 0, 0`.

The source top-inner range confirms the unresolved joint constraint:

```text
conditional strict capture:  0.647059 to 0.685512  (required >= 0.68)
unsafe selection rate:        0.272414 to 0.327759  (required <= 0.25)
```

Stage 199 did not persist private action or prediction rows. Stage 201 must
therefore reproduce the same inner predictions rather than infer detailed
failure causes from incomplete public aggregates.

## Diagnostic Populations

Stage 201 covers every cell, not only the five top-inner candidates:

```text
outer-context x policy-cell units:       140
inner-fold x outer-cell units:            560
inner-OOF question contexts:            1,480
question-context x policy-cell units:  41,440
```

Question-cell rows remain private and are reduced to streaming aggregates.
Question text, question identifiers, document identifiers, action identifiers,
feature rows, and prediction rows are forbidden in the public report.

## Constraint Attribution

All 13 Stage 199 eligibility constraints retain their exact thresholds. Every
constraint receives a signed margin where nonnegative means pass. Lower bounds
use `observed - threshold`; upper bounds use `threshold - observed`.

Required outputs include failure prevalence by constraint and factor, pairwise
co-failure counts and Jaccard values, exact failed-set frequencies,
failed-constraint-count distribution, margin quantiles, near-boundary counts,
single-constraint-removal pass counts, and capture/unsafe Pareto counts.

Single-constraint removal is diagnostic only. It measures whether a constraint
is a necessary blocker for a cell; it does not authorize threshold relaxation,
candidate promotion, or runtime behavior.

## Fold Attribution

The 560 fold-cell units retain gold-citation delta, mean F1 delta, pool recall,
conditional strict capture, and unsafe selection rate. Stage 201 must report
violation and worst-fold frequencies, cross-fold range and standard deviation,
and cells that pass an aggregate constraint while failing its required number
of per-fold checks. Fold IDs may be public; fold membership remains private.

## Question-Context Attribution

The selected outcome partition is mutually exclusive:

```text
baseline
strict_success
safe_zero
unsafe_citation_only
unsafe_f1_only
unsafe_citation_and_f1
```

Strict-opportunity outcomes are partitioned into no opportunity, safety-pool
exclusion, risk-frontier exclusion, winner-selection miss, and strict selected.
The diagnostic aggregates selected and best-strict gain/risk ranks, rank gaps,
lower-risk strict alternatives, higher-gain strict alternatives, pool/frontier
size, opportunity count, unsafe candidate count, action family, outer context,
risk signal, and winner rule.

Gold labels and oracle comparisons are diagnostic only. They cannot become a
runtime rule, policy filter, or selection feature.

## Execution Budget

Stage 201 exactly rebuilds the Stage 199 inner experiment:

```text
inner partitions:                        20
model fits per partition:                 5
exact model fits:                       100
exact LightGBM trees:                18,000
exact private predictions:          245,960
outer refits:                             0
additional diagnostic model fits:         0
```

The 28 cells share each partition's predictions. Diagnostics are streamed while
one model bundle is materialized at a time. The memory preflight remains 4 GiB.
The formal process must use one PowerShell `Wait-Process` call for one PID until
natural completion, without polling, experiment timeout, retry, fallback, or a
reduced diagnostic population.

## Authorization

Formal status is
`stage200_joint_risk_winner_failure_attribution_protocol_frozen`. Only Stage
201 train-only attribution is authorized. Development, test, new policy search,
constraint relaxation, full-train selection, replacement selection, runtime
E2E, Stage 178B, and default activation remain unauthorized.

The formal freeze loaded only the Stage 199 public report, completed `66/66`
guards, and took `0.002464` seconds. It performed zero model fits, private-row
materializations, predictions, oracle diagnostics, retries, and fallbacks.

## Visual Verification

All 10 SVGs passed XML parsing and were rasterized with pinned
`resvg_py==0.3.3`, project-owned Poppins fonts, a white background, and no font
fallback. Every PNG was opened at original resolution. The 66 guard rows,
13-constraint catalog, authorization boundary, zero execution boundary, source
ranges, and budgets are visible without clipping or overlap. Constraint chart
values `1-13` encode frozen order, not threshold magnitude; exact thresholds
and margin formulas live in the JSON protocol.

```text
formal report SHA-256:
9edf5f3ba725bba501ebe0325ae1a072288a219e5ef655a932ae7722fcf2cf32

resvg manifest SHA-256:
bd6bd8170e705eb296f7585cc2a93335348e2e474480f31fcb193a971c387097
```

## Current-Source Verification

Repository-wide Ruff lint, all three changed Python-file format checks,
`pip check`, CLI help, and `git diff --check` passed. The Stage 197-200 focused
regression set passed `29 tests in 2.11s`.

The complete pytest suite used the single Python PID `13636`. Its PowerShell
command called `Wait-Process` exactly once and waited for natural completion,
without polling or a pytest timeout. The result was
`1203 passed, 1 warning in 39.73s` with exit code `0`. The warning is the
existing FastAPI/Starlette `TestClient` deprecation.
