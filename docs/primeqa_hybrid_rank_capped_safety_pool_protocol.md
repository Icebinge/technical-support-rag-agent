# Stage 190 Rank-Capped Safety-Pool Protocol Freeze

## Objective

Stage 190 freezes a new train-only nested-CV protocol in response to the
Stage 189 finding that safety-frontier exclusion is the dominant bottleneck.
This stage reads only the corrected public Stage 189 aggregate report and
performs no training, prediction, policy evaluation, development/test access,
or runtime E2E.

## Frozen Candidate Pool

For every question, the two frozen Stage 188 safety heads produce:

```text
joint safety risk = max(p(citation loss), p(F1 loss))
```

Actions are ordered by ascending joint risk, then ascending summed risk, then
canonical runtime action order. Stage 191 will test pool caps `4`, `8`, `16`,
and `all`; the unique original baseline is always unioned into the pool. The
pool cannot be empty and has no fallback branch.

Inside each pool, Stage 191 will reuse the Stage 188 gain rankers and select by
descending gain score, ascending joint risk, and canonical action order. All
Stage 188 features, models, targets, hyperparameters, full pair construction,
and full listwise construction remain frozen.

## Cross-Validation And Grid

The grid contains two feature representations, two safety estimators, two gain
rankers, and four pool caps:

```text
2 * 2 * 2 * 4 = 32 policy configurations
20 inner partitions + 5 possible outer refits
12 model fits per partition
300 maximum model fits
```

Inner selection uses only inner OOF predictions. In addition to all Stage 188
eligibility constraints, a configuration must satisfy:

```text
aggregate strict-opportunity pool recall >= 0.80
per-fold strict-opportunity pool recall >= 0.70 in at least 3/4 folds
```

If no configuration is eligible, the outer fold records no-eligible and does
not substitute a weaker candidate. The final advancement protocol has 15
gates: all 14 Stage 188 gates plus held-out strict-opportunity pool recall of at
least `0.80`.

## Formal Freeze

The protocol was frozen from the exact corrected Stage 189 report:

```text
Stage 189 SHA-256:
48af548168e4e40972c4082fc24bec822ce264427f12c56b98a8d0966df2e5a0

Stage 190 SHA-256:
6558798d6cee0cedb7b01fb864cda749e3f8e63793535ce764acbfaabbb6e07b
```

All 34 protocol guards passed. Stage 191 train-only experimentation is
authorized, but development, test, runtime E2E, full-train policy selection,
replacement policy selection, Stage 178B, fallback, and default runtime
activation remain unauthorized. Freezing took `0.002538` seconds and loaded no
training rows or private predictions.

All eight protocol SVGs were deterministically rasterized and opened at
original resolution. Candidate-grid, budget, source evidence, recall gates,
decision flags, and all guard names were clear without clipping or overlap.
The rasterization manifest SHA-256 is:

```text
dc945b86dd9e437c43d05562a488e74cca8be4dff306067897b1e00edb0d6d07
```

## Next Stage

Stage 191 may now execute the frozen rank-capped safety-pool nested CV on the
training split only. It must measure candidate-pool recall before interpreting
the downstream ranker's quality, and it must preserve every frozen gate and
authorization boundary exactly.

## Current-Source Verification

The Stage 184-190 related regression suite passed `41 passed in 4.68s`.
Repository-wide Ruff lint passed, all 11 changed Python files passed Ruff
format check, and the complete repository suite passed:

```text
1131 passed, 1 warning in 23.21s
```

The full suite used PID `26576` and exactly one `Wait-Process` call until
natural completion. The warning is the pre-existing FastAPI/Starlette
`TestClient` deprecation.
