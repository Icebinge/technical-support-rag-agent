# Stage 196 Safety-First Frontier Nested CV

## Objective

Stage 196 executes the train-only experiment frozen by Stage 195. It tests
whether cost-sensitive unsafe classifiers and deterministic safest-prefix
filtering can reduce unsafe winner selection to `<= 0.25` while retaining
conditional strict capture `>= 0.68` and strict precision `>= 0.65`.

Development and test data remained closed. The experiment did not select a
full-train policy, run runtime E2E, choose a replacement, activate a default,
add fallback or retry behavior, or weaken any Stage 195 threshold.

## Implementation

The Stage 194 cap-16 safety pool and LambdaMART gain ranker were retained. Gain
and risk representations and tree profiles were independently crossed. Unsafe
heads used `scale_pos_weight` values `1.0`, `2.0`, and `4.0`; only their
within-question ordering was consumed.

Each complete pool was sorted by unsafe score. Prefixes `2, 4, 8, 12, 16`
were evaluated, then the unique baseline was unioned. The winner was selected
by descending LambdaMART score, ascending unsafe score, and canonical action
order. No absolute probability threshold or gain-risk utility blend was used.

The implementation fits and releases one weighted unsafe model at a time.
Each representation-partition fits four pool-safety heads, two LambdaMART
rankers, and six unsafe heads. Across both representations this is 24 fits per
partition.

## Verification Before Formal Run

The first core test run reported `3 passed, 1 failed`: its baseline was already
inside cap 16, so the expected post-union pool size of 17 was incorrect. The
fixture was corrected to place baseline outside the cap.

The first real LightGBM smoke fit stopped before fitting because the synthetic
data had no citation-loss class. After adding that class, the second smoke run
found an implementation error: the histogram wrapper expected a sparse heldout
matrix and performs its own dense conversion. Passing the original sparse
matrix fixed the contract. The final `-W error` smoke fit completed 12 fits and
2,400 trees with all six weighted unsafe prediction groups.

The preflight then verified every input, exact Stage 195 and wheel SHA-256,
`lightgbm==4.7.0`, `narwhals==2.24.0`, CLI help, and `pip check`. A handwritten
technotes path was initially wrong and stopped before PID creation; the second
preflight used `ProjectSettings.primeqa_raw_dir`. The process-level preflight
observed `5.100 GiB` system-available memory, above the frozen 4.0 GiB minimum.

## Formal Execution

The formal run used Python PID `15472`. One PowerShell command called
`Wait-Process` once for that PID and waited for natural completion. The command
did not poll or set an experiment timeout. Stderr contained only model-weight
loading progress bars and no exception.

```text
Stage 182 reproduction:          214.262892 s
Stage 196 nested CV:             673.629911 s
total wall:                      888.733349 s

inner partitions completed:     20 / 20
outer refits/evaluations:         0 / 5
model fits:                     480 / 600
pool safety / LambdaMART:       160 / 80
unsafe-head fits:               240
LightGBM trees:              96,000
private predictions:        983,840
process guards:                  35 / 35 passed
```

All five outer contexts had zero inner-eligible configurations, so the protocol
correctly skipped every outer refit. Aggregate and bootstrap fields are
therefore unavailable; stored aggregate zeros mean `not evaluated`, not zero
model performance.

## Top-Inner Results

| Fold | Prefix | Weight | Pool recall | Frontier recall | Capture | Precision | Unsafe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 2.0 | 0.993127 | 0.958763 | 0.671280 | 0.667845 | 0.311864 |
| 2 | 8 | 4.0 | 0.986207 | 0.948276 | 0.643357 | 0.637011 | 0.338983 |
| 3 | 4 | 4.0 | 0.986348 | 0.849829 | 0.647059 | 0.632509 | 0.327759 |
| 4 | 12 | 1.0 | 0.989510 | 0.972028 | 0.689046 | 0.663043 | 0.296552 |
| 5 | 4 | 2.0 | 0.986486 | 0.864865 | 0.681507 | 0.674825 | 0.295681 |

Fold 4 and fold 5 passed the aggregate capture and precision thresholds, but
their unsafe winner rates remained about `0.296`, above `0.25`. Prefix 4 in
folds 3 and 5 retained only `0.164577` and `0.163233` of unsafe pool actions,
yet unsafe winner rates stayed at `0.327759` and `0.295681`. This is evidence
that the remaining failure is not simply excessive unsafe-action retention:
the gain ranker disproportionately promotes some unsafe actions that survive
the risk prefix.

## Resources

```text
process peak working set:  3.761 GiB
process peak private use:  4.116 GiB
minimum system free:       2.960 GiB
CUDA allocated/reserved:   0 / 0
OOM:                       none
```

The complete grid naturally stopped at 480 fits because no outer context had
an eligible configuration. This was not a grid reduction or partial retry.

## Visual Verification

The initial charts truthfully showed `not evaluated` for all outer metrics, but
did not expose the available top-inner evidence. Without rerunning any model,
the visualization layer was changed to render top-inner pool recall, frontier
recall, capture, unsafe rate, unsafe retention, prefix, and positive-class
weight from the existing formal report. The initial generated directory was
retained; it was not presented as the final visualization set.

All 15 final SVGs passed XML parsing and were rasterized by pinned
`resvg_py==0.3.3` with project Poppins fonts and no font fallback. Every PNG was
opened at original resolution and had legible titles, labels, values, 17 gates,
and 35 process guards without overlap or clipping.

```text
formal report SHA-256:
e5a44fbc76acaa053ca809174b1f3f767afe31a4d55e0e05d0b6708aee41fa01

final resvg manifest SHA-256:
dfe7f383de182536a497e32923a6dcc847e5614ed1a50162e0ed0ba28f51c79e
```

## Decision

The formal status is `stage196_safety_first_frontier_insufficient`. The
experiment is valid, but the candidate family is not accepted. Development,
test, full-train selection, runtime E2E, replacement selection, and default
activation remain unauthorized.

The next train-only stage should attribute the surviving unsafe winners by
their unsafe-rank position, gain-rank position, outcome type, prefix, and risk
head. It should determine whether the next global change belongs in the risk
model, the final constrained winner rule, or both before another model grid is
frozen.

## Current-Source Verification

The Stage 193-196 related regression set passed `24 tests in 10.81s`.
Repository-wide Ruff lint, five changed Python-file format checks, `pip check`,
CLI help, and `git diff --check` passed.

The complete pytest suite used the single Python PID `3756`. Its PowerShell
command called `Wait-Process` exactly once and waited for natural completion
without polling or a pytest timeout:

```text
1173 passed, 1 warning in 35.17s
```

The warning remains the existing FastAPI/Starlette `TestClient` deprecation.
