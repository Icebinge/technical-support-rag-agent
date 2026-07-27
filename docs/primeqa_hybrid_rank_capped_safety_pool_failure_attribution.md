# Stage 192 Rank-Capped Safety-Pool Failure Attribution

## Objective

Stage 192 identifies where Stage 191 loses strict opportunities after the
high-recall safety-ranked candidate pool was introduced. It exactly reproduces
the frozen Stage 191 train-only nested CV and streams one diagnostic snapshot
per outer context. Development and test remain unopened.

For each strict-opportunity question context, the diagnostic assigns exactly
one outcome:

```text
candidate pool exclusion
retained in the pool but missed by the gain ranker
strict action selected
```

The attribution covers all 32 frozen combinations of feature representation,
safety estimator, gain ranker, and pool cap. It fits no additional model and
persists no action-level or prediction-level private rows.

## Method

The Stage 191 nested runner now exposes an optional immutable inner-OOF
snapshot after each outer context. Both policy evaluation and attribution use
the same candidate-pool and winner builders, so the diagnostic does not
reimplement the algorithm. Snapshots are consumed and reduced immediately to
public-safe counts. Snapshot construction freezes the inner-fold identifiers,
prediction sequences, and bundle mapping so a diagnostic consumer cannot
mutate the evaluated private state.

The reference trajectory follows the selected eligible Stage 191
configuration when one exists. Fold 2 had no eligible configuration, so its
reference is the deterministic top-ineligible configuration. This trajectory
is a diagnostic summary of five train-side inner-OOF contexts; it is not a
deployable policy result or a replacement for complete outer evaluation.

The formal run reproduced all 15 Stage 191 checks, its 288 model fits, and the
frozen outputs. Attribution consumed 393,536 private bundle predictions,
performed zero new fits, and wrote zero public action or prediction rows.

## Primary Result

Across 1,480 reference-trajectory question contexts, 1,456 had at least one
strict opportunity. The exact partition was:

```text
candidate pool exclusion:                 58
retained but missed by the gain ranker:   500
strict action selected:                   898
total strict opportunities:              1456

1456 = 58 + 500 + 898
```

The corresponding rates were:

```text
strict-opportunity pool recall:           0.960165
conditional ranker strict capture:        0.642346
actual strict-opportunity capture:        0.616758
baseline-change strict precision:         0.611300
unsafe selection rate:                    0.352703
strict-action retention rate:             0.356210
mean pool size:                           11.241892
baseline inclusion rate:                  1.000000
```

Of the 500 within-pool misses, the selected winner was safe-zero in 35 cases
and unsafe in 465 cases. The primary bottleneck is therefore
`within_pool_ranker_miss`, with a substantial safety-discrimination problem;
further enlarging the first-stage pool is not the next priority.

The selected reference actions had aggregate citation delta `+18`, mean F1
delta `+0.016210`, 45 citation-loss contexts, and 498 F1-regression contexts.
These are train-side diagnostic aggregates, not held-out deployment claims.

## Outer Contexts

| Context | Reference configuration | Eligible | Opportunities | Excluded | Ranker miss | Selected | Pool recall | Conditional capture | Unsafe rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold 1 | question-relative / histogram / pairwise / cap 16 | yes | 291 | 2 | 113 | 176 | 0.993127 | 0.608997 | 0.308475 |
| fold 2 | raw / histogram / ListNet / cap 8 | no, top-ineligible | 290 | 15 | 97 | 178 | 0.948276 | 0.647273 | 0.372881 |
| fold 3 | raw / logistic / ListNet / cap 8 | yes | 293 | 16 | 98 | 179 | 0.945392 | 0.646209 | 0.367893 |
| fold 4 | raw / histogram / ListNet / cap 8 | yes | 286 | 19 | 88 | 179 | 0.933566 | 0.670412 | 0.355172 |
| fold 5 | raw / logistic / ListNet / cap 16 | yes | 296 | 6 | 104 | 186 | 0.979730 | 0.641379 | 0.358804 |

## Factor Analysis

Pool size showed the expected trade-off. Increasing cap from 4 to all actions
raised strict-opportunity recall from `0.830185` to `1.000000`, while
conditional capture fell from `0.464529` to `0.422390` and unsafe selection
rose from `0.227534` to `0.282432`. Cap 16 already reached `0.986607` recall;
the remaining problem is ranking quality inside the retained set.

Pairwise ranking produced higher conditional strict capture than ListNet
(`0.562369` versus `0.313853`) but also a higher unsafe selection rate
(`0.319172` versus `0.191132`). Raw features likewise increased conditional
capture compared with question-relative features (`0.578815` versus
`0.298252`) while increasing unsafe selections (`0.342483` versus
`0.167821`). Histogram and logistic safety estimators were close: conditional
capture was `0.436521` versus `0.439709`, and unsafe rate was `0.248311`
versus `0.261993`.

The metric-wise best configurations are diagnostic extrema only:

| Metric | Configuration | Value |
| --- | --- | ---: |
| Pool recall | question-relative / logistic / ListNet / all | 1.000000 |
| Conditional capture | raw / histogram / ListNet / cap 4 | 0.701320 |
| Actual capture | raw / histogram / ListNet / cap 8 | 0.607143 |
| Baseline-change strict precision | question-relative / histogram / ListNet / cap 16 | 0.833333 |

No row above is claimed to satisfy the complete Stage 191 eligibility or
advancement gates. Optimizing any single value would hide the observed
precision, recall, and safety conflict.

## Execution And Resources

Formal PID `29444` was awaited to natural completion by one PowerShell
`Wait-Process` call. There was no polling, experiment timeout, retry, partial
continuation, fallback, OOM, or CUDA allocation.

```text
wall time:                         999.668081 s
Stage 191 reproduction:            999.665059 s
process CPU time:                 2103.296875 s
peak working set:                   5.719 GiB
peak private usage:                 3.435 GiB
minimum system available memory:    4.325 GiB
process guards:                    23 / 23 passed
```

The formal report SHA-256 is:

```text
8f454c07b8889d7cbbb6e66f2a0ce1960c89f7197c88018334a587947133887f
```

All 12 SVG charts were XML-validated and deterministically rasterized with
the pinned resvg implementation, project-owned Poppins fonts, and no font
fallback. Each PNG was opened at original resolution; titles, labels, values,
bars, and the 23-row guard chart were nonblank and free of clipping or overlap.
The pool-cap charts are ordered `4, 8, 16, all` for direct comparison.
The rasterization manifest SHA-256 is:

```text
9de94007254f8a46fc0e0860cf7ee72e7e39822727a35d69b25d2b7491428db1
```

## Decision

Stage 192 is complete and identifies `within_pool_ranker_miss` as the primary
bottleneck. It does not freeze or authorize Stage 193, full-train policy
selection, replacement, runtime E2E, development, test, or default activation.

The next protocol should be designed around a safety-constrained within-pool
reranker and evaluated only with train-side grouped nested CV. Its exact model
family, objective, and gates require a separate frozen protocol before any new
training begins.

## Current-Source Verification

The Stage 184-192 related regression suite passed `65 passed in 4.90s`.
Repository-wide Ruff lint passed, all seven changed Python files passed Ruff
format check, `pip check` found no broken requirements, CLI help succeeded, and
`git diff --check` found no errors.

The final full pytest suite used the single PID `7848`. The same PowerShell command
called `Wait-Process` once and waited for natural completion, without polling
or a pytest timeout:

```text
1145 passed, 1 warning in 21.73s
```

Stderr was empty. The warning is the existing FastAPI/Starlette `TestClient`
deprecation. PowerShell exposed an empty child `ExitCode` field after exit; it
is recorded as empty rather than represented as zero.
