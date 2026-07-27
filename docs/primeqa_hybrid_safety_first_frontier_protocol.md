# Stage 195 Safety-First Frontier Protocol

## Objective

Stage 194 completed its full train-only nested experiment, but all five outer
contexts had zero eligible configurations. Its strongest inner candidates kept
the cap-16 pool recall near `0.99`; the remaining conflict was between
conditional strict capture, strict precision, and unsafe selection. Stage 195
therefore freezes a safety-first selection experiment for Stage 196 instead of
expanding the candidate pool or relaxing any quality gate.

Stage 195 reads only the public aggregate Stage 194 report. It loads no
train/development/test rows, imports no LightGBM runtime, fits no model,
evaluates no policy, and changes no runtime default.

## Frozen Architecture

The first stage remains the four cap-16 safety-pool builders from Stage 194:
two feature representations crossed with two safety estimators. The original
baseline is unioned after the cap. Pool expansion is disabled because the five
Stage 194 top-inner pool recalls range from approximately `0.986` to `0.990`.

The Stage 194 LambdaMART gain ranker is retained. Its representation and tree
profile are now independent of the unsafe head representation and profile;
this Cartesian search adds policy combinations but no duplicate model fits.

## Cost-Sensitive Unsafe Heads

Stage 196 will train `lightgbm.LGBMClassifier` unsafe heads with frozen positive
class weights:

```text
scale_pos_weight: 1.0, 2.0, 4.0
is_unbalance:     false
class_weight:     none
```

LightGBM documents `scale_pos_weight` as the positive-class weight and forbids
using it together with `is_unbalance`. Its documentation also warns that class
weighting can produce poor individual probability estimates. The protocol
therefore uses only deterministic within-question risk ordering, never an
absolute probability threshold or calibrated probability claim.

Official sources:

- <https://lightgbm.readthedocs.io/en/latest/Parameters.html>
- <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>

## Safest-Prefix Frontier

For each cap-16 pool, actions are ordered by ascending unsafe score and
canonical runtime action order. Stage 196 evaluates prefixes of size
`2, 4, 8, 12, 16`, unions the original baseline after the prefix, and selects
the winner by descending LambdaMART score, then lower unsafe score, then
canonical action order.

This replaces Stage 194's gain-risk utility blend. There is no absolute risk
threshold, gold filter, retry, fallback, or weaker candidate substitution.

## Search And Fit Budget

```text
pool builders:                   4
gain models:                     4
unsafe-risk models:             12
safest-prefix sizes:             5
policy configurations:         960

pool safety fits / partition:    8
gain ranker fits / partition:    4
unsafe head fits / partition:   12
all fits / partition:           24
maximum partitions:             25
maximum model fits:            600
maximum LightGBM trees:    120,000
```

Models are shared across compatible policy configurations. One representation
and one weighted unsafe model are materialized at a time. The formal Stage 196
process must record event-driven resource statistics and be awaited by one
PowerShell `Wait-Process` call on its single PID until natural completion.

## Evaluation Contract

The nested cross-validation structure remains five outer folds and four inner
folds, grouped by question. The 13 inner eligibility constraints and all 17
outer advancement gates remain unchanged from Stage 194. Key thresholds remain:

```text
strict-success precision:             >= 0.65
first-stage pool recall:              >= 0.95
conditional strict capture:           >= 0.68
unsafe selection rate:                <= 0.25
```

If an outer context has no inner-eligible policy, Stage 196 records that fact
and does not evaluate a weaker substitute.

## Resource Boundary

The Stage 196 preflight threshold is `4.0 GiB` system-available memory. This
revision records the user's rejection of the older `6 GiB` requirement and the
Stage 194 observation that its full grid completed without OOM from `5.595 GiB`
preflight while reaching `3.756 GiB` minimum system-free memory. If preflight
is below `4.0 GiB`, the experiment must not start or silently shrink its grid.

## Authorization

When every Stage 195 guard passes, only the Stage 196 train-only experiment is
authorized. Development, test, full-train selection, runtime E2E, replacement
selection, Stage 178B, fallback, and default runtime activation remain closed.

## Formal Freeze

The formal Stage 195 command fingerprinted the exact Stage 194 public report
and completed the freeze itself in `0.001464` seconds:

```text
source report SHA matched:       true
protocol guards:                 58 / 58 passed
train/dev/test rows loaded:      false / false / false
LightGBM imported:               false
model fits / policy evaluations: 0 / 0
retry / fallback:                0 / 0
Stage 196 train-only experiment: authorized
runtime/dev/test/default changes: unauthorized
```

The formal report and deterministic rasterization manifest SHA-256 values are:

```text
report:   dc02e8423d633481802e42c6d52e85b9e1bda58861d1fc61492819b027b2c637
manifest: 602a02f2a5b565183de1a5166e759852b98313b550fc484dcc8cef04e5ad0caf
```

All ten SVG charts passed XML parsing and were rasterized by pinned
`resvg_py==0.3.3` with project-owned Poppins fonts, a white background, and no
font fallback. Every PNG was opened at original resolution. Titles, labels,
values, the 17 advancement gates, and all 58 guard rows were legible without
clipping or overlap.

The six-stage focused regression set passed `24 tests in 1.60s`. Repository-wide
Ruff lint, changed-file format checks, `pip check`, CLI help, and
`git diff --check` also passed. The complete pytest output reached 100% and
reported `1163 passed, 1 warning in 26.96s`, with empty stderr. The one warning
is the existing FastAPI/Starlette `TestClient` deprecation.

The current-source complete run used one Python PID (`24988`) and one
`Wait-Process` call without polling or a pytest timeout. PowerShell returned an
empty child `ExitCode` after complete stdout had reached 100%, so the outer
command reported failure rather than inventing exit code zero. The suite was
not rerun merely to manufacture that field.
