# Stage 193 Safety-Constrained LambdaMART Protocol

## Objective

Stage 192 showed that the cap-16 safety pool already reaches strict-opportunity
recall `0.986607`, while 500 strict opportunities retained by the reference
pool were still missed by the gain ranker. Of those misses, 465 selected an
unsafe winner. Stage 193 therefore freezes a new train-only protocol that
changes the within-pool ranking algorithm without expanding the pool.

The user explicitly selected route A: LightGBM LambdaMART grouped ranking with
an independent unsafe-risk head. Stage 193 itself reads only the public Stage
192 aggregate report. It does not install or import LightGBM, load train,
development, or test rows, fit a model, evaluate a policy, add fallback
behavior, or change a runtime default.

## Official Dependency Contract

Stage 194 must install exactly `lightgbm==4.7.0`. The official Python API
documents that `LGBMRanker` accepts query-group sizes whose sum equals the row
count and uses LambdaRank by default. The official parameter reference permits
integer relevance labels with explicit `label_gain`; it also states that CPU
`deterministic=true` should be paired with forced row-wise or column-wise
histogram construction.

The Windows x86-64 wheel is available on PyPI. Its frozen SHA-256 is:

```text
f42d1e5b32b6f170e606d7c689c6165671da98d7bf37f1addec2623efc8740c9
```

Sources:

- <https://pypi.org/project/lightgbm/4.7.0/>
- <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html>
- <https://lightgbm.readthedocs.io/en/latest/Parameters.html>

The experiment uses deterministic CPU training rather than assuming that GPU
is faster or equally reproducible for this 12,298-row workload. Stage 194 must
verify the installed version and import before loading training rows.

## Frozen First Stage

Stage 194 reuses the four Stage 191 safety-pool builders:

```text
feature representations:  raw_runtime, question_relative_runtime
safety estimators:        class-balanced logistic, histogram boosting
safety targets:           citation loss, F1 loss
pool cap:                 16
```

Actions remain ordered by joint safety risk, summed risk, and canonical action
order. The unique original baseline is unioned after the cap. The pool is
nonempty by construction and has no fallback branch. Pool-cap search is
removed because Stage 192 already established the cap-16 recall level.

## LambdaMART Labels

Every training action receives one integer relevance label:

```text
unsafe:       0
safe_zero:    1
strict_gain:  2
label_gain:   [0, 1, 4]
```

Strict gain requires nonnegative citation delta, F1 delta at least `-1e-12`,
and at least one positive delta. Safe zero requires zero citation delta and
absolute F1 delta at most `1e-12`. Every other action is unsafe. These outcomes
are training and offline-evaluation labels only, never runtime features.

Rows are grouped by question, and each row receives weight `1 / question
action count`. Thus every question has equal aggregate training weight even
when its action count differs.

## Model Profiles

Both feature representations are tested with two regularized tree profiles:

| Profile | Leaves | Max depth | Min child rows | L2 |
| --- | ---: | ---: | ---: | ---: |
| conservative | 7 | 3 | 40 | 2.0 |
| moderate | 15 | 4 | 25 | 1.0 |

The common frozen parameters are:

```text
boosting:                 gbdt
learning rate:            0.03
trees:                    300
max bins:                 63
feature / row sampling:   1.0 / 1.0
seed:                     193
threads:                  8 physical CPU cores
device:                   CPU
deterministic:            true
force_col_wise:           true
```

LambdaMART uses `objective=lambdarank`, `NDCG@1`, normalized lambdas, and
truncation level 4. Held-out labels are forbidden for fitting or early
stopping, so the number of trees is fixed at 300.

For every representation and tree profile, a separate LightGBM binary
classifier learns the unsafe label with the same tree structure. It uses no
class weighting or absolute runtime threshold. Only the deterministic
within-question ordering of its predicted risk is consumed, so probability
calibration is not required by this protocol.

## Safety-Constrained Selection

Within each cap-16 pool, raw LambdaMART margins and unsafe probabilities are
not directly added because their scales are unrelated. They are converted to
question-local deterministic ranks:

```text
gain_rank_fraction   = descending gain rank / (pool size - 1)
unsafe_rank_fraction = ascending unsafe-risk rank / (pool size - 1)

utility = 1 - gain_rank_fraction - lambda * unsafe_rank_fraction
```

For a one-action pool both fractions are zero. The frozen risk penalties are
`0.25`, `0.50`, `1.00`, and `2.00`. Ties are resolved by lower unsafe risk,
higher LambdaMART score, and canonical action order. No gold filter, absolute
probability threshold, retry, or fallback is present.

## Grid And Fit Budget

```text
pool builders:                    4
reranker representations:         2
tree profiles:                    2
risk penalties:                   4
policy configurations:           64

inner partitions:                20
possible outer refits:             5
pool-safety fits per partition:    8
new ranker/risk fits per partition: 8
total fits per partition:         16
maximum model fits:              400
```

Models are shared across pool builders and risk penalties. Sparse features are
materialized one partition and representation at a time, and released models
and matrices must be collected between partitions.

## Eligibility And Advancement

In addition to citation/F1 nonregression, changed-count, and strict-count
requirements, every inner-eligible policy must satisfy:

```text
strict-success precision:             >= 0.65
aggregate pool recall:                >= 0.95
pool recall in at least 3/4 folds:     >= 0.90
aggregate conditional strict capture: >= 0.68
conditional capture in at least 3/4:  >= 0.60
aggregate unsafe selection rate:      <= 0.25
unsafe rate in at least 3/4 folds:     <= 0.35
```

These are deliberately stronger than the Stage 192 reference trajectory
(`0.642346` conditional capture, `0.611300` strict precision, and `0.352703`
unsafe rate). A fold with no eligible configuration is recorded as such; it
does not receive a weaker substitute.

The outer advancement contract contains 17 gates: all 14 original Stage 188
quality gates, pool recall at least `0.95`, conditional capture at least
`0.68`, and unsafe selection rate at most `0.25`. Every gate must pass.

## Formal Freeze

The formal Stage 193 command fingerprinted the exact Stage 192 public report
and completed in `0.002742` seconds. It was not a long-running process.

```text
source report SHA matched:        true
protocol guards:                  59 / 59 passed
LightGBM installed/imported:       false / false
train/dev/test rows loaded:        false / false / false
model fits / policy evaluations:   0 / 0
retry / fallback:                  0 / 0
Stage 194 dependency provisioning: authorized
Stage 194 train-only experiment:   authorized
runtime/dev/test/default changes:  unauthorized
```

The formal report SHA-256 is:

```text
3124f186166fb8d04886c75801d271f82fb9b317a54026f97f055c10cefa9930
```

All nine SVG charts were XML-validated and rasterized with pinned
`resvg_py==0.3.3`, project-owned Poppins fonts, and no font fallback. Every PNG
was opened at original resolution. Titles, labels, values, zero/false states,
bars, and the 59-row guard chart were legible without clipping or overlap.

The rasterization manifest SHA-256 is:

```text
c4240aea630b2666a07d226c4bce247080718175b7493a580b5051a7d4d5cb36
```

## Authorization

Stage 193 authorizes Stage 194 to verify and install the frozen LightGBM wheel,
then run the train-only nested experiment if dependency and memory preflight
pass. It does not authorize development, test, full-train selection, runtime
E2E, replacement policy selection, Stage 178B, fallback, or default runtime
activation.

## Current-Source Verification

The Stage 189-193 protocol and attribution chain passed
`27 passed in 2.09s`. Repository-wide Ruff lint passed, all three new Python
files passed Ruff format check, `pip check` found no broken requirements, CLI
help succeeded, and `git diff --check` found no errors.

The complete pytest suite used the single PID `23868`. Its PowerShell command
called `Wait-Process` once and waited for natural completion without polling or
a pytest timeout:

```text
1149 passed, 1 warning in 22.03s
```

Stderr was empty. The warning is the existing FastAPI/Starlette `TestClient`
deprecation. PowerShell exposed an empty child `ExitCode` field after exit; it
is recorded as empty rather than represented as zero.
