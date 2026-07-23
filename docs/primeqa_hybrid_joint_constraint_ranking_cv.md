# Stage 186 Joint Citation/F1 Constraint Ranking Cross-Validation

## Objective

Stage 186 executes the train-only nested cross-validation protocol frozen by
Stage 185. It tests whether three separately learned signals can preserve
citation and F1 while still selecting strict improvements:

```text
citation_loss
f1_loss
strict_gain
```

The experiment compares 72 frozen policies across raw and question-relative
runtime features, logistic and histogram-gradient-boosting estimators, three
ranking rules, three safety margins, and two strict-gain margins.

This stage does not select a full-train policy, run runtime E2E, open
development or test, enable Stage 178B, add fallback behavior, or change the
default runtime.

## Data And Evaluation Boundary

The formal run loaded only the 562 training rows and the frozen five
question-grouped folds. The effective evaluation population was 370 answerable
train questions with 12,298 candidate actions.

Every outer fold used the other four folds for inner OOF policy selection.
The selected policy was refit on all four outer-training folds and evaluated
once on the held-out outer fold. No question crossed a fold boundary.

Gold citation and F1 outcomes were used only as train targets and offline
evaluation labels. The model inputs contained only runtime-visible action
features. The report persisted no private action rows or predictions.

## Implementation

Stage 186 adds a dedicated joint-constraint ranking module and PrimeQA runner.
Four representation/estimator bundles share their vectorizer and feature
matrix within each partition. Each bundle fits the three frozen heads once,
then reuses those predictions across ranking rules and margins.

```text
outer folds:                    5
inner folds per outer fold:     4
inner partitions:              20
outer refits:                   5
policy configurations:         72
model targets per bundle:       3
model-head fits:               300 / 300
private predictions:           209,066
public prediction rows:        0
```

Raw bundles used 140 features and question-relative bundles used 798 features.
No retry, replacement configuration, fallback, or post-hoc policy switch was
allowed.

## Formal Result

All five outer folds found inner-eligible configurations, but every selected
configuration used the `max_safety_risk_lexicographic` ranking rule. The number
of eligible configurations by fold was:

```text
fold 1: 25
fold 2: 51
fold 3: 52
fold 4: 48
fold 5: 60
```

The held-out aggregate result was:

```text
questions:                         370
changed questions:                 130 / 370 (0.351351)
strict-success actions:            0
strict-success precision:          0.000000
citation-gain actions:             0
citation-loss actions:             0
F1-regression actions:             0
gold-citation delta:               0
mean F1 delta:                     0.000000
citation delta vs Stage 182:      -5
mean F1 delta vs Stage 182:       -0.005249
Stage 182 regressions:             55
repaired Stage 182 regressions:    53
repair rate:                       0.963636
new F1 regressions:                0
```

Each fold produced zero citation and F1 delta relative to the original
baseline. The paired 2,000-replicate bootstrap therefore produced exact
`[0, 0]` intervals for both metrics.

## Interpretation

The policy family learned a conservative baseline-like action pattern. It
removed 53 of the 55 Stage 182 regressions and introduced no new regression,
but it did so by selecting no strict-success action. Consequently, it also
discarded Stage 182's aggregate `+5` citations and `+0.005249` mean F1 gain.

The 130 changed actions are not evidence of useful gains: they are
outcome-equivalent zero-delta actions relative to the original baseline.
This is exactly why Stage 185 froze both a minimum changed-question gate and a
strict-success precision gate. The former passed, while the latter correctly
rejected the degenerate safety-only solution.

The selected-bundle held-out head metrics were:

```text
                 ROC AUC    average precision    prevalence
citation_loss    0.863981   0.360039             0.058627
f1_loss          0.605203   0.537792             0.453895
strict_gain      0.602056   0.543826             0.460888
```

Citation-loss separation is strong, but F1-loss and strict-gain separation
remain weak. Ranking by maximum safety risk therefore dominates the gain
signals and collapses toward zero-delta actions.

## Advancement Decision

Eleven of the twelve frozen advancement gates passed. The only failed gate
was:

```text
strict_success_precision >= 0.65
actual strict_success_precision = 0.00
```

The formal decision is:

```text
status: stage186_joint_constraint_ranking_insufficient
experiment valid: true
candidate family accepted: false
full-train policy selection authorized: false
replacement policy selected: false
runtime E2E authorized: false
development opened: false
test opened: false
default runtime activation: false
```

The run is valid and informative, but the candidate family cannot advance.
No retry or weaker acceptance rule was used.

## Process And Resources

The formal process used PID `37344`. One PowerShell command invoked
`Wait-Process` once for that PID and waited for natural completion. There was
no polling, experiment timeout, restart, partial continuation, fallback, or
OOM. The child process object reported `HasExited=True`; its `ExitCode` field
was empty and was not reported as zero.

```text
source authorization:              0.002278 seconds
Stage 182 reproduction:          285.910042 seconds
joint nested CV:                 524.757007 seconds
formal wall time:                810.669327 seconds
model fit time:                  403.261122 seconds
process CPU time:               2400.015625 seconds
peak working set:               3.739 GiB
peak private usage:             3.458 GiB
minimum system available:       3.345 GiB
GPU allocated/reserved:         0 / 0 bytes
```

The initial preflight found only 3.558 GiB available. After the user cleared
memory, the second preflight found 4.822 GiB, and the formal run proceeded
without terminating any user process.

## Visual Verification

The first visualization pass produced eight valid SVGs, but one chart mixed
private-prediction counts, model-fit counts, and GiB on a single scale. It was
readable but analytically poor. The experiment was not rerun. Instead, the
already persisted aggregate metrics were used to create a v2 visualization
set with separate execution-count and memory-GiB charts.

All nine v2 SVGs passed XML parsing and were rasterized by the fixed
`resvg_py==0.3.3` pipeline with explicit project-owned Poppins fonts and no
font fallback. All PNGs were nonblank. The two replacement charts were opened
at original resolution and checked for complete titles, labels, values,
proportions, overlap, and clipping.

The intermediate report hash
`b563790dae40993bde0438bc5344acdc0bddb30f722107321dc93895eae71199`
was superseded only because the report's visualization list was updated to the
v2 paths. No metric, decision, process guard, or experimental result changed.

The final formal report SHA-256 is:

```text
a3aee4190aca1f71f2cd3c611675a8b69090e41eee00fdae0515bce55edf02f4
```

The final resvg manifest SHA-256 is:

```text
8bd16f76b60244cd3ca4765b1b1cb2d532bf72af360eb55a012f22f5f67fb297
```

## Verification

```text
Stage 186 focused tests: 8 passed in 3.24 seconds
full repository Ruff lint: passed
Stage 186 Ruff format check: 6 files already formatted
full pytest: 1107 passed, 1 warning in 27.48 seconds
```

Full pytest used PID `40436`. Its original PowerShell command used one
`Wait-Process` and waited for natural completion. The execution tool yielded
the still-running command once, then resumed that same command cell; it did
not issue another process query or another `Wait-Process`. `HasExited=True`;
the child `ExitCode` field was empty and was not represented as zero. The
single warning remains the existing FastAPI/Starlette `TestClient`
deprecation.
