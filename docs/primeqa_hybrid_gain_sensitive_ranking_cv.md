# Stage 188 Gain-Sensitive Within-Question Ranking Nested CV

## Objective

Stage 188 executes the train-only nested cross-validation protocol frozen by
Stage 187. It tests whether a relative safety frontier followed by a learned
within-question gain ranker can preserve citation and F1 while retaining the
strict gains that Stage 186 discarded.

The frozen candidate family combines:

```text
feature representations:       raw runtime / question-relative runtime
safety estimators:             balanced logistic / histogram boosting
gain rankers:                  pairwise Pareto logistic / linear ListNet
safety-frontier margins:       0.00 / 0.02 / 0.05 / 0.10
policy configurations:         32
```

This stage does not select a full-train policy, run runtime E2E, open
development or test, enable Stage 178B, add fallback behavior, or change the
default runtime.

## Data And Evaluation Boundary

The formal run loaded only the 562 training rows and the frozen five
question-grouped folds. The effective population contained 370 answerable
train questions and 12,298 candidate actions, including 11,928 nonbaseline
actions.

Every outer fold used the other four folds for inner OOF selection. An outer
refit and held-out evaluation were permitted only after at least one of the 32
configurations passed every frozen inner eligibility gate. Gold citation and
F1 outcomes were used only as training targets and offline labels. No private
action row, pair row, listwise target, or prediction was persisted publicly.

The Stage 182 reference was reproduced before training. All ten reproduction
checks passed, including the 129 selected questions and 55 F1 regressions.

## Implementation

Stage 188 adds a dedicated gain-sensitive ranking module and PrimeQA runner.
The pairwise learner retains every within-question Pareto-comparable pair in
both orientations and omits incomparable trade-offs. The ListNet learner uses
each complete question list and a target distribution uniform over the Pareto
frontier in the highest available outcome tier.

The user selected implementation choice A for the two Stage 187 details that
required confirmation:

```text
ListNet scaling:                StandardScaler(with_mean=False)
patience improvement:          at least 1e-12 objective reduction
```

ListNet uses deterministic full-batch Adam with the frozen learning rate,
regularization, iteration, gradient, and patience settings. Pairwise scaling
uses the same sparse-safe standardization and performs the transform in place
to avoid retaining a second copy of the large pair matrix. This is a memory
implementation detail and does not change the frozen examples, features,
labels, weights, model, grid, or selection rules.

```text
inner partitions completed:                 20
outer refits completed:                      0
model fits:                            240 / 300 maximum
comparable pairs across fits:           3,349,920
omitted incomparable pairs:             1,409,208
listwise question fits:                     8,880
ListNet iterations:                          8,420
private predictions:                      393,536
public private-detail rows:                      0
raw feature count:                            140
question-relative feature count:              798
```

Only 240 of the maximum 300 fits were needed because all 20 inner partitions
completed but no outer fold produced an inner-eligible configuration. The
remaining 60 possible fits were the five forbidden outer refits, not skipped
inner work.

## Formal Result

All five outer folds had zero inner-eligible configurations. Consequently,
no configuration was selected, no outer model was refit, and no held-out outer
prediction was made.

The top-ranked ineligible inner candidate in each fold was:

```text
fold  changed  strict  precision  citation  mean F1   citation vs 182  F1 vs 182
1        151      34    0.225166         0  0.005080              -4   -0.001934
2        159      29    0.182390        -1  0.003320              -6   -0.003263
3        130      27    0.207692        -1  0.003903              -6   -0.000468
4        110      26    0.236364         1  0.004305              -3   -0.000621
5        117      29    0.247863         2  0.003200               0   -0.000194
```

Folds 1-4 favored question-relative features, balanced logistic safety, the
pairwise Pareto ranker, and margin 0.10. Fold 5 favored raw features,
histogram safety, ListNet, and margin 0.10.

The decisive common failure is strict-success precision. Even the top
candidate in each fold reached only 0.182390-0.247863, far below the frozen
0.60 inner threshold. Several top candidates also lost citations or F1 versus
Stage 182, missed citation nonregression in one inner fold, or exceeded the 2%
new-F1-regression target. This is not a near miss caused by one unstable fold.

The report's aggregate zero fields are structural no-selection values. They
must not be interpreted as a measured zero-delta held-out policy. Paired
bootstrap and prediction metrics are explicitly unavailable because the
required outer predictions do not exist.

## Advancement Decision

Three of fourteen advancement gates are mechanically true under the empty
selection state: citation-loss count at most four, F1-regression count at most
27, and new-F1-regression rate at most 2%. They are not evidence of policy
quality. The other eleven gates failed, beginning with the requirement for an
inner-eligible configuration in every outer fold.

```text
status: stage188_gain_sensitive_ranking_insufficient
experiment valid: true
candidate family accepted: false
full-train policy selection authorized: false
replacement policy selected: false
runtime E2E authorized: false
development opened: false
test opened: false
default runtime activation: false
```

The experiment is a valid negative result. The relative-frontier candidate
family changes many actions and finds some gains, but cannot distinguish
strict success with adequate precision under held-out inner evaluation. No
threshold was relaxed after observing the result, and no retry, weaker
candidate, fallback, or post-hoc outer evaluation was used.

## Process And Resources

The memory preflight initially found 2.764 GiB available. After the user
cleared applications and restarted Codex, the final preflight found 6.203 GiB
available and 6,339 MiB of free GPU memory. No Stage 188 or Python process was
left over.

The formal process used PID `5484`. One PowerShell command invoked
`Wait-Process` once for that PID and waited for natural completion. The tool
then resumed the same command cell; it did not issue another process query or
another `Wait-Process`. There was no polling, experiment timeout, restart,
partial continuation, fallback, or OOM. The child process object's `ExitCode`
field was empty and was not represented as zero; the enclosing command and
the valid report completed successfully.

```text
source authorization:              0.015842 seconds
Stage 182 reproduction:          231.795372 seconds
gain-sensitive nested CV:        658.365921 seconds
formal wall time:                890.177135 seconds
model fit time:                  581.044936 seconds
process CPU time:               1911.531250 seconds
peak working set:                  4.744 GiB
peak private usage:                3.460 GiB
minimum system available:          2.953 GiB
GPU allocated/reserved:                 0 / 0 bytes
```

## Visual Verification

The formal runner produced 12 SVG visualizations. They were rasterized with
the fixed `resvg_py==0.3.3` pipeline, explicit project-owned Poppins fonts,
and no font fallback. All 12 PNGs were nonblank and were opened at original
resolution. Titles, labels, zero and unavailable states, gate names, values,
bars, and axes were complete, with no incoherent overlap or clipping.

The formal report SHA-256 is:

```text
c68946d08750d0e07dadee7f70780048615919d79fe617520f17df078f1c6bcc
```

The resvg manifest SHA-256 is:

```text
6081c8761462eeabbfa3f495395463c7a943520f07cbd33e4068996033984cc3
```

## Verification

```text
Stage 188 focused tests:                 11 passed in 7.70 seconds
Stage 184-188 regression tests:          32 passed in 4.35 seconds
full repository Ruff lint:               passed
Stage 188 Ruff format check:             5 files already formatted
full pytest:                              1122 passed, 1 warning in 26.02 seconds
```

Full pytest used PID `16500` and one `Wait-Process` call for natural
completion. Its child `ExitCode` field was empty and was not fabricated as
zero. The warning remains the existing FastAPI/Starlette `TestClient`
deprecation.

An additional repository-wide Ruff format check was attempted and truthfully
failed because 310 pre-existing files would be reformatted. Those unrelated
files were not mechanically rewritten in Stage 188. The five new Stage 188
Python files pass their targeted format check.
