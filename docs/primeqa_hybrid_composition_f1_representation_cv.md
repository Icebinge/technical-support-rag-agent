# Stage 184 Composition F1 Representation Cross-Validation

## Objective

Stage 184 responds to the frozen Stage 183 bottleneck
`f1_risk_separability_and_ranking`. It tests whether richer runtime-visible
within-question representations can separate F1 regressions and rank safe
alternatives better than the raw Stage 182 binary heads. It selects at most a
representation for a later nested policy experiment; it does not select or
activate an answer-composition policy.

The experiment loads only the 562 train questions and uses the frozen five
question-grouped folds. Development, test, Stage 178B, runtime E2E, retry,
fallback, and default activation remain closed. Gold F1 and citation outcomes
are training targets and offline evaluation labels only.

## Source Boundary

The formal Stage 183 report must have SHA-256
`4dd611c9a759fd791288886c638bd9ec36b7564328eb53f2fef1742544540f1a`,
status `stage183_f1_risk_failure_attribution_complete`, 18/18 passing process
guards, and primary bottleneck `f1_risk_separability_and_ranking`. Stage 184
then reproduces the formal Stage 182 experiment exactly in one process to
recover its private action rows and 129 selected actions. Public output remains
aggregate-only.

## Runtime-Visible Features

The raw feature space is the frozen Stage 181 action feature mapping. The
question-relative feature space adds only label-free transforms calculated
from the current question's complete deterministic action list:

```text
within-question delta from mean
within-question z-score
within-question percentile
within-question minimum/maximum flags
added-candidate aggregate minus selected-action aggregate contrasts
```

These transforms require no second retrieval, model call, document identity,
gold answer, gold document, or outcome label. Every question is wholly inside
one frozen fold, and model fitting never sees the held-out fold.

## Frozen Candidate Grid

Eight candidates are fixed before the formal run:

```text
raw_logistic_binary
raw_hist_binary
relative_logistic_binary
relative_hist_binary
relative_logistic_ordinal
relative_hist_ordinal
relative_hist_quantile_p25
relative_pairwise_logistic
```

Binary candidates predict any F1 regression. Ordinal candidates fit three
heads for any, material (`<-0.01`), and large (`<-0.05`) regression and combine
them with increasing severity weights. The quantile candidate predicts the
25th percentile of F1 delta. The pairwise candidate trains on every
within-question unequal-F1 pair in both orientations, with each question given
equal total weight, then scores an action by its mean predicted win probability
against the other actions for that question.

Across five folds the grid performs exactly 60 model fits. Predictions remain
private in memory. Public reports contain only aggregate and fold metrics.

## Selection And Gates

The selected non-raw candidate maximizes aggregate five-fold OOF F1-regression
ROC AUC, then average precision, Stage 182 regression safe-alternative top-3
and top-1 rate, and deterministic name order. The best raw candidate is the
reference selected by the same metric.

All five gates must pass before Stage 185 may run a nested policy experiment:

```text
selected aggregate risk ROC AUC >= 0.62
ROC AUC gain versus best raw >= 0.03
selected fold ROC AUC >= raw fold ROC AUC in at least 4/5 folds
same-or-better-citation safe alternative in F1-safety top 3 for >= 70% of the 55 regressions
same-or-better-citation safe alternative in F1-safety top 5 for >= 85% of the 55 regressions
```

The last two metrics use gold outcomes only for offline evaluation. They do not
define a runtime filter. If any gate fails, Stage 184 ends as insufficient and
does not authorize policy selection, development evaluation, or runtime E2E.

## Required Reporting

The formal report records exact Stage 182 reproduction, raw and relative
feature counts, aggregate/fold AUC and average precision, safety-only top-1
outcomes, safe-alternative headroom on the 55 Stage 182 regressions, model fit
counts, timing, resources, process guards, and six SVG visualizations. No
question id, action id, selected indices, feature row, prediction, answer,
document id, or gold label may enter the public report.

## Formal Result

The formal process used PID `28628`. One PowerShell `Wait-Process` invocation
waited for that PID to end naturally; there was no polling, restart, partial
continuation, experiment timeout, retry, or fallback. The run reproduced Stage
182, evaluated 11,928 non-control actions from 370 answerable train questions,
and kept development and test closed. It completed 60 model fits and 95,424
private held-out predictions.

The best raw reference was `raw_logistic_binary`, with ROC AUC `0.570751`.
The selected non-raw representation was `relative_hist_ordinal`, with ROC AUC
`0.594294`, average precision `0.560587`, and a raw-reference AUC gain of
`0.023543`. It matched or exceeded the raw fold AUC in four of five folds.
Among the 55 Stage 182 F1 regressions, its safe-alternative top-1/top-3/top-5
rates were `0.527273`, `0.690909`, and `0.836364`.

Only the fold-stability gate passed. The selected representation failed the
minimum AUC, AUC-gain, top-3, and top-5 gates. The frozen decision is therefore:

```text
status: stage184_f1_representation_cv_insufficient
candidate accepted: false
nested policy experiment authorized: false
process guards: 21 / 21 passed
```

The result shows a real but insufficient representation gain. Stage 184 does
not authorize Stage 185 policy selection, runtime E2E, default activation, or
opening development/test.

## Timing And Resources

```text
source loading:           0.002592 seconds
Stage 182 reproduction: 236.690990 seconds
cross-validation:       546.834553 seconds
formal wall:            783.528135 seconds
peak working set:     6,994,673,664 bytes
peak private usage:   3,709,411,328 bytes
minimum system free:  3,838,169,088 bytes
CPU time:              1,634.828125 seconds
CUDA allocated/reserved: 0 / 0
OOM: false
```

The public report SHA-256 is:

```text
bdbb49bf31a0f889a431924ee1630c7593ec485fb1bf283def44048776a29eea
```

## Deterministic Visualization

Post-run browser rendering proved operationally unsuitable. A batch Edge
command was given a 120-second outer shell limit before the project's
no-monitoring-limit rule was reapplied; it produced two PNGs, then the third
renderer process hung for more than one hour. The formal experiment had
already completed and its report was unaffected. After explicit user
confirmation, only that isolated Edge process tree was terminated.

The project now uses `resvg_py==0.3.3` through a dedicated rasterizer port and
CLI. Rendering disables system fonts, uses bundled Poppins Regular/Bold files
with their OFL license, has no browser or other fallback, and verifies the PNG
signature, dimensions, SHA-256, and non-background pixel count. The six Stage
184 SVGs were regenerated with the explicit font and rasterized to six
1560-pixel-wide PNGs. Every PNG was opened and inspected; titles, category
labels, values, bars, and axis descriptions were present without clipping or
overlap. The deterministic rendering manifest SHA-256 is:

```text
cec2daa085cc8293cb352571da33f8371dd9e0ce25ce1a486a002e837ffb024b
```

## Verification

The final current-source verification completed as follows:

```text
Stage 184 and rasterizer focused tests: 9 passed in 3.66 seconds
full repository Ruff lint: passed
Stage 184 related Ruff format check: 9 files already formatted
full pytest: 1096 passed, 1 warning in 31.23 seconds
pip check: no broken requirements
wheel package data: OFL plus both Poppins TTF files present
```

Full pytest used PID `30192` and one `Wait-Process` invocation. The process
ended naturally with `HasExited=True`; the Windows child `ExitCode` field was
empty and was not reported as zero. The warning is the existing
FastAPI/Starlette `TestClient` deprecation. A repository-wide format check also
identified 310 pre-existing files that would be reformatted; Stage 184 did not
rewrite those unrelated files.
