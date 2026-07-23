# Stage 183 Composition F1-Risk Failure Attribution

## Objective

Stage 183 explains why the valid but insufficient Stage 182 nested policy
selected 55 actions with answer-F1 regression and why one outer fold had no
inner-eligible policy. It freezes the Stage 182 action space, models, utility
functions, thresholds, and outer selections. It does not fit or select a
replacement model.

Only the 562 train questions are loaded through a full Stage 182 reproduction.
Development and test remain closed. No runtime E2E, fallback, Stage 178B, or
default activation is authorized.

## Private In-Memory Boundary

Stage 182 intentionally excludes per-question action rows and predictions
from its public report. Its runner therefore exposes a private in-memory
diagnostic callback containing:

```text
the reproduced Stage 181 action rows
the Stage 182 outer selected actions
the Stage 182 outer dual-head predictions
```

The default Stage 182 API still removes these fields from its public result.
Stage 183 consumes them in the same process and writes aggregate diagnostics
only. Question ids, answers, document ids, action ids, selected indices,
runtime feature rows, and per-question predictions remain forbidden publicly.

## Required Reproduction

Before attribution, the rerun must exactly reproduce the formal Stage 182
status, Stage 181 dataset guard, selected-question count, strict-action count,
F1-regression count, citation delta, mean F1 delta, selected policy counts,
and paired-bootstrap result. It must also capture exactly 12,298 action rows,
129 selected actions, and every outer prediction from folds that fitted an
outer model. Any mismatch invalidates Stage 183.

## Frozen Diagnostics

The report measures:

```text
selected F1-regression severity
regression concentration by fold, route, action family, and selected policy
fixed-bin predicted-risk calibration for selected and all outer actions
univariate separation of runtime-visible numeric/boolean action features
same-candidate safe-alternative headroom for the 55 regressions
failure reasons for every candidate in folds with no inner-eligible policy
```

A same-or-better safe alternative has nonnegative F1 delta and citation delta
at least as large as the selected regressed action. The report also checks
whether such an alternative appears in the frozen model utility top 3 or top
5. Gold outcomes define this offline diagnostic only and are not runtime
features.

## Interpretation Thresholds

The following thresholds are frozen before the formal run:

```text
maximum F1-risk head ROC AUC below 0.65 = weak separability
selected F1-regression rate above 0.25 = high regression rate
same-or-better safe-alternative rate at least 0.50 = high ranking headroom
```

When all three conditions hold, the primary bottleneck is classified as
`f1_risk_separability_and_ranking`. This authorizes only the design of a
better runtime-visible F1-risk representation and ranking experiment. It does
not authorize a runtime policy or development/test evaluation.

## Completion Boundary

A valid diagnostic ends as
`stage183_f1_risk_failure_attribution_complete`. Completion requires exact
Stage 182 reproduction, zero attribution model fits, closed dev/test, no
retry or fallback, and a public-safe aggregate report. The formal result is
recorded only after the run and visual verification finish.

## Formal Result

The formal second attempt completed as
`stage183_f1_risk_failure_attribution_complete`. All 18 process guards passed,
the Stage 182 result was reproduced exactly, and the private callback captured
12,298 action rows, 129 selected actions, and 9,502 outer predictions. The run
used one runtime resource build, 562 score-provider calls, 9,714 scored pairs,
and the frozen 88 Stage 182 head fits. Attribution fitted zero new models.
Development, test, Stage 178B, retry, fallback, runtime registration, and
default activation remained closed.

The 129 selected actions contained 55 F1 regressions, a regression rate of
`0.426357`. Their mean F1 delta was `-0.060563`, with median `-0.031972` and
minimum `-0.349874`. Severity counts were 12 mild, 23 moderate, 11 large, and
9 severe. The highest family regression rate with meaningful support was
`replace_slot_1` at `0.540000` (27/50); the broad `other` route was `0.534483`
(31/58). The two-item security-remediation route was 2/2 and is reported as a
small-sample concentration, not a general route estimate.

Selected actions had mean predicted F1 risk `0.374917` versus observed
regression rate `0.426357`, ECE `0.060952`, and Brier score `0.253928`. Across
all 9,502 outer predictions the corresponding values were predicted
`0.497259`, observed `0.474005`, ECE `0.061010`, and Brier `0.253417`.
Citation-head AUC ranged from `0.676301` to `0.757907`, while F1-risk-head AUC
ranged only from `0.535398` to `0.588032`.

All 55 regressed selections had a strict same-question alternative, and all
55 had an alternative with nonnegative F1 delta and citation delta at least as
large as the selected action. Only 5/55 had a safe alternative that also
strictly increased citation. The frozen model utility ranked a same-or-better
safe alternative in its top 3 for 29/55 and top 5 for 43/55. The selected
action's utility advantage over the best safe alternative had median
`0.148194`, showing that safe candidates existed but were often ranked too
low.

All 118 runtime-visible numeric or Boolean features were evaluated without
fitting a new model. The best oriented univariate ROC AUC was only `0.542595`
for `selected_answer_signal_score_mean`; no feature approached the frozen
`0.65` separability threshold. In outer fold 2, all 32 candidate policies
failed F1 fold nonregression; 9 also failed citation fold nonregression and 8
failed aggregate strict-A. Several candidates still achieved nonnegative
aggregate citation and F1 with 4/4 citation-safe and 3/4 F1-safe inner folds,
so the no-eligible result is specifically universal F1 fold instability under
the frozen rule.

All three preregistered findings therefore hold: weak F1-risk separability,
high selected regression rate, and high safe-alternative ranking headroom.
The primary bottleneck is
`f1_risk_separability_and_ranking`. This authorizes only a next-stage design
for stronger runtime-visible F1-risk representation and ranking. It does not
authorize runtime E2E or a replacement policy.

Formal wall time was `219.216661` seconds: `217.253131` seconds for the exact
Stage 182 reproduction and `1.961276` seconds for attribution. Peak working
set was `4,036,251,648` bytes, peak private usage `5,518,098,432` bytes,
minimum system available memory `2,065,408,000` bytes, and process CPU time
`674.515625` seconds. CUDA allocation and reservation were zero; no OOM
occurred. The formal PID `39748` was waited to natural completion without an
experiment timeout or polling.

The first formal attempt, PID `40168`, also completed naturally and reproduced
the data, but only 17/18 guards passed. A legitimate aggregate field named
`selected_actions` collided with the forbidden private-key scanner. That
attempt is invalid and fully archived under
`artifacts/stage183_attempt1_invalid_public_key_collision/`; the aggregate key
was renamed to `selected_action_population`, a public-safety regression test
was added, and the second attempt was run from the corrected source. Two
multi-file `apply_patch` attempts were atomically rejected before any write
because their context markers did not match; both were reapplied file by file.
Ruff also found one overlong visualization-label line before the formal run;
it was formatted and retested.

Seven SVG files passed XML parsing. Visual QA initially found incomplete Edge
text composition in three PNGs. Independent profiles, cache-busted file URLs,
one natural `Wait-Process` per Edge PID, and a shorter semantically equivalent
runtime-feature title produced seven complete verified renders. This changed
only generated SVG/PNG presentation, not the formal JSON.

Formal public report SHA-256:

```text
4dd611c9a759fd791288886c638bd9ec36b7564328eb53f2fef1742544540f1a
```

Current-source verification completed with 17 focused Stage 182/183 tests,
Ruff format checks for all 9 affected Python files, full-repository Ruff lint,
and full pytest. The full suite started PID `21184`; the same PowerShell
command called `Wait-Process` once and waited for natural completion. The
result was `1087 passed, 1 warning in 33.90s`, stderr was empty, and the process
object reported `HasExited=True`. Its child `ExitCode` field remained blank and
is not represented as zero. The warning is the existing FastAPI/Starlette
`TestClient` deprecation.
