# Stage 182 Composition Dual-Target Nested Cross-Validation

## Objective

Stage 182 tests whether the Stage 181 action space becomes safely selectable
when citation benefit and answer-fidelity risk are modeled separately. It is a
train-only experiment over the same 562 frozen train questions and 370
answerable questions. Development and test remain closed. No selected model is
registered in runtime, and no default behavior changes.

The two offline labels are:

```text
citation_gain = gold-citation delta > 0
f1_regression = answer token-F1 delta < -1e-12
```

Gold values assign training labels and evaluate held-out outcomes only. Model
features are the runtime-visible Stage 181 action features.

## Frozen Candidate Grid

Two model families are compared:

```text
class-balanced logistic regression
class-balanced histogram gradient boosting
```

Each family supplies separate citation-gain and F1-regression heads. Their
probabilities are combined by four fixed utility functions:

```text
p(citation gain)
p(citation gain) * (1 - p(F1 regression))
p(citation gain) - 0.5 * p(F1 regression)
p(citation gain) - p(F1 regression)
```

Each utility is paired with 10%, 25%, 50%, and 100% target coverage, yielding
32 candidates. Coverage is converted into a utility threshold from training
OOF predictions. At evaluation or runtime, each question is independent: the
top action is applied only when it reaches the learned threshold. Otherwise
the answer is deliberately left unchanged. This abstention is part of the
policy definition, not a retry or fallback.

## Nested Selection

The existing five question-grouped train folds are frozen. For each outer
fold, the other four folds run an inner four-fold OOF experiment. A candidate
is inner-eligible only when it:

```text
has nonnegative aggregate citation delta
has nonnegative aggregate mean F1 delta
strictly improves at least one aggregate metric
does not regress citation in any of the four inner folds
does not regress F1 in any of the four inner folds
```

The eligible candidate with the largest citation gain is selected, followed
by F1 gain, strict-action precision, lower coverage, and stable policy name as
deterministic tie-breakers. The two heads are then refit on all four outer
training folds. The outer fold is used exactly once for evaluation. If no
candidate is inner-eligible, that outer fold records no selected policy and
keeps the baseline; it does not silently choose a weaker candidate.

## Reproduction Boundary

Because the Stage 181 public report intentionally excludes private action
rows, Stage 182 rebuilds the frozen collection once. Before model evaluation,
it must exactly reproduce:

```text
12,298 rows including baseline controls
11,928 unique nonbaseline actions
5,668 strict-expected actions
the complete outcome-class distribution
the Stage 181 oracle citation and F1 deltas
the Stage 180 reconstructed citation and F1 deltas
```

Any mismatch invalidates the experiment before dual-target results are used.

## Advancement Gates

A valid experiment advances only when all process guards pass and the nested
outer result satisfies every gate:

```text
all five outer folds selected an inner-eligible policy
aggregate gold-citation delta is strictly positive
aggregate mean answer F1 does not regress
95% paired-bootstrap lower bounds are nonnegative for citation and F1
citation does not regress in at least four of five outer folds
F1 does not regress in at least four of five outer folds
```

Passing authorizes only a Stage 183 optional runtime Agent E2E experiment. It
does not authorize default activation, Stage 178B, fallback, or opening
development/test. Failure ends as
`stage182_dual_target_nested_cv_insufficient`.

## Required Outputs

The public report contains aggregate source authorization, reproduction
checks, inner candidate summaries, selected policy names, head metrics,
outer-fold policy outcomes, paired bootstrap intervals, resources, timing,
quality gates, and process guards. Raw questions, answers, document ids,
selected indices, runtime feature rows, and per-question predictions are
forbidden.

## Formal Result

The formal train-only run completed with all 21 process guards passing. It
exactly reproduced all eight frozen Stage 181 aggregate checks, including
12,298 rows, 11,928 nonbaseline actions, 5,668 strict-expected actions, the
complete outcome distribution, oracle citation/F1 deltas, and reconstructed
Stage 180 citation/F1 deltas.

The nested outer result selected 129 of 370 questions (`0.348649`). It found
69 strict-expected actions (`0.534884` precision), nine citation-gain actions,
four citation-loss actions, and 55 actions with F1 regression. Aggregate gold
citation improved by `+5`; mean F1 delta across all 370 answerable questions
was `+0.005249`, while selected-action mean F1 delta was `+0.015056`.

The paired bootstrap result was:

```text
metric                 observed     95% CI
gold citation delta      +5         [-2, 12]
mean answer F1 delta     +0.005249  [-0.001185, 0.012077]
```

Citation did not regress in all five outer folds, and F1 did not regress in
four of five. However, outer fold 2 had no inner-eligible policy. Five of the
eight advancement gates passed; the failed gates were all outer folds
selecting a policy, citation bootstrap lower bound nonnegative, and F1
bootstrap lower bound nonnegative. The formal status is therefore
`stage182_dual_target_nested_cv_insufficient`, with no candidate selected and
no runtime authorization.

## Fold Behavior

The five outer-fold outcomes were:

```text
fold   selected policy                                      actions  citation  mean F1
1      logistic citation-minus-risk, 10% coverage                4       +1    -0.001694
2      no inner-eligible policy                                  0        0     0.000000
3      logistic citation-minus-risk, 50% coverage               34        0    +0.008945
4      histogram boosting safe-product, 50% coverage            52       +1    +0.006419
5      logistic citation-minus-risk, 50% coverage               39       +3    +0.013342
```

The citation-gain heads were materially more separable than the Stage 181
single target. Inner OOF ROC AUC ranged from `0.694015` to `0.757907` for
histogram boosting and from `0.676301` to `0.727036` for logistic regression,
against citation-gain prevalence from `0.026100` to `0.034966`. F1-regression
prediction remained weak: histogram-boosting ROC AUC ranged from `0.535398`
to `0.568966`, and logistic ROC AUC from `0.536127` to `0.588032`. The weak
risk head explains why 55 selected actions still regressed F1 and why the
bootstrap interval crossed zero despite positive aggregate F1.

## Runtime And Resources

The run performed one resource build, 562 Agent collection turns, 562 score
provider calls over 9,714 pairs, 50 frozen Stage 180 model-head fits, and 88
Stage 182 dual-target model-head fits. The latter equals 80 mandatory inner
heads plus eight outer heads; fold 2 had no eligible policy and therefore did
not fit outer heads. There were no retries, fallbacks, failed Agent calls,
development/test reads, or runtime registrations.

Wall time was `218.780654` seconds. Peak working set was `4,032,180,224`
bytes, peak private usage was `5,514,928,128` bytes, minimum system-available
memory was `1,778,139,136` bytes, and process CPU time was `683.671875`
seconds. CUDA allocated/reserved were both zero. The memory estimate before
the run was conservative but sufficient; no OOM occurred.

The first outer shell invocation contained a single PowerShell
`Wait-Process`, but the command channel itself returned code 124 after about
14 seconds because of its default transport limit. It did not terminate or
restart Python PID `29792`. A second command attached one `Wait-Process` to
that same PID and then remained blocked until its natural completion. There
was no polling, process restart, partial continuation, or experiment-level
timeout.

## Visualization And Verification Audit

All six SVGs parsed successfully. The first batch browser-render attempt
interleaved Edge startup/compositor state and produced three horizontally
incomplete screenshots even though their SVG sources were correct. These
screenshots were not accepted as visual verification. Fresh profiles,
cache-busted file URLs, compositor completion, and individual waited Edge
processes produced complete titles, labels, bars, and values for the six
charts. The existing QQBrowser-import and disk-cache warnings did not affect
the corrected PNGs. No formal JSON metric changed during visual verification.

The first documentation append attempt used an incorrect end-of-file context
and was rejected atomically. The file was inspected and this result was then
appended against its actual final lines; no partial documentation write
occurred.

The formal report SHA-256 is:

```text
c3dfbe7484a604b8491bed0531fc82b20bd092016fd7ddf303955b7c7c89044a
```

Final current-source verification produced:

```text
Stage 181-182 focused regression: 21 passed in 1.61s
five Stage 182 Python files: Ruff format check passed
full repository Ruff lint: passed
full repository pytest: 1081 passed, 1 warning in 19.46s
full pytest PID / stderr bytes: 39872 / 0
```

The full pytest process used one PowerShell `Wait-Process` call and completed
naturally. The Windows child `ExitCode` field remained blank, so it is not
reported as zero. The warning is the existing FastAPI/Starlette `TestClient`
deprecation.
