# Stage 191 Rank-Capped Safety-Pool Nested CV

## Objective

Stage 191 executes the train-only nested-CV protocol frozen by Stage 190. It
tests a two-stage policy: first build a high-recall candidate pool by ranking
all runtime-generatable actions by predicted safety risk, then apply the frozen
Stage 188 gain ranker inside that pool.

The frozen grid contains two feature representations, two safety estimators,
two gain rankers, and pool caps `4`, `8`, `16`, and `all`, for 32 policy
configurations. Every pool includes the unique original baseline. There is no
fallback, pair sampling, list sampling, weaker-candidate substitution, retry,
development/test access, runtime E2E, full-train policy selection, or default
runtime activation.

## Implementation

Stage 191 reuses the exact Stage 188 encoders, safety heads, pairwise learner,
ListNet learner, labels, hyperparameters, full pair construction, and complete
question lists. For each question, actions are ordered by:

```text
ascending max(p(citation loss), p(F1 loss))
ascending p(citation loss) + p(F1 loss)
canonical runtime action order
```

The first `4`, `8`, `16`, or all actions form the initial pool, after which the
baseline is unioned. Selection inside the pool uses descending gain score,
ascending joint safety risk, and canonical action order.

Strict-opportunity pool recall is question-level and uses the same opportunity
definition as Stage 189: among questions having at least one strict action, the
pool must retain at least one strict action. Inner eligibility requires
aggregate recall at least `0.80` and per-fold recall at least `0.70` in at
least three of four folds, in addition to all Stage 188 eligibility rules.

## Formal Execution

The formal run loaded only 562 training rows and the frozen five
question-grouped folds. Stage 182 was reproduced before Stage 191 training;
all 10 reproduction checks passed. Development and test remained closed.

Formal PID `15940` was awaited to natural completion by exactly one PowerShell
`Wait-Process` call. There was no polling, experiment timeout, retry, partial
continuation, fallback, or OOM. PowerShell exposed an empty child `ExitCode`
after completion; it is recorded as empty rather than fabricated as zero.

```text
Stage 182 reproduction:               202.943518 s
Stage 191 nested CV:                  814.401546 s
total wall:                          1017.348335 s
process CPU time:                    2138.906250 s
model fits:                           288 / 300
inner partitions:                      20 / 20
outer refits:                            4 / 5
private predictions:                  403,333
public private-detail rows:                 0
```

The 288 fits are exact: 240 inner fits plus 48 fits for four authorized outer
refits. Fold 2 had no inner-eligible configuration, so its outer refit and
evaluation were correctly not run.

## Candidate-Pool Result

The first-stage candidate pool achieved its intended high recall on every
evaluated outer fold:

```text
fold 1: 1.000000, selected cap 16
fold 2: not evaluated
fold 3: 0.929577, selected cap 8
fold 4: 0.935897, selected cap 8
fold 5: 0.970588, selected cap 16
```

Across the four evaluated folds, 278 of 290 strict-opportunity questions were
recalled, giving `0.958621`. Strict-action retention was `0.383251`, mean pool
size was `11.932203`, and baseline inclusion was exactly `1.0`.

This aggregate covers 295 evaluated questions, not the complete 370-question
OOF population. It demonstrates that the new first stage solved the narrow
frontier-recall problem on evaluated folds, but it is not a complete final
policy metric because fold 2 was never opened for outer evaluation.

## Eligibility Failure

Fold 2's top inner candidates all passed the pool-recall constraints, with
recall between `0.934483` and `1.0` and all four folds above `0.70`. They failed
downstream safety/precision constraints instead:

```text
pool 8 histogram/ListNet:  citation -2, citation-safe folds 2, precision 0.608541
pool 8 logistic/ListNet:   citation -1, citation-safe folds 2, precision 0.604982
pool all ListNet:           citation  0, citation-safe folds 3, precision 0.593640
pool 16 logistic/ListNet:  citation +4, citation-safe folds 3, precision 0.580986
```

Thus no candidate simultaneously met citation, per-fold safety, and strict
precision `>= 0.60`. The candidate-pool recall gate itself was not the blocker.

## Partial Outer Metrics

The four evaluated folds selected pool caps 8 or 16; three selected ListNet and
one selected pairwise Pareto logistic. Their partial 295-question aggregate was:

```text
changed questions:                   276
strict successes:                    154
strict-success precision:       0.557971
gold-citation delta:                   0
mean F1 delta:                  0.010573
citation-loss actions:                12
F1-regression actions:               112
Stage 182 regression repair:     0.363636
new F1-regression rate:          0.325000
```

These values are diagnostic only. Since fold 2 was not evaluated, paired
bootstrap was correctly unavailable. The 15 frozen advancement gates passed
5 and failed 10; the candidate family was not accepted.

Held-out selected-bundle AUC further identifies the new bottleneck:

```text
citation-loss AUC:   0.808116
F1-loss AUC:         0.597291
strict-gain AUC:     0.490554
pairwise accuracy:   0.565943
```

Safety-ranked pool construction now has high recall, while pool-internal gain
ranking is weak and selected actions still incur excessive F1 and citation
loss. Stage 191 therefore authorizes no full-train selection, replacement
policy, runtime E2E, development/test access, or default activation.

## Resources And Verification

```text
peak working set:                   5.724 GiB
peak private usage:                 3.457 GiB
minimum system available memory:    4.247 GiB
CUDA allocated / reserved:          0 / 0
process guards:                     31 / 31 passed
```

All 15 SVG charts were deterministically rasterized with the pinned Poppins
fonts and no font fallback, then opened at original resolution. Gate/guard
names, negative F1, `not run` states, zero citation values, pool metrics, and
resource labels were legible without clipping or overlap.

```text
formal report SHA-256:
1747bd9a47a7f233b97e62e38550fc61d8eee8c3ea54cd063c32a66ee14f29d9

resvg manifest SHA-256:
662a45ff86d8ee41eef9eed931d6d0892fd150c400c0bace9f696ebcd9a64c5f
```

## Conclusion

Stage 191 validates the two-stage architecture but rejects the current
pool-internal rankers. The next stage should diagnose within-pool ordering
errors and the trade-off between strict precision and safety using only
train-side inner OOF aggregates. It must not reopen test or treat the partial
295-question outer aggregate as a complete result.

## Current-Source Verification

The Stage 184-191 related regression suite passed `49 passed in 5.16s`.
Repository-wide Ruff lint passed, all six changed Python files passed Ruff
format check, `pip check` reported no broken requirements, and the Stage 191
CLI help path loaded successfully.

The complete repository suite used PID `27508` and exactly one PowerShell
`Wait-Process` call until natural completion:

```text
1139 passed, 1 warning in 21.94s
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. The
child `ExitCode` field was empty after exit and was not fabricated as zero.
