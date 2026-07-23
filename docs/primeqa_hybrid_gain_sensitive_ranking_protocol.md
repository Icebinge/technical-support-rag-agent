# Stage 187 Gain-Sensitive Within-Question Ranking Protocol

## Objective

Stage 186 repaired 53 of 55 Stage 182 F1 regressions and introduced no new
loss, but it selected zero strict-success actions. Its five outer folds all
chose `max_safety_risk_lexicographic`, producing a baseline-like policy with
zero citation and F1 gain.

Stage 187 freezes a new train-only experiment that changes the learning
problem rather than retuning Stage 186 margins. Stage 188 will learn relative
gain directly from candidates belonging to the same question, while separate
citation-loss and F1-loss models define a predicted safety frontier.

Stage 187 itself reads only six aggregate public reports. It does not load
train, development, or test rows; construct pairwise or listwise examples; fit
a model; evaluate a policy; run runtime E2E; add fallback behavior; select a
replacement policy; or change the default runtime.

## Evidence

The frozen evidence is:

```text
Stage 181 questions:                         370
questions with at least one strict action:   364
nonbaseline actions:                      11,928
strict actions:                           5,668
oracle citation delta:                      +58
oracle mean F1 delta:                 +0.111694

Stage 182 selected strict actions:            69
Stage 182 strict-success precision:      0.534884
Stage 182 citation delta:                      +5
Stage 182 mean F1 delta:                +0.005249
Stage 182 F1 regressions:                      55

Stage 186 changed questions:                  130
Stage 186 repaired regressions:                53
Stage 186 strict-success actions:               0
Stage 186 citation/F1 delta:                0 / 0
Stage 186 delta vs Stage 182:       -5 / -0.005249
```

Stage 186's held-out citation-loss head reached ROC AUC `0.863981`, while
F1-loss and strict-gain reached only `0.605203` and `0.602056`. The candidate
space and citation safety signal exist. The failure is the ranking semantics:
continuous safety minimization dominates the weak gain signal.

## Outcome Tiers

Gold outcomes define training labels only:

```text
strict_gain:
  citation_delta >= 0
  f1_delta >= -1e-12
  at least one delta is strictly positive

safe_zero:
  citation_delta == 0
  abs(f1_delta) <= 1e-12

unsafe:
  citation_delta < 0 or f1_delta < -1e-12
```

The original baseline action guarantees that every question has at least one
`safe_zero` action during training. Gold outcomes are forbidden runtime
features.

## Gain Rankers

Stage 188 compares two gain-sensitive rankers.

### Pairwise Pareto Logistic

The pairwise ranker operates only on actions from the same question. Its
feature vector is `features(left) - features(right)`.

Tier preference is strict gain over safe zero, and safe zero over unsafe.
Inside the same non-unsafe tier, a pair is retained only when one action
componentwise Pareto-dominates the other on citation and F1. Incomparable
citation/F1 trade-offs are omitted instead of receiving an arbitrary scalar
utility.

Both pair orientations are emitted, all comparable pairs are retained, and
question-balanced pair weights prevent questions with many candidates from
dominating. There is no pair sampling.

### Linear ListNet Top Frontier

The listwise ranker consumes the complete action list for one question. Its
target distribution is uniform over the citation/F1 Pareto frontier inside
the highest outcome tier available for that question.

The frozen objective is question-mean top-one ListNet cross-entropy plus
`0.001` L2 regularization. Optimization uses deterministic full-batch Adam
with learning rate `0.05`, at most 400 iterations, gradient tolerance `1e-7`,
patience 20, and zero initialization. All actions remain in every list; there
is no list sampling.

## Relative Safety Frontier

Separate citation-loss and F1-loss heads retain the Stage 186 logistic and
histogram-gradient-boosting hyperparameters.

For each question:

```text
citation_excess = p(citation_loss) - min_question_p(citation_loss)
f1_excess       = p(f1_loss) - min_question_p(f1_loss)
joint_excess    = max(citation_excess, f1_excess)
```

An action is admissible when:

```text
joint_excess <= min_question_joint_excess + frontier_margin
```

The frozen margins are `0.00`, `0.02`, `0.05`, and `0.10`. This definition is
mathematically nonempty for every nonempty candidate set. It is the ranking
algorithm itself, not a fallback.

Inside the frontier, actions are ordered by descending learned gain score,
ascending joint safety excess, then canonical runtime action order. Gain is
therefore primary among comparably safe actions instead of being the last
tie-break after two continuous risk scores.

## Candidate Grid

```text
feature representations: 2
  raw_runtime
  question_relative_runtime

safety estimators: 2
  class_balanced_logistic
  histogram_gradient_boosting

gain rankers: 2
  pairwise_pareto_logistic
  linear_listnet_top_frontier

safety frontier margins: 4
  0.00, 0.02, 0.05, 0.10
```

The grid contains `2 * 2 * 2 * 4 = 32` policy configurations. Safety
predictions are shared across rankers and margins. Gain scores are shared
across safety estimators and margins.

## Nested Cross-Validation

Stage 188 must reuse the frozen five question-grouped outer folds. Each outer
fold uses the other four folds for inner OOF policy selection, then refits the
selected configuration on all outer-training folds and evaluates the held-out
fold once.

For each representation and partition, Stage 188 fits four safety heads
(two targets by two estimators) and two gain rankers. That is 12 fits per
partition:

```text
inner partitions:        20
outer refits:              5
fits per partition:       12
maximum model fits:      300
```

If an outer fold has no inner-eligible configuration, it is recorded as
no-eligible. No weaker configuration, retry, reduced pair set, or fallback may
replace it.

## Inner Selection

An inner policy is eligible only when all conditions hold:

```text
aggregate citation delta >= 0
aggregate mean F1 delta >= 0
citation nonregression in at least 3 / 4 inner folds
F1 nonregression in at least 3 / 4 inner folds
changed questions >= 10% of inner questions
strict successes >= 8% of inner questions
strict-success precision >= 0.60
```

Eligible policies are selected lexicographically by strict-success count,
strict-success precision, F1-regression count, citation-loss count, citation
delta, F1 delta, repaired Stage 182 regressions, and stable policy name.

This moves strict gain into eligibility and the first selection objectives.
Stage 186 allowed a zero-strict-success policy to remain eligible and only
used strict precision as a late objective.

## Advancement Gates

All 14 gates must pass:

```text
inner-eligible configuration in all 5 outer folds
gold-citation delta >= 5
mean F1 delta >= 0.005249
citation bootstrap 95% CI lower >= 0
F1 bootstrap 95% CI lower >= 0
citation nonregression in at least 4 / 5 outer folds
F1 nonregression in at least 4 / 5 outer folds
strict-success actions >= 37
strict-success precision >= 0.65
citation-loss actions <= 4
F1-regression actions <= 27
repair at least 50% of Stage 182 regressions
new F1-regression rate <= 2%
changed questions >= 37
```

The citation and F1 floors equal Stage 182's aggregate gains. A policy cannot
advance merely by becoming safer while discarding the best previously
observed gain. The F1-regression ceiling requires at least a 50% reduction
from Stage 182's 55 regressions.

## Resource Contract

Pair differences must remain sparse and be materialized one partition and
representation at a time. The dense histogram matrix must be released before
pair construction. All comparable pairs are retained without sampling.

If preflight memory is insufficient, Stage 188 must not start. Resource
clearance must be requested instead of reducing the protocol. GPU is not
required. The formal process must be followed by one PowerShell
`Wait-Process` call for the same PID until natural exit, with no polling or
experiment timeout.

## Formal Freeze

The formal command loaded and fingerprinted only the Stage 181-186 aggregate
reports. All six expected hashes and statuses matched.

```text
status: stage187_gain_sensitive_ranking_protocol_frozen
protocol valid: true
guard checks: 41 / 41 passed
Stage 188 train-only experiment authorized: true
train rows loaded: false
development/test opened: false / false
model fits: 0
pair rows materialized: 0
listwise questions materialized: 0
policy evaluations: 0
fallback/retry: 0 / 0
runtime E2E authorized: false
full-train policy selection authorized: false
```

The freeze took `0.055598` seconds. It was not a long-running process and did
not require a process monitor.

The formal report SHA-256 is:

```text
b6125e28f532774dd2137374f6a236520f71e247c774eca1e4d8c078f31e21b2
```

## Visual Verification

Eight SVGs passed XML parsing and were rasterized by the fixed
`resvg_py==0.3.3` pipeline with explicit project-owned Poppins fonts and no
font fallback. All eight PNGs were nonblank and opened at original resolution.
Titles, long guard and gate names, zero values, bars, and axis descriptions
were complete without clipping or overlap.

The resvg manifest SHA-256 is:

```text
d803e4057e5cf7ea30ef37b536ec170bab02e10a272088ffc27f562befe98076
```

## Authorization

Passing Stage 187 authorizes only the Stage 188 train-only nested experiment.
It does not authorize development/test evaluation, full-train selection,
runtime E2E, replacement-policy activation, Stage 178B, fallback, or default
runtime changes.

## Verification

```text
Stage 185-187 protocol-chain regression: 15 passed in 7.50 seconds
full repository Ruff lint: passed
Stage 187 Ruff format check: 3 files already formatted
full pytest: 1111 passed, 1 warning in 26.94 seconds
```

Full pytest used PID `16340`. Its original PowerShell command invoked one
`Wait-Process` for that PID and waited for natural completion. The execution
tool yielded the still-running command cell once, then resumed that same cell;
it did not issue another process query or another `Wait-Process`.
`HasExited=True`, stderr was empty, and the child `ExitCode` field was empty
rather than being represented as zero. The warning remains the existing
FastAPI/Starlette `TestClient` deprecation.
