# Stage 185 Joint Citation/F1 Constraint Ranking Protocol

## Objective

Stage 184 improved F1-risk ROC AUC from `0.570751` to `0.594294`, but passed
only one of five frozen quality gates. Promoting that representation would be
post-hoc and unauthorized. Stage 185 instead freezes a new train-only
experiment that treats citation preservation and F1 preservation as separate
constraints, then ranks expected gains only after predicted safety.

Stage 185 is a protocol stage. It reads only the aggregate public reports from
Stages 181-184. It does not load action rows, split files, questions,
documents, or gold labels; fit a model; evaluate a policy; open development or
test; run runtime E2E; add fallback behavior; or select a replacement policy.

## Evidence

The protocol is grounded in four frozen observations:

```text
Stage 181: 364 / 370 questions have at least one strict-success action
Stage 181 oracle: citation +58, mean F1 +0.111694
Stage 182: 55 F1 regressions among 129 changed selections
Stage 183: all 55 regressions have a citation-preserving safe alternative
Stage 184: best AUC 0.594294, only 1 / 5 quality gates passed
```

This means the action space has substantial headroom, while the current
single-target representation and ranking are not strong enough.

## Action And Feature Contract

For each train question, the Stage 182 emitted action is the reference. The
candidate set contains all unique runtime-generatable Stage 181
counterfactual actions plus that reference. The reference is an ordinary
ranked candidate and receives no special tie preference. Canonical runtime
action generation order resolves exact ties.

No empty candidate set can occur, so the protocol defines no fallback branch.
The ranking uses the frozen raw Stage 181 runtime features and the label-free
question-relative Stage 184 transforms. Neither representation is privileged.
Gold outcomes, question identity, split membership, and document identity are
forbidden runtime features.

## Model Targets

Each representation and estimator family fits three heads:

```text
citation_loss: citation delta < 0
f1_loss:       F1 delta < -1e-12
strict_gain:   neither metric regresses and at least one strictly improves
```

Gold targets are available only during train fitting and offline evaluation.
Runtime receives only the three predictions.

## Frozen Candidate Grid

```text
feature representations: 2
  raw_runtime
  question_relative_runtime

estimator families: 2
  class_balanced_logistic
  histogram_gradient_boosting

ranking rules: 3
  max_safety_risk_lexicographic
  citation_first_lexicographic
  pareto_constraint_dominance

safety dominance margins: 0.00, 0.02, 0.05
strict-gain margins:       0.00, 0.05
```

The complete policy grid contains `2 * 2 * 3 * 3 * 2 = 72`
configurations. Predictions are shared across ranking rules and margins.

## Nested Cross-Validation

Stage 186 must use the frozen five question-grouped outer folds. For each
outer fold, the remaining four folds produce inner OOF predictions for
selection. The selected configuration is refit on all four outer-training
folds and evaluated once on the held-out outer fold.

There are 20 inner partitions and five outer refits. Each partition fits
`2 * 2 * 3 = 12` heads, for a maximum of exactly 300 model-head fits.
All actions for one question remain in one fold.

An inner candidate is eligible only if aggregate citation and F1 are
non-regressing, both metrics are non-regressing in at least three of four
inner folds, and at least 10% of inner questions change. Eligible candidates
are selected lexicographically by regression repairs, newly induced
regressions, strict-success precision, citation delta, F1 delta, and stable
name order.

If an outer fold has no eligible inner candidate, it is recorded as
no-eligible. Stage 186 must not substitute a weaker configuration, retry, or
activate a fallback.

## Advancement Gates

All gates must pass:

```text
inner-eligible configuration in all 5 outer folds
aggregate gold-citation delta >= 0
aggregate mean F1 delta >= 0
citation bootstrap 95% CI lower >= 0
F1 bootstrap 95% CI lower >= 0
citation non-regression in at least 4 / 5 outer folds
F1 non-regression in at least 4 / 5 outer folds
repair at least 50% of the 55 Stage 182 regressions
new F1 regression rate <= 2%
citation-loss actions <= 4
strict-success precision >= 65%
changed questions >= 37
```

The last gate prevents a trivial keep-everything policy from passing.
Bootstrap uses 2,000 question-level replicates and seed 185.

Passing Stage 185 guards authorizes only the Stage 186 train-only nested
experiment. It does not authorize development/test evaluation, runtime E2E,
replacement-policy selection, Stage 178B, fallback, or default activation.

## Formal Freeze Result

The formal command loaded and fingerprinted only the four aggregate reports.
All expected source hashes and statuses matched. It loaded no train rows,
development rows, test rows, questions, documents, action features, or private
predictions; performed zero model fits and zero policy evaluations; and
persisted no private row-level data.

```text
status: stage185_joint_constraint_ranking_protocol_frozen
protocol valid: true
guard checks: 27 / 27 passed
Stage 186 train-only experiment authorized: true
development/test opened: false / false
runtime E2E authorized: false
replacement policy selected: false
default runtime activation: false
```

The formal report SHA-256 is:

```text
742ea385e76faa950677941d760f321c834cb23cbfb054458a66ed17807b837e
```

The seven SVGs were XML-validated and converted by the fixed resvg pipeline.
All seven PNGs were opened and inspected. Titles, long gate and guard names,
values, bars, and axis descriptions were complete without clipping, overlap,
or blank output. The rasterization manifest SHA-256 is:

```text
0a639a6a12eac30296c79e37ce662399f743a5ee511b54a43c844a757ee5e06a
```

## Verification

```text
Stage 184-185 and rasterizer focused regression: 12 passed in 8.99 seconds
full repository Ruff lint: passed
Stage 185 Ruff format check: 3 files already formatted
full pytest: 1099 passed, 1 warning in 25.98 seconds
```

Full pytest used PID `26932`. One PowerShell `Wait-Process` invocation waited
for it to end naturally, without polling or an experiment monitoring limit.
The process object reported `HasExited=True`; its child `ExitCode` field was
empty and was not represented as zero. The single warning remains the existing
FastAPI/Starlette `TestClient` deprecation.
