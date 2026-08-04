# Stage 206 Two-Stage Change Ranker Nested CV

## Scope

Stage 206 is a train-only `5 outer x 4 inner` nested-CV experiment authorized by the
strictly amended Stage 205 protocol. It evaluates one exact Stage 196 control and ten
two-stage policies. Development and test data remain closed. This stage does not perform
full-train selection, runtime integration, replacement, or default activation.

## Two-Stage Decision

1. The source Stage 196 safety heads construct the fixed cap-16 pool.
2. A conditional LambdaMART ranker selects one nonbaseline action from that pool.
3. A binary change/abstain gate decides whether that action replaces the baseline.
4. The gate threshold is learned from training gate scores by an order statistic for
   target change coverage `0.25/0.40/0.55/0.70/0.85`.

The conditional ranker never trains on or selects the baseline. The gate receives runtime
features for the candidate and baseline, numeric feature deltas, source safety estimates
and within-question safety ranks, and the min-max normalized top1-top2 ranker margin. Raw
absolute LambdaMART scores are excluded from gate features.

## Strict OOF Contract

Every inner training partition is split into four deterministic question-grouped folds by
`SHA-256(outer context, heldout context, stable question id, seed 205) modulo 4`.

For every cross-fit fold:

- both source citation-loss and F1-loss heads fit only the complementary questions;
- both conditional ranker families fit only the complementary questions;
- source safety and ranker predictions are produced only for heldout questions;
- each training question contributes exactly one OOF gate row per ranker family.

The full source models still predict the partition heldout rows. After all gate-training
OOF rows are complete, one gate and one full conditional ranker are fitted per requested
ranker family. An outer refit repeats this complete procedure and never reuses inner models.

## Candidate Grid

- `strict_binary`: unsafe/safe-zero/strict labels `0/0/1`, `label_gain=[0,1]`.
- `strict_safety_graded`: unsafe/safe-zero/strict labels `0/1/2`,
  `label_gain=[0,1,4]`.
- Five target change coverages per family.
- Ten two-stage policies plus one exact Stage 196 control.

The original 13 inner eligibility constraints and 17 advancement gates remain unchanged.
No eligible inner configuration means no outer refit for that context. A weaker candidate
must not be substituted.

## Resource Contract

Each full inner partition uses:

- 4 full source fits;
- 8 source-safety cross-fit fits;
- 10 conditional-ranker fits;
- 2 gate fits;
- 24 total fits, including 14 LightGBM models.

The maximum complete run is 570 model fits and 96,000 LightGBM trees. The public report
retains aggregate metrics and diagnostics only. Training rows, question identifiers,
features, per-action predictions, gate scores, and candidate pools are private and are not
persisted.

## Verification and Outputs

Implementation tests cover deterministic cross-fit assignment, exact OOF row conservation,
baseline exclusion, raw-score exclusion, order-statistic thresholds, real model fitting,
full-budget nested orchestration, strict protocol authorization, atomic report persistence,
process guards, and SVG validity.

Formal outputs:

- `artifacts/primeqa_hybrid_two_stage_change_ranker_stage206.json`
- `artifacts/primeqa_hybrid_two_stage_change_ranker_stage206_visuals/`

Formal execution must use one Python PID and one PowerShell `Wait-Process` call, with no
polling and no experiment timeout. Any failed or interrupted execution is recorded as such;
it is never represented as a completed experiment.
