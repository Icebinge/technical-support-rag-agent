# Stage 201 Joint Risk/Winner Failure Attribution

## Objective

Stage 201 explains why every Stage 199 risk-signal and winner-rule cell failed
the frozen inner eligibility contract. The experiment remains train-only.
Development and test stay closed, all 13 thresholds remain unchanged, and the
diagnostic does not select a replacement policy or runtime rule.

## Diagnostic Stream

Stage 199 now exposes one immutable private snapshot after all 28 cells in an
outer context have been evaluated and the exact control has reproduced. The
Stage 201 sink aggregates each snapshot immediately and never persists a
question-level row. One model reconstruction therefore supplies all three
frozen populations:

```text
outer context x policy cell:                 140
inner fold x outer context x policy cell:    560
question context x policy cell:           41,440
model fits:                                  100
LightGBM trees:                           18,000
private predictions:                    245,960
outer refits:                                  0
additional diagnostic fits:                   0
```

The user selected near-boundary route A after Stage 200 was frozen. A failed
count constraint is near its boundary within one count, a failed rate
constraint within `0.01`, and failed mean F1 delta within `0.001`. This
clarification is recorded in Stage 201 and does not rewrite the Stage 200
artifact.

## Constraint Findings

No policy cell passed. Every cell failed between one and six constraints; the
modal failure count was four (`49/140` cells). The dominant blockers were:

```text
conditional strict capture:             135 / 140 failed
strict success precision:               125 / 140 failed
unsafe selection rate:                  116 / 140 failed
minimum capture-fold count:              89 / 140 failed
minimum unsafe-fold count:               51 / 140 failed
```

The capture and precision constraints co-failed in `125` cells with Jaccard
`0.925926`. Capture and unsafe rate co-failed in `111` cells with Jaccard
`0.792857`. Removing exactly one constraint repaired only five cells, all by
removing unsafe selection rate. This is diagnostic evidence that simply
relaxing one gate cannot resolve the candidate family.

The first-stage pool is not the blocker: pool recall and minimum pool-recall
fold count failed in zero cells. Citation delta, mean F1 delta, changed count,
and pool recall also passed all 140 cells. Representation-level regressions
were limited to 12 citation fold-count failures and one F1 fold-count failure.

## Fold Findings

Fold-level capture failures totaled `292`, compared with `171` unsafe-rate,
`70` citation-delta, and `36` mean-F1 violations. Fold 5 had the most unsafe
violations (`74`) and citation violations (`39`); fold 1 had the most capture
violations (`84`). Twelve cells passed aggregate citation delta but failed its
minimum nonregressing-fold count, and one did the same for F1. No aggregate
capture, unsafe, or pool-recall pass was overturned only by its fold-count
constraint.

The capture/unsafe Pareto frontier contained `5, 4, 6, 3, 9` cells across the
five outer contexts. Multiple tradeoff points exist, but no point satisfies
the complete frozen contract.

## Question Findings

Both frozen question partitions were exact over all `41,440` contexts:

```text
strict selected:                         22,044
baseline selected:                        6,270
safe zero selected:                       1,176
unsafe F1-only selected:                 11,134
unsafe citation-only selected:              354
unsafe citation-and-F1 selected:            462

winner selection miss:                  15,294
risk frontier exclusion:                 2,954
no strict opportunity:                     672
safety pool exclusion:                     476
```

`strict_selected` is the largest overall partition, but it is a successful
outcome and is not a failure mechanism. After excluding it, the dominant
failure mechanism is `winner_selection_miss`. A strict alternative survived
to the frontier in `37,338` contexts; a lower-risk strict alternative existed
in `10,477` (`0.280599`) and a higher-gain strict alternative in `11,700`
(`0.313354`).

Risk-signal marginals are close: unsafe winner rates range from `0.282915` to
`0.296815`. Winner rules move much more. `gain_only` selects strict outcomes
at `0.624662` with unsafe rate `0.338514`; `rank_utility_2.00` lowers unsafe
rate to `0.201520` but strict selection falls to `0.393750`. The top-4 gain
shortlist reaches unsafe rate `0.182601` with strict selection `0.335304`.
This is a strong objective tradeoff, not evidence that one risk signal solves
the problem.

## Decision

The formal status is
`stage201_joint_risk_winner_failure_attribution_complete`. The frozen scoring
rule assigns failure-count scores of `292` to objective research, `224` to
model research, and `13` to representation research. The next recommended
focus is therefore a question-conditional constrained selection objective
that directly models strict capture, precision, and unsafe loss together.

This recommendation is diagnostic, not causal. Stage 201 does not authorize a
new policy search, threshold relaxation, full-train selection, replacement,
runtime E2E, development, test, or default activation. The next step should
first freeze the objective experiment and its cross-validation contract.

## Execution History

Formal attempt 1 used shell-tracked PID `11668` and failed during the Stage 182
exact-reproduction preflight at `2026-07-27T15:48:58+08:00`. It completed zero
Stage 199 fits and produced no Stage 201 report. Its stdout and stderr logs are
retained. The original guard did not expose the failed sub-check, so the exact
source of that transient drift is unknown.

After user selection A, the reproduction guard was extended to carry formal
and actual values for all ten checks, a structured preflight-failure artifact
sink was added, and an explicit recovery-context file recorded attempt 1.
Formal attempt 2 used shell-tracked PID `10216` and Python runtime PID `27800`.
It naturally completed and the new failure sink did not trigger, so no failed
sub-check was observed on the second attempt. The final report preserves both
attempts rather than presenting the rerun as an initial success.

Attempt 2 wall time was `501.422619` seconds. Peak working set was `3.755 GiB`,
peak private usage was `4.140 GiB`, and minimum system-available memory was
`2.723 GiB`. CUDA allocation and reservation remained zero. Stage 199 exact
reproduction passed `15/15` checks and all Stage 201 process guards passed
`36/36`.

## Visual Verification

All 17 SVGs passed XML parsing and deterministic rasterization with pinned
`resvg_py==0.3.3`, project-owned Poppins fonts, a white background, and no font
fallback. Every PNG was nonblank and inspected at original resolution. One
postprocessing issue was found: the longest selected-outcome label needed a
wider left margin. The margin was corrected and all 17 images were rerendered
without rerunning any model or changing an aggregate metric.

Representative views:

- [constraint failures](../artifacts/primeqa_hybrid_joint_risk_winner_failure_attribution_stage201_visuals_png/stage201_constraint_failures.png)
- [question outcomes](../artifacts/primeqa_hybrid_joint_risk_winner_failure_attribution_stage201_visuals_png/stage201_selected_outcomes.png)
- [opportunity mechanisms](../artifacts/primeqa_hybrid_joint_risk_winner_failure_attribution_stage201_visuals_png/stage201_opportunity_mechanisms.png)
- [winner-rule unsafe rates](../artifacts/primeqa_hybrid_joint_risk_winner_failure_attribution_stage201_visuals_png/stage201_winner_rule_unsafe.png)
- [research-axis scores](../artifacts/primeqa_hybrid_joint_risk_winner_failure_attribution_stage201_visuals_png/stage201_research_axis_scores.png)

The final report and raster manifest hashes are recorded in the learning
journal after current-source verification.
