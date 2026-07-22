# Stage 176 Listwise View Calibration Nested CV

## Objective

Stage 176 determines whether Stage 175 failed because its listwise ranker was
poorly calibrated at the evidence-view level. It freezes the selected
`listwise_none` training family and compares four predeclared calibration
policies on the same grouped OOF logits.

The experiment remains train-only. It does not load development or test data,
run answer generation or Agent turns, write model checkpoints, introduce
retries or fallbacks, or alter the default runtime.

## Frozen Protocol

Every fit starts from local snapshot
`cross-encoder/ms-marco-MiniLM-L-6-v2` revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8` and uses the exact Stage 175
`listwise_none` grouped objective. The protocol performs 25 fresh fits:

```text
5 outer folds x (4 inner fits + 1 outer fit) = 25 fits
```

The four view policies are:

```text
absolute_top1:
  top1_logit

top1_top2_none_margin:
  top1_logit - max(0, top2_logit)

candidate_mass_vs_none:
  logsumexp(all visible candidate logits) - none_logit(0)

bounded_absolute_relative:
  0.5*tanh(top1/4) + 0.5*tanh((top1-max(0,top2))/4)
```

The first three policies use the frozen 21-value Stage 175 logit grid from
-4 to 8. The bounded policy uses 21 values from -1 to 1 in 0.1 increments.
This produces 84 fixed policy/threshold specifications. Inner OOF selects one
spec; the outer held-out fold is evaluated once. Twelve report, split,
document, and model sources are SHA-256 authenticated before loading.

## Execution Checks

Nine focused tests passed on the first run. A pure-synthetic GPU smoke used no
dataset rows and completed one real listwise fit, 14/14 predictions, and all
four policy calculations. Its loss changed from 4.100567 to 2.681946 and CUDA
allocated/reserved peaks were 0.443/0.490 GiB.

The formal process was PID `17264`. It completed all 25 fits and 2,046
optimizer steps, passed every process guard, and exited `0` on its first
formal launch. There was no OOM, retry, fallback, restart, or partial run.

## Formal Result

All five inner loops selected `top1_top2_none_margin`. None selected the other
three policies, and no inner loop had an eligible specification. The strict
nested OOF result was:

```text
balanced accuracy:                    0.659947
ROC AUC:                              0.758915
initial-visible compose:              0.514286  fail (>= 0.70)
alternate-only inspect:               0.956522  pass (>= 0.50)
alternate-only final compose:         0.152174  fail (>= 0.70)
alternate-only exact path:            0.108696  fail (>= 0.40)
insufficient final compose:           0.122034  pass (<= 0.20)
```

Compared with Stage 175, balanced accuracy increased by 0.020236, initial
compose by 0.045715, final compose and exact path by 0.010870, while false
compose decreased by 0.010169. AUC was effectively unchanged at -0.000219.
This is a modest operating-point improvement, not a solved gate.

Every held-out fold passed the false-compose ceiling. Fold rates were
0.101695, 0.150943, 0.116667, 0.049180, and 0.193548. However, final-compose
rates were only 0.10, 0.181818, 0.111111, 0.058824, and 0.333333. The paired
alternate-only transition remains the dominant failure.

Complete outer-OOF policy diagnostics were:

```text
policy                         bal acc   AUC       initial   final     exact     false
absolute_top1                  0.580947  0.757688  0.251429  0.032609  0.021739  0.050847
top1_top2_none_margin          0.663802  0.758915  0.531429  0.163043  0.119565  0.128814
candidate_mass_vs_none         0.535718  0.745264  0.114286  0.000000  0.000000  0.023729
bounded_absolute_relative      0.592511  0.760332  0.302857  0.032609  0.021739  0.071186
```

All policies had zero eligible thresholds. The original relative margin won
5/5 folds and the complete OOF diagnostic. Therefore the remaining failure is
not explained by trying the wrong simple calibration rule. Continuing to tune
thresholds or algebraic combinations is not justified.

Mean listwise loss decreased from 1.420865 to 1.056289. The model learned its
ranking objective, but the resulting score does not provide the evidence
entailment/answerability separation required by the strict gate.

## Resources

```text
wall time:                       767.297528 seconds
candidate replay:                93.006403 seconds
nested calibration:             646.608632 seconds
fine-tuning compute:             508.135325 seconds
fold inference:                 125.225027 seconds
process working-set peak:         4.598 GiB
process private-usage peak:      11.900 GiB
GPU allocated peak:               2.954 GiB
GPU reserved peak:                5.734 GiB
minimum system available memory:  1.267 GiB
generation calls:                 0
```

The formal PowerShell command started one process and waited on the same PID
with `Wait-Process` until natural completion. No monitoring loop or segmented
status command was used.

## Visual Verification

Nine SVG charts cover quality gates, Stage 175/176 rates, policy selection,
policy OOF quality, outer-fold safety/path, training loss, timing, and
resources. All parse successfully. Four key charts rendered to PNG and match
the JSON report. Edge emitted only its existing QQBrowser profile warning;
its headless screenshots abbreviated two long titles to `S5`/`S6`, while the
SVG source retains the complete titles.

## Decision

The formal status is `stage176_view_calibration_insufficient` and
`candidate_selected=false`. Development and test remain unopened, no model is
saved or registered, runtime E2E is not authorized, and the default runtime
is unchanged.

Stage 176 closes the simple calibration branch. The next justified train-only
stage is to stop treating this model as an evidence-sufficiency gate and
measure it strictly as the second-stage document reranker it was trained to
be. Stage 177 should report grouped outer-OOF gold rank, MRR, Recall@1/3/5/10,
and changes versus original RRF and the frozen cross-encoder. If reranking is
materially better, it can advance to an Agent retrieval-chain E2E while the
failed sufficiency gate remains disabled.

## Process Corrections

- One parallel preflight wrapper reached its approximately 10.4-second tool
  timeout after printing the successful `46 passed` result. Source
  authorization and Ruff outputs were not visible, so they were not assumed
  successful; both were rerun independently and passed.
- The Stage 176 implementation, focused tests, GPU smoke, and formal run had
  no functional failures.
- Headless screenshot title abbreviation was checked against SVG source and
  was not misreported as a generator defect.

The current-source public report SHA-256 is:

```text
61619c229fd786698b37f456e0cfee7568db198a0842bac4a0903d7faf5005c1
```

Final current-source repository verification:

```text
Stage 172-176 focused regression: 46 passed in 10.02s
full repository Ruff lint: passed
three-file Ruff format check: passed
full repository pytest: 1029 passed, 1 warning in 18.46s
full pytest PID / exit code: 29920 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. The
full suite ran in one process, and the same PowerShell command waited on that
PID with one `Wait-Process` call until natural completion.
