# Stage 174 Supervised Cross-Encoder Nested CV

## Objective

Stage 174 evaluates whether train-only supervised adaptation can turn the
Stage 173 frozen MS MARCO cross-encoder into a reliable evidence-sufficiency
classifier. The experiment uses only the frozen 562-row train split. It does
not load development or test data, run answer generation or Agent turns,
write a model checkpoint, add retry or fallback behavior, or change the
default runtime.

This is a screening experiment. A candidate can advance only if its grouped
nested-CV operating point passes every aggregate quality gate, every inner
selection gate, and every outer-fold safety gate.

## Frozen Protocol

The local base snapshot is
`cross-encoder/ms-marco-MiniLM-L-6-v2` revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`. Ten upstream report, split,
document, and model-file sources are authenticated before data loading.

The protocol performs 25 fresh fits:

```text
5 outer folds x (4 inner fits + 1 held-out outer fit) = 25 fits
```

Questions, including their initial/final evidence views and all sampled
pairs, remain in one grouped fold. Each fit starts from a fresh local model
and trains all parameters with pointwise weighted binary cross-entropy:

```text
epochs:                         2
training batch size:            32
inference batch size:           64
learning rate:                  2e-5
weight decay:                   0.01
gradient clipping:              1.0
maximum sequence length:        512
positive-group negatives:       top 4 by frozen score
negative-only group negatives:  top 2 by frozen score
positive class weight:          fit-local negative / positive count
```

The decision score for an evidence view is the maximum sigmoid probability
among that view's candidate pairs. No second learner is fitted. The threshold
grid was expanded before the formal run from the inherited 17 values to 21
frozen values because supervised BCE probabilities can require a safety
operating point above 0.90:

```text
0.05, 0.10..0.90 in 0.05 increments, 0.95, 0.975, 0.99
```

The expansion was made before any formal Stage 174 data result existed and
was not tuned after observing the outcome.

## Formal Result

The formal process was PID `31148` and exited `0` after one actual launch.
It completed all 25 fits, 1,952 optimizer steps, 9,714 complete pairs, and
1,124 grouped evidence-view cases. All process guards passed.

The strict nested OOF operating points selected thresholds 0.90 or 0.95 in
the five outer folds because no inner loop contained an eligible threshold.
The resulting held-out OOF metrics were:

```text
balanced accuracy:                    0.563853
ROC AUC:                              0.749272
initial-visible compose:              0.200000  fail (>= 0.70)
alternate-only inspect:               0.989130  pass (>= 0.50)
alternate-only final compose:         0.010870  fail (>= 0.70)
alternate-only exact path:            0.000000  fail (>= 0.40)
insufficient final compose:           0.040678  pass (<= 0.20)
```

Compared with Stage 173, AUC increased by only 0.003835 while balanced
accuracy fell by 0.049714. The high thresholds reduced false compose by
0.098305, but also reduced initial compose by 0.108571, alternate-only final
compose by 0.141304, and exact-path success by 0.097826.

All five outer folds passed the false-compose safety ceiling. Their held-out
false-compose rates were 0.016949, 0.037736, 0.033333, 0.016393, and 0.096774.
This safety result is not enough for selection because every inner loop had
zero eligible thresholds and three aggregate utility gates failed.

For diagnosis only, the complete-train five-fold OOF predictions selected
threshold 0.85. That operating point reached balanced accuracy 0.615431 and
false compose 0.132203, but initial compose 0.405714, alternate-only final
compose 0.076087, and exact path 0.043478 still failed badly. It is not a
deployment estimate and does not override the outer nested result.

Training loss decreased from a mean first-epoch loss of 1.505392 to a mean
final-epoch loss of 1.074603. The model therefore did learn the pointwise
objective; the failure is the mismatch between that objective/score and the
required view-level routing trade-off, not an OOM or an aborted fit.

## Resources

```text
wall time:                       762.527954 seconds
candidate replay:                 88.315168 seconds
frozen pair build and score:      26.977440 seconds
nested fine-tuning:              646.789325 seconds
fine-tuning compute:             508.768841 seconds
fold inference:                  126.432752 seconds
process working-set peak:          4.637 GiB
process private-usage peak:        9.683 GiB
GPU allocated peak:                2.954 GiB
GPU reserved peak:                 3.494 GiB
minimum system available memory:   2.324 GiB
generation calls:                  0
```

Resource capture was event-driven inside the formal process. PowerShell
started one process and waited on that same process with `Wait-Process`; no
polling monitor, retry, restart, kill, partial continuation, or OOM occurred.

## Visual Verification

Eight SVG charts cover aggregate gates, Stage 173/174 rate comparison, outer
fold safety and path outcomes, selected thresholds, training loss, timing,
and resources. All files parse and the three key charts were rendered to PNG
and visually checked against the JSON values. Edge emitted only its existing
QQBrowser profile warning. Its headless rendering abbreviated one long chart
title to `S3`; direct SVG source inspection confirms the full
`Stage 173 versus Stage 174` title is present.

## Decision

The formal status is
`stage174_supervised_cross_encoder_insufficient` and
`candidate_selected=false`. The model is not saved or registered, runtime E2E
is not authorized, development and test remain unopened, and the default
runtime remains unchanged.

Pointwise pair classification plus a single view-max threshold has now failed
in both frozen and supervised forms. The next justified train-only experiment
is a grouped pairwise/listwise reranker objective that directly learns the
within-question gold-versus-hard-negative ordering and then calibrates a
separate view-level sufficiency margin. That must be a new explicit Stage 175
protocol, not an automatic fallback from this failed candidate.

## Process Corrections

- The first focused implementation run reported `7 passed, 2 failed`: one
  hard-negative expectation mistakenly included a positive row, and one test
  used attribute access on a serialized dictionary. Both tests were corrected
  before the formal run; the related regression then passed 27 tests.
- The 17-to-21 threshold-grid change was frozen before formal execution. A
  truncated patch response was treated as unconfirmed until source search
  verified the values on disk.
- One preflight pytest command referenced a nonexistent Stage 172 filename and
  therefore ran zero tests. The actual file was located and the corrected
  command passed 27 tests.
- The first source-authentication preflight assumed the wrong raw-data root and
  failed before loading data. Rebuilding the path from `ProjectSettings`
  authenticated all ten sources.
- A tool call used `timeout_ms: 0`, which the shell adapter interpreted as an
  immediate 11 ms timeout. It did not execute the PowerShell body, create a
  PID, or start formal training. The subsequent real command launched PID
  `31148` once and waited directly until natural completion.
- A rendered screenshot was initially mistaken for a chart-source title bug.
  Source inspection immediately showed the title was already correct, so no
  fabricated code fix was made.

The current-source public report SHA-256 is:

```text
7d949a300f58f3205397c76e3accf4c9a71932d466fa4811144f3cbc04b86019
```

Final current-source repository verification:

```text
Stage 172/173/174 focused regression: 27 passed in 4.24s
full repository Ruff lint: passed
three-file Ruff format check: passed
full repository pytest: 1010 passed, 1 warning in 13.13s
full pytest PID / exit code: 7780 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. The
full suite ran in one process, and the same PowerShell command waited on that
PID with one `Wait-Process` call until natural completion.
