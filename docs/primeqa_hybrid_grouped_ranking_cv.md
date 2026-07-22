# Stage 175 Grouped Ranking Nested CV

## Objective

Stage 175 tests whether question-grouped ranking objectives can fix the
pointwise evidence-classification failure observed in Stage 174. It compares
two predeclared model families under one train-only grouped nested-CV
protocol. Neither family is a fallback for the other.

The experiment loads only the frozen 562-row train split. Development, test,
answer generation, Agent turns, checkpoint writing, retries, fallbacks, and
default runtime activation remain closed.

## Frozen Protocol

Both families start each fit from the same local
`cross-encoder/ms-marco-MiniLM-L-6-v2` snapshot revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`:

```text
pairwise_anchor:
  RankNet gold-over-hard-negative loss
  + 0.5-weight fixed none=0 anchor loss

listwise_none:
  group softmax over gold/hard-negatives/fixed none=0
  negative-only questions target the none choice
```

Positive groups contain one gold pair and the four highest frozen-score hard
negatives. Negative-only groups contain the two highest frozen-score pairs.
Groups are packed without splitting under a 32-pair training budget. Every
fit trains all model parameters for two epochs with learning rate `2e-5`,
weight decay `0.01`, gradient clipping `1.0`, and maximum length 512.

Both families use the same evidence-view score:

```text
top1_logit - max(0, top2_logit)
```

The zero term is the explicit none anchor. Family and threshold are selected
together on inner OOF predictions from a 21-value frozen margin grid:

```text
-4, -3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0,
0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8
```

The complete protocol performs 50 fresh fits:

```text
5 outer folds x (2 families x 4 inner fits + 2 family outer fits) = 50
```

Both outer family fits are retained only as public aggregate OOF diagnostics.
The strict nested estimate uses the family and threshold selected without the
held-out fold. Eleven report, split, document, and model sources are
authenticated by SHA-256 before data loading.

## Execution Checks

The synthetic GPU smoke used no dataset rows. Both objectives completed real
backpropagation and 14/14 inference coverage. Pairwise loss changed from
0.976238 to 0.534559. The two-step listwise smoke changed from 1.637238 to
1.997387; with one batch per epoch this is not treated as a quality result,
only as confirmation that the execution path remained finite and complete.
Smoke CUDA allocated/reserved peaks were 0.445/0.490 GiB.

The formal process was PID `30416`. It completed all 50 fits and 4,094
optimizer steps, then exited `0` on its first actual launch. All process
guards passed.

## Formal Result

The strict family-selected nested OOF result was:

```text
balanced accuracy:                    0.639711
ROC AUC:                              0.759134
initial-visible compose:              0.468571  fail (>= 0.70)
alternate-only inspect:               0.956522  pass (>= 0.50)
alternate-only final compose:         0.141304  fail (>= 0.70)
alternate-only exact path:            0.097826  fail (>= 0.40)
insufficient final compose:           0.132203  aggregate pass (<= 0.20)
```

Compared with Stage 174, balanced accuracy increased by 0.075858, AUC by
0.009862, initial compose by 0.268571, final compose by 0.130434, and exact
path by 0.097826. False compose also increased by 0.091525, but remained under
the aggregate safety ceiling.

This is meaningful progress, not a passing result. Initial coverage recovered
substantially, while the alternate-only final path remained far below the
required level. Fold 5 also failed held-out safety: its pairwise-selected
false-compose rate was 0.258065, above the 0.20 ceiling. Therefore aggregate
safety alone cannot authorize the candidate.

Inner selection chose `listwise_none` in four outer folds and
`pairwise_anchor` in one. No family/threshold combination was eligible in any
inner loop. Complete outer-OOF family diagnostics selected threshold 1.0 for
both families:

```text
family             balanced acc  AUC       initial  final    exact    false
pairwise_anchor    0.613271      0.761082  0.354286 0.076087 0.054348 0.067797
listwise_none      0.622573      0.758645  0.388571 0.108696 0.076087 0.084746
```

The final train-OOF family choice is `listwise_none`, but it is not eligible
and no model is saved. Both objectives learned: mean pairwise loss decreased
from 0.771774 to 0.499491 and mean listwise loss from 1.428496 to 1.056601.
The remaining bottleneck is the view-level sufficiency score, especially the
paired transition where alternate retrieval adds the gold document.

## Resources

```text
wall time:                       1398.167229 seconds
candidate replay:                  92.531527 seconds
nested ranking:                  1278.125964 seconds
fine-tuning compute:             1002.993377 seconds
fold inference:                   250.956774 seconds
process working-set peak:           4.835 GiB
process private-usage peak:        13.278 GiB
GPU allocated peak:                 2.955 GiB
GPU reserved peak:                  6.814 GiB
minimum system available memory:    0.418 GiB
generation calls:                   0
```

No OOM occurred, but system-memory and CUDA-reserved headroom became narrow.
Future experiments must remain sequential and should not increase model size,
pair budget, or concurrent workers without a new resource design.

The formal PowerShell command started one process and called `Wait-Process`
on that same process until natural completion. There was no polling monitor,
PowerShell wait timeout, retry, restart, kill, fallback, or partial
continuation.

## Visual Verification

Nine SVG charts cover quality gates, Stage 174/175 rates, family selection,
family OOF quality, outer-fold safety/path, training loss, timing, and
resources. All nine parse successfully. Four key charts were rendered to PNG
and visually matched against the JSON report. Edge emitted only its existing
QQBrowser profile warning. The headless screenshot abbreviated long titles to
`S4`/`S5`; direct SVG inspection retains the full Stage 174/175 titles.

## Decision

The formal status is `stage175_grouped_ranking_insufficient` and
`candidate_selected=false`. Development and test remain unopened. No model is
registered, runtime E2E is not authorized, and the default runtime is
unchanged.

The next justified train-only experiment is to keep the selected
`listwise_none` objective and compare multiple predeclared view calibration
scores on the same grouped OOF logits: absolute top-1 logit, top-1/top-2
margin, candidate evidence mass versus none, and bounded combinations of
absolute and relative confidence. This directly tests whether the remaining
failure is calibration rather than ranking and requires only 25 fits instead
of another two-family 50-fit run.

## Process Corrections

- The first combined Ruff/test invocation found one Ruff `UP012` issue for an
  unnecessary UTF-8 argument. Because the parallel tool did not return pytest
  output, no test success was assumed. The style issue was corrected; the
  next Stage 175 run passed 10 tests and the Stage 172-175 regression passed
  37 tests.
- The initial planning estimate said 45 fits. Before formal execution it was
  corrected to 50 because both families need five complete outer-OOF
  predictions. Code, tests, and the process guard had already used 50; the
  formal run was never launched under a 45-fit protocol.
- The tiny listwise smoke loss increased across its two single-batch epochs.
  It is retained as an execution observation and was not presented as model
  quality evidence.
- Headless screenshots again abbreviated long SVG titles. Source inspection
  confirmed complete titles, so no false code fix was made.

The current-source public report SHA-256 is:

```text
27641cf6754762260a7400aa431762c5e8e34cf9f1645f4038fa8867cc04dec8
```

Final current-source repository verification:

```text
Stage 172/173/174/175 focused regression: 37 passed in 7.95s
full repository Ruff lint: passed
three-file Ruff format check: passed
full repository pytest: 1020 passed, 1 warning in 16.99s
full pytest PID / exit code: 13304 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. The
full suite ran in one process, and the same PowerShell command waited on that
PID with one `Wait-Process` call until natural completion.
