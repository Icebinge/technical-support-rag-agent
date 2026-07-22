# Stage 172 Evidence-Entailment Nested Cross-Validation

## Objective

Stage 172 tests whether a dedicated binary classifier can replace Stage 171's
unsafe generative evidence layer. It predicts whether the evidence currently
visible to the Agent contains the answer document. The experiment uses only
the frozen 562-question train split. Development, test, answer generation,
Agent execution, retries, fallbacks, and default runtime activation remain
closed.

This is a candidate-selection experiment, not a runtime integration. A failed
quality result is a valid experiment outcome and does not authorize Stage 173
end-to-end evaluation.

## Frozen Protocol

Each train question produces two evidence views:

```text
initial: current-query-overlap Top10
final:   deduplicated union of initial Top10 and original-RRF Top10
```

The two views share one question group and one existing fold ID. The sufficient
label is true only when the answerable question's answer document occurs in
that view. Gold labels are used only to fit and evaluate models; they are not
model features and private rows are not written to the public report.

The 29 runtime-safe inputs consist of the 27 frozen Stage 167 aggregate
retrieval/evidence features plus `phase_final` and `visible_document_count`.
No answerability label, gold document ID/rank, sample identity, question text,
selected action, or generated output is accepted by the classifier.

The candidate grid was frozen before execution:

```text
model families: balanced logistic regression, balanced histogram GBDT
thresholds:     0.10 through 0.90 in steps of 0.05
total specs:    2 x 17 = 34
```

Five-fold nested grouped cross-validation is used. For each outer held-out
fold, predictions from leave-one-training-fold-out inner fits select one model
family and threshold. That selected model is refit on the four outer training
folds and predicts the outer held-out fold once. A separate full-train OOF pass
selects the eventual refit specification without using development or test.

The five aggregate gates are:

```text
initial-visible compose              >= 70%
alternate-only initial inspect       >= 50%
alternate-only final compose         >= 70%
alternate-only exact path            >= 40%
insufficient final compose           <= 20%
```

The last safety gate must also pass independently in every outer fold.

## Dataset Result

The exact frozen inputs were authorized by SHA-256 before data loading. The
formal replay produced 112,400 Top200 candidate rows and 1,124 evidence views:

```text
train questions:                       562
sufficient views:                      442
insufficient views:                    682
initial gold visible questions:        175
alternate-only gold visible:            92
union gold missing / candidate hit:     78
candidate-pool gold missing:            25
unanswerable:                           192
```

## Cross-Validation Result

No inner fold found an eligible specification. The nested outer selections
were histogram GBDT at threshold 0.80 for folds 1-4 and threshold 0.85 for fold
5. Their combined one-shot OOF result is:

```text
balanced accuracy:                    60.3887%
ROC AUC:                              76.2910%
initial-visible compose:              28.0000%  fail
alternate-only inspect:               96.7391%  pass
alternate-only final compose:         10.8696%  fail
alternate-only exact path:             7.6087%  fail
insufficient final compose:            8.8136%  pass
```

The final full-train OOF diagnostic also selects histogram GBDT, threshold
0.75. It is safe in all five folds but remains ineligible:

```text
balanced accuracy:                    63.5269%
ROC AUC:                              76.2910%
initial-visible compose:              37.1429%
alternate-only inspect:               93.4783%
alternate-only final compose:         15.2174%
alternate-only exact path:             8.6957%
insufficient final compose:           13.5593%
```

The classifier has ranking signal, but the aggregate features do not provide
enough separation at the required safety operating point. Raising the
threshold protects against unsupported composition while suppressing most
supported initial and final views. This is not one anomalous fold:

```text
fold    insufficient final compose    alternate exact path
1       13.5593%                        0.0000%
2        9.4340%                        0.0000%
3        8.3333%                       22.2222%
4        6.5574%                        5.8824%
5        6.4516%                       13.3333%
```

All folds pass safety, while all fall far below the 40% exact-path target.
Consequently:

```text
candidate_selected = false
status = stage172_no_grouped_oof_safe_evidence_classifier
```

No classifier is registered, Stage 173 runtime E2E is not authorized, and the
default runtime remains unchanged.

## Resource Consumption

Formal PID `32332` was started once and waited to natural exit by one direct
PowerShell `Wait-Process` in the same command. There was no process polling,
wait timeout, kill, restart, retry, or partial continuation.

```text
wall time:                              99.675895 s
process CPU time:                      405.578125 s
source authorization:                   0.134158 s
evidence replay and case construction: 94.862135 s
nested CV:                               4.674951 s
report assembly:                         0.004651 s
process peak working set:                4.273 GiB
process private usage at end:            5.020 GiB
minimum boundary system memory:          3.669 GiB
GPU model loaded:                        no
generation calls:                        0
```

Resource measurement uses process-boundary snapshots without a monitoring
loop. Candidate replay dominates 95.17% of wall time; the complete nested CV
grid takes only 4.69%.

## Visualizations

Six SVGs cover aggregate quality gates, OOF proxy rates, per-fold safety,
per-fold exact path, inner eligible-spec counts, and label distribution. All
six pass XML parsing and all six rendered to PNG with independent Edge
profiles. The key charts were visually inspected: titles, labels, values, and
bars are present and the plotted values match the JSON report. Edge stderr
contains only the existing QQBrowser-profile notice; all render processes exit
successfully.

## Process Corrections

The initial read-only audit used a Windows wildcard path directly with `rg` and
returned exit 1. A later search used `src\\ts_rag_agent\\config*`, which is also
an invalid Windows path expression and returned exit 1 after other matches had
already printed. Neither search changed files.

The full-train OOF patch produced more tool output than the conversation could
display, so its result was treated as unknown rather than assumed successful.
An explicit source search confirmed every intended block before further work.

The first focused check found one unused test import and one visualization
defect: three ratio charts omitted the required `x_label`. Ruff failed and the
test result was `8 passed, 1 failed`. Removing the import and supplying the
three labels produced `26 passed` across the Stage 167/169/172 focused
regression set.

The combined documentation patch then failed atomically because the learning
journal's actual final context differed from its terminal rendering. No file
was changed by that failed patch; the document and journal were applied
separately afterward.

The formal Stage 172 run succeeded on its first attempt. The tool channel
yielded once while the original shell command and its direct `Wait-Process`
remained active. Waiting on that existing tool cell did not launch another
shell command or poll the PID.

## Artifacts

The public report is
`artifacts/primeqa_hybrid_evidence_entailment_cv_stage172.json`, SHA-256:

```text
48d0309e98d044f2cc89fa42526ef9c5da1c8bf9e7b2e188a60c372f8c7dd827
```

All 17 process guards pass. The report contains no raw question, answer,
document text, generated output, private sample ID, or gold document ID.

## Next Boundary

Stage 173 must not be runtime E2E. The next train-only experiment should
replace aggregate evidence-view classification with direct question-to-passage
semantic evidence scoring, then apply grouped nested CV under the same safety
gates. Development and test remain closed until a candidate passes every
train-only OOF gate and every-fold safety.

## Repository Verification

```text
Stage 167/169/172 focused regression: 26 passed in 2.71s
full repository Ruff lint: passed
three-file Ruff format check: passed
full repository pytest: 992 passed, 1 warning in 11.30s
full pytest PID / exit code: 30464 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation warning.
The full suite started one process, and the same PowerShell command waited once
on that PID until natural completion.
