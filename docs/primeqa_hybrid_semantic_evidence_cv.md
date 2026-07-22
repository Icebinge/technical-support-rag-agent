# Stage 173 Frozen Cross-Encoder Semantic Evidence CV

## Objective

Stage 173 evaluates the user-selected A route: add direct question-to-passage
semantic scores from the already cached
`cross-encoder/ms-marco-MiniLM-L-6-v2` model, without fine-tuning that model.
The experiment remains train-only. It does not load development or test, run
answer generation or Agent turns, add retries or fallbacks, or alter the
default runtime.

This stage is a screening experiment. It must pass every grouped nested-CV
quality gate and every outer-fold safety gate before any runtime E2E work is
authorized.

## Frozen Protocol

The local snapshot revision is
`c5ee24cb16019beea0893ab7796b1df96625c6b8`. Six required model files and the
four upstream data/report sources are authorized by SHA-256 before loading.
The model remains frozen and produces one scalar relevance score per
question-document pair.

Each question uses the same Stage 172 evidence views:

```text
initial: current-query-overlap Top10
final:   deduplicated union of initial Top10 and original-RRF Top10
```

Every document in the final union is scored exactly once; the initial view
reuses its subset of those scores. The bounded text policy uses the full
question and the document title plus a query-aware 1,600-character body
window. Cross-encoder maximum input length is 512 tokens, GPU batch size is 64,
and resource/progress capture occurs after each 512-pair event batch.

Twelve runtime-safe semantic features summarize each view:

```text
maximum and second score
top1-top2 margin
top3 mean, overall mean, median, standard deviation, and range
nonnegative-score fraction
top semantic document baseline inverse rank
final-view gain over the initial maximum
whether the final top semantic document came from the alternate-only set
```

Two profiles are compared:

```text
semantic_only: 12 semantic + phase + visible count = 14 features
hybrid:        Stage 172's 29 + 12 semantic          = 41 features
```

Balanced logistic regression and balanced histogram GBDT are crossed with both
profiles and 17 thresholds from 0.10 through 0.90. The frozen grid therefore
contains 68 specifications. Model/profile/threshold selection uses grouped
nested five-fold CV exactly as in Stage 172. Gold labels are fit/evaluation
targets only and are never runtime features or cross-encoder inputs.

## Semantic Diagnostics

The frozen cross-encoder scored 9,714 unique question-document pairs:

```text
positive pairs:                       267
negative pairs:                     9,447
pair-level ROC AUC:                0.821097
view-maximum ROC AUC:              0.756961
positive document ranked top1: 156 / 267 = 0.584270
```

Score distributions overlap materially:

```text
                         p25       median       p75
positive pairs         2.737359    4.725685    6.619828
negative pairs        -2.851084    0.171207    2.728148
sufficient view max    3.745591    5.623125    7.218841
insufficient view max  0.889506    2.678299    4.723242
```

The positive-pair first quartile is almost identical to the negative-pair
third quartile. The frozen retriever relevance model has useful ranking signal,
but its unadapted score is not a direct evidence-entailment boundary.

## Nested-CV Result

No outer inner loop finds an eligible specification. Four outer folds select
the semantic-only profile, while fold 4 selects hybrid. Four choose histogram
GBDT and fold 5 chooses logistic; thresholds are 0.70 or 0.75.

The combined one-shot outer OOF result is:

```text
balanced accuracy:                    61.3567%
ROC AUC:                              74.5437%
initial-visible compose:              30.8571%  fail
alternate-only inspect:               93.4783%  pass
alternate-only final compose:         15.2174%  fail
alternate-only exact path:             9.7826%  fail
insufficient final compose:           13.8983%  aggregate pass
```

Fold 5 has an insufficient-final compose rate of 20.9677%, so the mandatory
every-fold safety condition also fails. The other fold safety rates are
10.1695%, 18.8679%, 6.6667%, and 13.1148%.

Compared with Stage 172:

```text
balanced accuracy:                    +0.9680 pp
ROC AUC:                              -1.7473 pp
initial-visible compose:              +2.8571 pp
alternate-only inspect:               -3.2608 pp
alternate-only final compose:         +4.3478 pp
alternate-only exact path:            +2.1739 pp
insufficient final compose:           +5.0847 pp
```

The full-train OOF diagnostic selects hybrid histogram GBDT at threshold 0.75.
It is safe in all five folds, with ROC AUC 0.781251, initial compose 44.0%, final
compose 15.2174%, exact path 10.8696%, and false compose 13.5593%. It remains
ineligible because three required recall/path gates fail.

The formal decision is:

```text
candidate_selected = false
status = stage173_frozen_cross_encoder_semantics_insufficient
```

No model or threshold is registered and runtime E2E remains blocked.

## Resource Consumption

Formal PID `21616` completed naturally on its first run:

```text
wall time:                         123.669439 s
process CPU time:                  448.328125 s
candidate replay:                   88.873777 s
cross-encoder load:                  0.147502 s
pair build and semantic scoring:    26.706377 s
pure semantic scoring:              22.451182 s
nested CV:                           7.735781 s
semantic throughput:              432.672093 pairs/s
process peak working set:            4.287 GiB
process peak private usage:           6.653 GiB
minimum system available memory:      2.866 GiB
CUDA peak allocated:                  0.626 GiB
CUDA peak reserved:                   0.900 GiB
```

The 19 semantic event batches and six phase boundaries produce 25 in-process
snapshots. This is event-driven work instrumentation, not an external monitor.
The process was started once and the same PowerShell command called
`Wait-Process` once for PID `21616` until natural exit. There was no polling,
PowerShell wait timeout, kill, restart, retry, or partial continuation.

## Visualizations

Eight SVGs cover quality gates, Stage 172/173 rate comparison, fold safety,
fold exact path, selected feature profiles, semantic diagnostics, timing, and
resource peaks. All eight parse as XML and render successfully with independent
Edge profiles. Key charts were visually inspected and their labels, bars, and
values match the JSON report. At conversation-image scale, two long titles
appear abbreviated as `S3`; direct SVG inspection retains the complete
`Stage 173` title text. Edge stderr contains only the existing QQBrowser-profile
notice and all PNGs are written successfully.

## Process Corrections

The first read-only repository/model-cache search returned exit 1 because the
second candidate Hugging Face cache directory did not exist. Valid results from
the first directory had already been printed and no files were changed.

A subsequent read-only command returned exit 1 after its valid model and source
reads because the final `rg` included a Windows wildcard and nonexistent lock
files. Another search printed a Windows-wildcard error while the overall
PowerShell command exited 0 after later reads succeeded. None changed files.

The first model-hash command failed in PowerShell parsing because a `foreach`
statement was piped directly. No hash ran and no file changed. Collecting the
objects before piping then authorized all model files and exposed their real
resolved sizes.

The implementation's first Ruff and focused pytest run passed: nine Stage 173
tests passed, followed by 26 Stage 169/172/173 regression tests. A synthetic
two-pair GPU smoke test also succeeded before formal execution: the relevant
pair scored 9.4798, the irrelevant pair -11.0890, and peak CUDA allocated was
0.094 GiB. This smoke test used no dataset row.

The formal run succeeded on its first attempt. The tool channel yielded once
while the original command and its direct `Wait-Process` remained active;
waiting on that existing cell did not launch another shell command or poll the
PID.

## Artifact

The public report is
`artifacts/primeqa_hybrid_semantic_evidence_cv_stage173.json`, SHA-256:

```text
b75c3aea469cbe22fb5581210e0d96afb9094502aaa36d9c21aa60c22db9b366
```

All 21 process guards pass. The report contains no raw question, answer,
document/passage text, pair/sample identity, generated output, or gold document
ID.

## Next Boundary

Stage 174 is not runtime E2E. The measured pair-level AUC and semantic-only
selection in four folds make supervised train-only cross-encoder adaptation a
reasonable next candidate, but it is a distinct user-choice route, not an
automatic fallback from A. If authorized, its training and threshold selection
must remain grouped and nested, preserve per-fold safety, and keep development
and test closed.

## Repository Verification

```text
Stage 169/172/173 focused regression: 26 passed in 3.53s
full repository Ruff lint: passed
three-file Ruff format check: passed
full repository pytest: 1001 passed, 1 warning in 11.41s
full pytest PID / exit code: 5992 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation warning.
The full suite started one process, and the same PowerShell command waited once
on that PID until natural completion.
