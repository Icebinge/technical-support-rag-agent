# Stage 177 Listwise Second-Stage Reranker OOF Evaluation

## Objective

Stage 177 tests whether the Stage 176 `listwise_none` model is useful as a
pure second-stage document reranker even though it failed as an evidence
sufficiency gate. It compares grouped out-of-fold listwise ranking against
the original RRF order and the frozen cross-encoder order.

This experiment remains train-only. It does not load development or test
data, run answer generation or Agent turns, save a model checkpoint, execute
the failed sufficiency gate, introduce retries or fallbacks, or alter the
default runtime.

## Frozen Protocol

The experiment uses the same frozen candidate replay and local
`cross-encoder/ms-marco-MiniLM-L-6-v2` snapshot as Stages 173-176. It performs
five grouped OOF fits, one per held-out fold, with the fixed Stage 176
`listwise_none` training family. There is no remaining family, threshold, or
calibration selection in this stage.

Three ranking methods are evaluated on exactly the same final-union candidate
sets:

```text
original_rrf:          original fused candidate order
frozen_cross_encoder:  untrained local cross-encoder score order
listwise_oof:          held-out score from five grouped supervised fits
```

Gold rank, mean reciprocal rank, and Recall@1/3/5/10 are reported both
conditionally on gold being present in the candidate pool and against all
answerable train questions. Two thousand paired bootstrap replicates compare
`listwise_oof` with each baseline.

The predeclared advancement gates require, against both baselines:

```text
MRR delta 95% CI lower bound:       > 0
Recall@3 delta 95% CI lower bound: >= 0
Recall@10 delta CI lower bound:    >= -0.02
MRR fold wins:                     >= 4 / 5
```

Thirteen report, split, document, and model sources are SHA-256 authenticated
before loading.

## Execution Checks

Eight focused Stage 177 tests passed. The related Stage 172-177 regression
set passed 54 tests in 9.19 seconds. Source authorization independently
confirmed 13/13 sources and the exact Stage 176 report hash.

Stage 177 reuses the unchanged GPU trainer that completed the real synthetic
backpropagation smoke in Stage 176; no redundant Stage 177-only GPU smoke was
run. The formal process was PID `35692`. It completed all five OOF fits and
511 optimizer steps, passed all 20 process guards, and exited `0` on its first
formal launch. There was no OOM, retry, fallback, restart, or partial run.

## Candidate-Pool Boundary

There are 370 answerable train questions. The fixed final-union candidate
pool contains a gold document for 267 of them:

```text
gold-present questions:      267 / 370
candidate-pool gold coverage: 0.721622
gold-absent questions:       103 / 370
complete scored pairs:       9,714
```

The reranker can only reorder the 267 gold-present questions. It cannot
recover any of the 103 questions whose gold document is absent from the
first-stage candidate pool. Conditional metrics therefore measure ranking
quality, while all-answerable metrics preserve the true end-to-end retrieval
ceiling.

## Formal Result

Conditional on gold being present, the three methods are:

```text
method                 MRR       mean rank  median  R@1       R@3       R@5       R@10
original_rrf           0.698964  2.734082   1.0     0.573034  0.782772  0.861423  0.955056
frozen_cross_encoder   0.708886  2.745318   1.0     0.584270  0.805243  0.868914  0.947566
listwise_oof           0.767388  2.337079   1.0     0.655431  0.853933  0.898876  0.962547
```

Against all 370 answerable questions, Recall@K is:

```text
method                 R@1       R@3       R@5       R@10
original_rrf           0.413514  0.564865  0.621622  0.689189
frozen_cross_encoder   0.421622  0.581081  0.627027  0.683784
listwise_oof           0.472973  0.616216  0.648649  0.694595
```

The listwise reranker improves conditional MRR by 0.068424 versus original
RRF and by 0.058502 versus the frozen cross-encoder. It also improves
conditional Recall@3 by 0.071161 and 0.048689, respectively. Recall@10
changes are smaller because both baselines already retain most gold-present
documents within ten positions.

Paired bootstrap 95% intervals are:

```text
baseline                metric      delta     CI lower   CI upper
original_rrf            MRR         0.068424  0.028249   0.106332
original_rrf            Recall@3    0.071161  0.022472   0.116105
original_rrf            Recall@10   0.007491 -0.018727   0.033708
frozen_cross_encoder    MRR         0.058502  0.028386   0.088318
frozen_cross_encoder    Recall@3    0.048689  0.007491   0.086142
frozen_cross_encoder    Recall@10   0.014981 -0.007491   0.041199
```

Listwise MRR wins 4/5 folds against original RRF and 5/5 against the frozen
cross-encoder. All eight predeclared quality gates pass. Mean training loss
decreases from 1.405709 to 1.054080, confirming that the five OOF models
learned the listwise objective.

## Resources

```text
wall time:                       276.893420 seconds
candidate replay:                 93.764750 seconds
frozen pair build and scoring:    27.025545 seconds
listwise grouped OOF:             155.618333 seconds
OOF inference:                     25.116566 seconds
OOF inference throughput:         386.756693 pairs/second
estimated 20-pair query ranking:    0.051712 seconds
process working-set peak:           4.644 GiB
GPU allocated peak:                 2.954 GiB
GPU reserved peak:                  5.541 GiB
minimum system available memory:    1.419 GiB
generation calls:                   0
```

The estimated per-query number covers model inference over 20 already-built
pairs. It does not include first-stage retrieval, candidate construction, or
document text loading, so it is not yet an end-to-end latency measurement.

The formal PowerShell command started one process and waited on the same PID
with one `Wait-Process` call until natural completion. No polling loop,
segmented monitor, timeout, or separate status command was used.

## Visual Verification

Eight SVG charts cover method MRR, conditional recall, all-answerable recall,
quality gates, per-fold MRR, training loss, timing, and resources. All 8/8
parse as XML. Four key charts were independently rendered to PNG and visually
checked; they are nonblank, fit their canvases, and match the JSON values.
Edge emitted only its existing QQBrowser profile warning.

## Decision

The formal status is `advance_to_stage178_listwise_reranker_agent_e2e` and
`candidate_selected=true`. This is the first candidate in the Stage 172-177
branch to pass every frozen advancement gate.

Authorization is limited to a Stage 178 Agent retrieval-chain end-to-end
experiment. Development and test remain unopened, no checkpoint has yet been
saved or registered, the sufficiency gate remains disabled, and the default
runtime is unchanged. Stage 178 must measure retrieval quality, answer and
citation behavior, and real request latency against the frozen runtime
baseline before any default activation is considered.

## Process Corrections

- The first Ruff run found two unused imports, `json` and `asdict`. They were
  removed, after which Ruff and all eight focused tests passed.
- A parallel preflight wrapper reached its approximately 10.5-second tool
  timeout after printing `54 passed`. Source authorization and format-check
  outputs were hidden, so neither was assumed successful; both were rerun
  independently and passed.
- A later combined Ruff/search check returned only the search command's
  exit `1` because its guessed filename pattern matched nothing, hiding the
  parallel Ruff outputs. The real test filenames were found separately and
  both Ruff checks were rerun independently and passed.
- The final focused-regression wrapper hit the tool channel's approximately
  14-second default timeout while its single pytest child was still running.
  The child was not interrupted and naturally completed `54 passed in
  15.50s`; one post-completion process check confirmed it had exited. There
  was no polling loop.
- An initial documentation lookup used a guessed Stage 176 filename that did
  not exist. It made no file changes; the real document name was then found
  with `rg`.
- A combined read command consequently returned only the missing-file error.
  The journal, ignore rules, and project configuration were reread with valid
  paths before documentation was written.
- The first compact report extraction used obsolete guessed JSON paths and
  returned blank metric fields. No values were inferred from blanks; the
  actual schema keys were enumerated and every reported value was then read
  from the real report.

The current-source public report SHA-256 is:

```text
6e028ed9e90fe153fda39f3073861c5d0b8eb019675635edb21a6825a472be50
```

Final current-source repository verification:

```text
Stage 172-177 focused regression: 54 passed in 15.50s
full repository Ruff lint: passed
three-file Ruff format check: passed
full repository pytest: 1037 passed, 1 warning in 20.15s
full pytest PID / PowerShell command exit: 4016 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. The
full suite ran in one pytest process, and the same PowerShell command used one
`Wait-Process` call on PID `4016` until natural completion. The pytest output
proved success and the enclosing command exited `0`; the retained
`Start-Process` object's child `ExitCode` field printed blank after
`Wait-Process`, so no child exit-code value is inferred or reported.
