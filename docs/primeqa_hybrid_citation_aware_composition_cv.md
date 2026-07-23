# Stage 180 Runtime-Visible Citation-Aware Composition CV

## Objective

Stage 180 tests whether answer composition can convert the Stage 178A
listwise Top10 context into more gold citations and better answer F1. It uses
only the frozen 562-row train split and the authenticated Stage 178A five-fold
OOF document scores. Development and test remain closed.

Gold answers and gold document ids may be used only as labels inside an
outer fold's training partition and for offline held-out scoring. A runtime
policy may see only the question, route, candidate sentence, candidate score,
document title, document retrieval score, and listwise context rank.

## Nested Five-Fold Protocol

For each outer fold, the other four frozen folds form the selection data.
Every learned policy is evaluated on those four folds by four-way inner OOF:
three folds fit both model heads and the fourth is scored. Rule policies have
no fit and are evaluated on the same inner rows. No outer-fold outcome may
participate in policy selection.

After selection, both learned heads are fit once on all four inner folds. The
selected policy is evaluated exactly once on the outer fold. Across five
outer folds this produces 20 inner partitions, five outer fits, and exactly
50 model-head fits: 25 gold-document classifiers plus 25 sentence-F1
regressors.

The fixed selection order is:

```text
1. maximize minimum inner-fold gold-citation delta
2. maximize aggregate inner-OOF gold-citation delta
3. maximize minimum inner-fold answerable F1 delta
4. maximize aggregate inner-OOF answerable F1 delta
5. minimize F1 regression count
6. minimize changed-answer count
7. stable policy id
```

There is no fallback. A policy is always selected by the frozen ordering and
must subsequently pass the outer-OOF quality gates to advance.

## Frozen Policy Family

The eight rule policies are:

```text
rule_score_rank_p000_cap1
rule_score_rank_p025_cap1
rule_score_rank_p050_cap1
rule_score_rank_p100_cap1
rule_score_rank_p025_cap3
rule_score_rank_p050_cap3
rule_score_rank_p100_cap3
rule_context_rank_coverage_top3
```

`score_rank` divides the existing sentence score by
`log2(context_rank + 1) ** rank_power`. `cap1` permits one sentence per
document; `cap3` preserves the current maximum of three. Context-rank
coverage takes the best eligible sentence from context ranks 1, 2, and 3,
then fills any empty slots by the original sentence order.

The six learned policies combine a gold-document probability head and a
sentence-token-F1 regression head with citation weights `0.25`, `0.50`, and
`0.75`, each under document caps 1 and 2. Both heads consume exactly the same
runtime-safe feature dictionary. Scores are clipped to `[0, 1]` before the
fixed weighted combination.

## Advancement Gates

Stage 180 advances only if all process guards pass and the combined outer OOF
candidate satisfies all of these frozen gates against the Stage 178A
listwise-plus-top3 composition baseline:

```text
gold-citation delta >= 1
gold-citation paired-bootstrap 95% CI lower >= 0
answerable mean F1 delta > 0
answerable F1 paired-bootstrap 95% CI lower >= 0
gold-citation non-regression in at least 4 of 5 folds
answerable F1 non-regression in at least 4 of 5 folds
answerable refusal count does not increase
unanswerable false-answer count does not increase
generation-context gold-hit count is exactly unchanged
candidate request p95 overhead <= 0.050 seconds
no retry or fallback action
```

Passing authorizes only a frozen follow-up validation. It does not authorize
default runtime activation, Stage 178B, or development/test access.

## Formal Result

The third formal attempt completed the frozen protocol on all 562 train rows.
All 21 process guards passed. The run executed 20 inner partitions, fit 50
model heads, and completed 1,686 Agent turns with no retry or fallback.
Development and test remained closed, Stage 178B was not run, and the default
runtime was unchanged.

The five outer folds selected these policies:

```text
fold 1  rule_context_rank_coverage_top3
fold 2  dual_c50_f50_cap1
fold 3  dual_c75_f25_cap1
fold 4  rule_context_rank_coverage_top3
fold 5  dual_c75_f25_cap1
```

The paired Agent result was:

```text
metric                         baseline   candidate   delta
answerable average token F1   0.2004     0.1999      -0.0005
gold citation count           183        214         +31
gold citation rate            0.4959     0.5799      +0.0840
context gold-hit count         257        257          0
request p95 seconds           0.144237   0.148896    +0.004659
```

The 2,000-replicate paired-bootstrap intervals were
`[-0.010720, 0.009508]` for answer F1 delta and
`[0.051351, 0.118919]` for gold-citation-rate delta. Citation improved in all
five folds, while F1 changed by `+0.0069`, `+0.0104`, `-0.0160`, `-0.0015`,
and `-0.0031`; only two folds were non-regressing. Context-to-citation
conversion improved from `183/257` (`0.712062`) to `214/257` (`0.832685`).

The candidate passed 8 of 11 quality gates. It failed the strict positive F1
gain, nonnegative F1 confidence-bound, and four-of-five F1 fold-nonregression
gates. The formal status is therefore
`stage180_citation_aware_composition_insufficient`, with
`candidate_selected=false`. This policy family must not advance to frozen
validation or default runtime activation.

The completed run took `279.823876` wall-clock seconds. Peak working set and
private usage were `4,774,424,576` and `5,700,812,800` bytes, and minimum
system-available memory was `2,069,917,696` bytes. The in-process tracker
reported zero CUDA allocation and reservation for this run. The public report
SHA-256 is:

```text
3605db66c11a3a9f527bfe44f9a442e6d139b114766c8d7d0edd2a0286f53be1
```

## Attempt Audit

Attempt 1, PID `3416`, completed data collection, nested selection, and paired
Agent evaluation, then failed while writing visualizations because the caller
used the unsupported keyword `left_margin` instead of `margin_left`. It emitted
an insufficient decision but no official JSON report, so it is not treated as
a completed formal experiment. Its logs remain under
`artifacts/stage180_attempt1_visualization_api_failure/`.

After correcting the visualization API and adding a regression test, attempt
2, PID `20308`, emitted only `sources_authorized` and then exited without a
traceback or report. No Windows crash, OOM event, or dump confirmed the cause;
only low available memory (`3.401 GiB`) was observed, so OOM is not claimed as
fact. Its logs remain under
`artifacts/stage180_attempt2_silent_exit_after_authorization/`.

Attempt 3, PID `16120`, was started after available memory increased to
`5.421 GiB`. One PowerShell command called `Wait-Process` once for that PID and
waited for natural termination. The Windows `Start-Process` object's child
`ExitCode` field was blank, so no child exit code is asserted; the outer shell
returned zero, the complete report and all six SVGs were written, stderr held
only model-weight loading progress, and every process guard passed.

## Verification

All six SVG files parsed as XML. Rendering four key charts to PNG exposed a
real overlap in the signed fold-F1 chart: the largest negative value label
occupied the category-label area. The shared horizontal-bar renderer was
updated to move a colliding negative value label inside its bar with
high-contrast text, and a regression test now covers both colliding and
noncolliding negative labels. The corrected fold-F1 chart was rendered again
with an isolated browser profile and visually checked for complete title,
separate category labels, and unobscured values.

Final current-source verification produced:

```text
Stage 180 focused tests: 12 passed in 1.02s
six Stage 180 Python files: Ruff format check passed
full repository Ruff lint: passed
full repository pytest: 1060 passed, 1 warning in 29.90s
full pytest PID / outer command exit: 26808 / 0
```

The full pytest command started one process and called `Wait-Process` once
until it ended naturally. The process object reported `HasExited=True`; its
child `ExitCode` field was blank and is not represented as zero. Stderr was
empty. The warning is the existing FastAPI/Starlette `TestClient` deprecation.
