# Stage 161: Protected Generation-Context Selector Training

## Status

Stage 161 completed the user-selected route A as a train-only grouped five-fold
experiment. It trained six lightweight selectors over the exact Stage 116
Top200 candidate pool, compared them with the current query-overlap Top10 and
untouched RRF Top10 controls, and produced one public aggregate report plus ten
SVG charts.

All 18 process guards pass, but no selector passes the strict quality gates.
No configuration is selected, development and test remain closed, no model
artifact is written, and runtime behavior is unchanged.

The public report SHA-256 is
`a13b8ee5538581f0eb87a649c48fdf4ae715b6cfa8a43a97b5115001f9cd1197`.

## Frozen protocol

The only question split loaded is the exact Stage 68 train file:

```text
rows: 562
answerable / unanswerable: 370 / 192
SHA-256: cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155
```

The technote corpus SHA-256 is
`f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b`.
Stage 119, 121, 160, and 80 reports are authorized by exact fingerprints and
decision states before train is parsed. No development or test path exists in
the Stage 161 CLI.

Train rows are grouped by normalized question plus answer document, or the
`UNANSWERABLE` marker, and assigned to five folds by the established project
function. Every validation prediction is produced by a model fitted on the
other four folds. Gold document membership is used only as the train label;
the runtime feature list contains no gold field.

The candidate pool is the exact original Stage 116 RRF Top200. Generation
context depth remains ten. The six frozen configurations are the Cartesian
product of protected original-RRF prefix depths `3/5/7` and these models:

```text
pairwise logistic regression
pointwise balanced histogram gradient boosting
```

Pairwise training uses symmetric positive-minus-hard-negative feature
differences. Pointwise training uses one recoverable positive and the
deduplicated union of the first 20 baseline and first 20 query-overlap hard
negatives. Both models see runtime-visible retrieval ranks, route presence,
query/document overlap, special-token matches, and RRF score only.

Selection is intentionally strict. A candidate must improve current Top10 gold
hit count, remain at least as good as untouched RRF Top10, avoid aggregate F1,
citation, refusal, and unanswerable regressions, preserve the protected prefix,
and avoid hit-rate or F1 regression in every fold. There is no fallback when
no configuration passes.

## Formal result

The formal process started once with the local dense channels and explicit
CUDA query encoding. It had no runtime deadline, monitoring deadline,
automatic termination, restart, retry, or fallback. It ended naturally with
exit code 0 after `131.874477s`.

The exact Top200 pool contains `112,400` in-memory candidate records. Every row
has depth 200. Gold is present for `345/370` answerable train rows
(`93.2432%`). Raw candidate rows and case-level content are not written.

Control results are:

| Control | Gold in Top10 | Rate | F1, all answerable | Gold citations | Answerable refusals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current query overlap | 175 | 0.472973 | 0.194072 | 151 | 1 |
| Untouched original RRF | 255 | 0.689189 | 0.201990 | 177 | 1 |

The current control also reproduces completed-answerable F1 `0.194597`, which
rounds to the frozen `0.1946` reference, and exactly 151 gold citations. The
untouched RRF control is materially stronger than the currently deployed
query-overlap shortlister on this train OOF evaluation.

The six OOF model results are:

| Prefix and model | Gold in Top10 | F1, all | Citations | Minimum fold hit delta | Minimum fold F1 delta | Selectable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 + pairwise logistic | 224 | 0.210249 | 190 | +0.072464 | -0.000412 | No |
| 3 + histogram GBDT | 216 | 0.208407 | 188 | +0.057971 | +0.002797 | No |
| 5 + pairwise logistic | 244 | 0.210292 | 189 | +0.115942 | -0.001341 | No |
| 5 + histogram GBDT | 233 | 0.203880 | 189 | +0.101450 | -0.007573 | No |
| 7 + pairwise logistic | 250 | 0.204102 | 183 | +0.146667 | -0.005047 | No |
| 7 + histogram GBDT | 245 | 0.202255 | 183 | +0.133334 | -0.007408 | No |

All six improve aggregate context hit, F1, citation count, and answerable
refusals relative to the current query-overlap control. None reaches the
untouched RRF hit count of 255. Five also regress F1 in at least one fold; the
prefix-3 histogram model is the only model without a fold-level F1 regression,
but its 216 hits remain far below untouched RRF. Therefore the strict decision
is `no_train_cv_safe_config`, not a weak selection.

All controls and candidates produce 191 unanswerable false answers out of 192
unanswerable rows. Stage 161 only enforces non-regression on this metric; it
does not claim that the underlying answer behavior is acceptable.

## Performance and visualizations

Measured phase time is:

```text
retrieval-channel construction: 49.514033s
Top200 record construction: 47.994819s
two control evaluations: 5.908825s
six five-fold model evaluations: 25.844280s
selection and no-op refit decision: 0.039095s
total: 131.874477s
```

Average selector-only latency is `0.770539-0.793777ms` for pairwise logistic
and `1.596058-1.815369ms` for histogram GBDT. This excludes candidate-pool
construction and answer generation, so it must not be presented as complete
request latency.

Ten public-safe SVGs cover hit rate, answer F1, citations, answerable refusals,
unanswerable false answers, tail promotions, selector latency, minimum fold hit
delta, config selection status, and process guards. All ten parse as XML. No
pixel-level screenshot review is claimed.

## Process history

The implementation history is retained rather than rewritten as a clean first
attempt:

- Two initial read-only searches guessed nonexistent historical module names;
  one `rg` exited 1. Neither changed files.
- The first core Ruff pass formatted two files, then rejected an unused
  `statistics.median` import. The unused import and unused private accumulation
  were removed before the next pass.
- The CLI patch completed while tool output was truncated. File existence and
  size were checked before any success claim.
- CLI formatting first failed `ruff format --check`; after formatting, Ruff
  found `Mapping` imported from `typing`. Moving it to `collections.abc`
  resolved the issue.
- The first focused test run was `12 passed, 3 failed`. It exposed two real
  bugs: missing zero-valued frozen features when a route is absent, and a
  missing chart `x_label`. The third failure was an invalid test assumption
  that histogram trees cannot tie scores. The implementation and test were
  corrected.
- The first correction introduced a visualization-loop indentation error.
  Ruff rejected the file before tests ran. The indentation was repaired, and
  the next focused run was `15 passed`.
- The first source-authorization preflight used a manually guessed wrong corpus
  path and failed with `FileNotFoundError`. Reading `ProjectSettings` supplied
  the real configured path; the corrected authorization matched all six
  expected fingerprints.
- The formal launcher did not print its intended JSON summary. No second formal
  was started. The written PID, wrapper, child Python command line, progress
  log, and exit file were inspected read-only and followed through the same
  process to natural completion.

Formal stderr is nonempty. It contains successful local model weight-loading
progress plus PowerShell CLIXML first-use progress; it has no Python traceback,
and the formal exit code is 0.

Current-source verification after documentation is:

```text
Stage161 five-file Ruff format check: passed
full repository Ruff lint: passed
Stage161 targeted pytest: 15 passed in 2.05s
full repository pytest: 830 passed, 1 existing warning in 13.66s
full pytest exit code: 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. Full
pytest stderr contains 382 bytes of PowerShell first-use CLIXML progress and no
test traceback; it is intentionally not described as empty. The single full
pytest process had no test timeout or monitoring deadline and exited naturally.

## Decision and next direction

The strict Stage 161 decision is:

```text
process guards: 18 / 18
completed configurations: 6 / 6
selectable configurations: 0 / 6
selected configuration: none
development used: false
test used: false
model artifact written: false
runtime changed: false
fallback enabled: false
```

The experiment reveals that the current query-overlap shortlister is the wrong
control to optimize in isolation: untouched RRF already recovers 80 additional
train gold documents in Top10 and has higher verified F1. A follow-up should
first freeze whether the product baseline should return to untouched RRF, then
evaluate a conservative learned swap policy that preserves eight or nine RRF
positions and promotes at most one or two candidates. Any threshold or swap
budget must be selected inside nested train-only CV before a one-shot
development evaluation. Test and runtime defaultization remain closed.
