# Stage 163 Untouched-RRF Development Validation

## Objective

Stage 163 performs one independent development comparison between two policies
that were frozen before development data was loaded:

- current query-overlap Top10;
- untouched original-RRF Top10.

The experiment does not train a model, search a threshold, choose among
multiple candidates, run Qwen Agent, open test, register a runtime default, or
enable fallback, query rewrite, or second retrieval.

## Frozen Protocol

The authorized inputs are:

```text
Stage162 report:
  ff126db5efc2b117ab77cf99a62ec5c399110a938b3a37ea449055e76e622d93
Stage160 report:
  e17e5fe5bbc5fef4e25e41234e47b89daf19ea4ef18f3c7270601f0fee7d9377
Stage80 report:
  2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f
Stage68 development split:
  071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f
Technote corpus:
  f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b
```

The development split contains 121 rows: 76 answerable and 45 unanswerable.
Normalized-question plus answer-document grouping produces 117 groups and
five folds with row counts `25/24/24/24/24`. Folds are used only for strict
stability reporting; they do not fit or select anything.

The Stage116 offline retrieval contract builds exactly 200 RRF candidates per
row, or 24,200 in-memory records. Both fixed Top10 policies are evaluated with
the existing deterministic answer generator and verifier. The candidate may be
adopted only if it strictly improves context hit, does not regress aggregate
F1, citations, refusals, or unanswerable false answers, and does not regress
hit or F1 in any fold.

## One-Shot Development Result

The formal process was launched once with the local CUDA encoder and observed
until it ended naturally. It produced the report and ten SVGs after
`74.246744s`. The detached launcher did not retain a reusable process exit
code, so no exit-code claim is made. Completion is evidenced by the final
saved-report message, the parseable JSON report, ten parseable SVGs, and the
absence of a traceback. Formal stderr contains successful Transformers weight
loading progress and is not empty.

The exact results are:

```text
current query-overlap Top10:
  context gold hit:             36 / 76 = 47.3684%
  all-answerable F1:            0.187282
  completed-answerable F1:      0.187282
  gold citations:               33
  answerable refusals:           0
  unanswerable false answers:   45 / 45

untouched original-RRF Top10:
  context gold hit:             55 / 76 = 72.3684%
  all-answerable F1:            0.186694
  completed-answerable F1:      0.186694
  gold citations:               42
  answerable refusals:           0
  unanswerable false answers:   45 / 45
```

RRF therefore gains 19 Top10 gold hits and nine gold citations, while aggregate
F1 decreases by `0.000588`. Answerable case-level F1 improves for 37 rows,
regresses for 20, and ties for 19.

Fold deltas versus current are:

```text
fold_1: hit +0.052632, F1 +0.012525
fold_2: hit +0.437500, F1 +0.006367
fold_3: hit +0.400000, F1 +0.006969
fold_4: hit +0.133333, F1 -0.013365
fold_5: hit +0.312500, F1 -0.015860
```

Six of eight strict policy guards pass. Aggregate F1 and every-fold F1 fail.
Untouched RRF is therefore not development-safe and is not integrated into the
Agent or registered as a default. The unanswerable false-answer result remains
`45/45` for both deterministic policies and is not acceptable runtime behavior.

The immutable original formal report SHA-256 is:

```text
66f3b3185d4a7a3447fc9524a78729bc1e307f5feab04b6308268cdb06642e05
```

## Contract Correction

The original formal report passed 16 of 17 process guards and was correctly
marked invalid. Its failed `candidate_pool_exact` guard expected 70 answerable
gold hits in the Stage163 Top200 pool. That value came from Stage160 runtime,
whose actual per-row contract is:

```text
candidate pool:        400
verification context: 200
generation context:    10
rows:                  121
```

Stage163 instead evaluates the Stage116 offline Top200 pool and observes 69
gold hits. The two gold-hit counts belong to different pool depths and cannot
serve as a reproducibility equality check. Current Top10 generation hit still
reproduces the Stage160 value of 36.

The user selected correction option A. The original report was preserved
byte-for-byte. A separate correction audit reads only the original Stage163
public report, the Stage160 public report, and Stage160's hashed diagnostic
artifact. It does not load development, rebuild retrieval, reevaluate a case,
or change any metric. The replacement process guard checks only the frozen
Top200 shape: 121 pools, 24,200 records, and depth exactly 200.

Correction evidence is:

```text
original Stage163 report unchanged: true
metric snapshot before/after SHA-256:
  ab176cacba2a844b998e49744626a9555a0c30dda61e0935cdb999f410e4a147
correction guards:             12 / 12
corrected Stage163 process:    17 / 17
strict policy guards:           6 / 8
development rows reloaded:      0
development cases reevaluated:  0
```

The correction report SHA-256 is:

```text
f31efd39fc87f3c9289d2cc2521d0928e283a2535418565cf6d1d668565da15b
```

It produces two additional XML-parseable SVGs. Together with the original ten,
Stage163 has twelve structural visualizations. Pixel-level screenshot review
is not claimed.

## Implementation and Verification Notes

The shared candidate builder now accepts explicit progress-stage and
progress-phase ownership. The Stage161 compatibility alias remains intact;
Stage163 emits `Stage 163 / dev_candidate_pool_build` events.

Observed pre-final failures are preserved rather than overwritten:

- one Windows wildcard `rg` command failed before being replaced with a
  directory plus `-g` search;
- one historical script filename was guessed incorrectly during read-only
  inspection;
- the first source-authorization preflight manually guessed the wrong corpus
  path and stopped before development evaluation; `ProjectSettings` provided
  the real path;
- initial Ruff format checks found mechanical formatting changes;
- the correction module's first Ruff check found import-source and ordering
  issues, which were corrected;
- the original formal process exposed the Top400-versus-Top200 process-guard
  error; it was not silently rewritten or rerun.

Final current-source verification is full-repository Ruff passing, combined
Stage162/163 targeted `19 passed in 1.40s`, and replacement full-repository
`854 passed, 1 existing warning in 12.14s`. The replacement full pytest exits
0 with empty stderr. The replacement was explicitly authorized after an
ignored driver path error interrupted the first full collection attempt.

## Decision

The corrected process audit is valid, but untouched RRF fails the strict
development quality gate. Test remains locked. Runtime integration, default
registration, fallback, query rewrite, and second retrieval remain closed.

The next eligible work is to stop this context-policy family and analyze the
existing Stage160 gold-visible Agent refusals. That work must address the
generation decision itself rather than treating higher Top10 recall as
sufficient evidence of answer quality.
