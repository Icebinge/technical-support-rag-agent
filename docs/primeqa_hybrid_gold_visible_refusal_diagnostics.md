# Stage 164 Gold-Visible Agent Refusal Diagnostics

## Objective

Stage 164 analyzes the existing Stage160 development outcomes in which the
gold document was already present in the Qwen Agent generation context. It
asks why the Agent refused 19 of those 36 answerable cases while answering 17.

This is a read-only diagnostic join, not another development evaluation. The
stage does not run the Agent, rebuild retrieval, fit a model, tune a threshold,
select a policy, load test, register a runtime default, or enable fallback,
query rewrite, or second retrieval.

## Frozen Evidence And Prompt Contract

The diagnostic authorizes the immutable Stage163 correction, Stage160 public
and hashed-private evidence, the frozen development split, the technote
corpus, and the router prompt source. Principal public source fingerprints
are:

```text
Stage163 correction:
  f31efd39fc87f3c9289d2cc2521d0928e283a2535418565cf6d1d668565da15b
Stage160 public report:
  e17e5fe5bbc5fef4e25e41234e47b89daf19ea4ef18f3c7270601f0fee7d9377
development split:
  071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f
technote corpus:
  f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b
```

The executable source authorization contains and verifies all additional
private-artifact and source-code fingerprints in full.

`StructuredRouterPromptBuilder` includes at most ten evidence results. For
each result it exposes the title and only the first 600 document characters.
Therefore, a gold document appearing in the generation context does not prove
that the answer-bearing passage appears in the model prompt.

The 36 rows form 35 normalized-question plus answer-document diagnostic
groups and five fixed folds. The folds are descriptive stability checks only;
they do not fit or select anything. The public report contains no case rows.
The ignored private artifact contains 36 hashed feature rows but no raw
question, answer, document identifier, or document text.

## Observed Results

The fixed cohort is:

```text
gold document visible: 36
Agent refused:         19
Agent answered:        17
refusal rate:          52.7778%
diagnostic groups:     35
folds:                  5
```

Answer visibility under the actual 600-character prompt excerpt is:

```text
exact answer span visible:  11 / 36
all answer tokens visible:  11 / 36
partial answer tokens:      24 / 36
no answer tokens:            1 / 36
answer in full document:    36 / 36
gold document truncated:    33 / 36
```

Cases without an exact visible answer span have refusal rate `14/25 = 56%`;
cases with the exact span have refusal rate `5/11 = 45.4545%`. The aggregate
difference is `+10.5455` percentage points and the Haldane-corrected odds ratio
is `1.490119`. However, the five fold differences are:

```text
-0.500000, +0.500000, -0.066667, -0.500000, +0.750000
```

Only two folds point in the aggregate risk direction and three point in the
opposite direction. Prompt answer visibility is therefore a real data issue
but not a fold-stable explanation for refusal in this small development
cohort. It does not authorize a prompt intervention.

The strongest observed numeric association is lower question-token recall in
the gold prompt excerpt:

```text
refused median:   0.461538
answered median:  0.692308
risk-aligned AUC: 0.730650
```

This is aggregate diagnostic evidence, not a cross-validated model result or
a causal estimate. Answer-token visibility is weaker at AUC `0.609907`.
Preserved gold candidate rank is also weak at AUC `0.530960`.

Post-first-turn cases have refusal rate `15/25 = 60%`, compared with `4/11 =
36.3636%` on first turns, a difference of `+23.6364` percentage points. All
five comparable folds point in that direction, with differences
`0.10/0.50/0.25/0.166667/0.50`. The Stage160 histories are synthetic groupings
of unrelated questions rather than natural conversations, so this pattern
cannot be interpreted causally. It motivates train-only history-isolation and
question-evidence-alignment diagnostics, not a runtime history rule.

Route cells are small and exploratory. Security-bulletin cases are `0/6`
refused, while limitation or restriction cases are `2/2` refused; no route
policy is selected from these counts.

## Formal Failure And Contract Correction

The one formal aggregate diagnostic completed its intended read-only analysis
and wrote the original public/private artifacts and ten SVGs. It then exited
with code 1 because only 15 of 16 process guards passed. The failed
`gold_generation_ranks_bounded` guard incorrectly treated
`RetrievalResult.rank` as a dense generation-list position in `1..10`.

The actual value preserves candidate-pool rank after results enter the
generation context. One gold result therefore legitimately has rank 14 while
all 36 generation contexts still contain exactly ten results. The invalid
guard was not hidden, and the original report remains unchanged:

```text
original public report SHA-256:
  2a7dcef4fbc007f53d141cd246e7ad4bf327c3f5ad75899424f0a69c273ed3ae
original private artifact byte SHA-256:
  ddddc77f3e5bdfe1a756aa680dae2102e340a6b7c5452b90a53271ccc5f98507
original private canonical SHA-256:
  20d42bae9b954e223dbc55798fa8de71fe5beec42dbed835f5f96b2e7aba3a63
```

After the user selected option A, a separate correction audit read only the
immutable Stage164 public/private artifacts and Stage160 hashed evidence. It
loaded zero development rows and zero documents, recomputed zero feature rows,
and ran the Agent and retrieval zero times. The corrected guard checks exact
gold membership plus a ten-result context rather than bounding preserved rank.

The correction also separates an aggregate visibility gap from fold-stable
evidence. The metric snapshot is byte-stable at:

```text
adbdd33664fbc66c42a341caf817ebf98cfba53014291eb33114b95dc6a1288f
```

Correction results are:

```text
correction guards:              11 / 11
corrected Stage164 process:     16 / 16
metric snapshot changed:        false
development rows reloaded:          0
documents reloaded:                 0
feature rows recomputed:             0
Agent runs:                          0
retrieval runs:                      0
fold-stable visibility gap:      false
policy selected:                 false
```

The correction report SHA-256 is:

```text
d80b786c32462cb9032e657ee1d1abc67f5cd995da66c1abd3831b3067c299fa
```

The original ten and correction two SVGs are all XML-parseable. Pixel-level
screenshot review is not claimed.

## Implementation And Verification Notes

The diagnostic is implemented as a dedicated application analyzer, an
explicit CLI, and a separate contract-correction auditor. Fixed feature
families cover answer visibility, gold rank/score, question-to-prompt lexical
alignment, route, and history load. Public-safety checks prevent raw case data
from entering the report.

Observed pre-final failures are preserved:

- one read-only search used an invalid wildcard and a guessed router path;
- initial Ruff checks found two long lines and later mechanical formatting;
- the first source-authorization preflight shadowed an imported module with a
  local `stage160` variable and failed before loading development;
- correction-module Ruff checks found import-source, ordering, and long-line
  issues;
- the formal diagnostic exposed the incorrect dense-rank guard and exited 1;
  it was not rerun after the user chose the no-data-rerun correction route;
- the first combined documentation patch used incorrectly decoded journal
  context and was rejected atomically without writing any file.

Current-source targeted verification is `16 passed in 0.85s`, with all five
Stage164 source files passing Ruff format and lint. Full-repository Ruff also
passes. The single full-repository pytest process exits 0 with empty stderr and
reports `870 passed, 1 existing warning in 12.60s`. It has no pytest timeout,
monitor deadline, kill, restart, retry, or fallback. The warning is the existing
FastAPI/Starlette `TestClient` deprecation.

## Decision

Stage164 is a valid diagnostic after the separate contract correction, but it
does not select a policy. Test remains locked. Agent/retrieval reruns, runtime
defaultization, fallback, query rewrite, and second retrieval remain closed.

The next eligible direction is a train-only diagnostic protocol for synthetic
history contamination/isolation and question-to-evidence alignment. Any later
Agent experiment requires a separately frozen protocol and explicit approval.
