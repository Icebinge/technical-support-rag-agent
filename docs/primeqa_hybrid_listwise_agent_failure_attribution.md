# Stage 179 Listwise Agent Failure Attribution

## Objective

Stage 179 explains why Stage 178A added 82 gold-containing contexts and 32
gold citations but did not produce statistically stable F1. It replays the
same deterministic paired tool-Agent with authenticated five-fold OOF scores
and decomposes the path into prefix availability, two-view union availability,
selected Top10 context, gold citation, and final answer F1.

This is train-only diagnosis. It performs no model fit or policy selection,
does not load the full checkpoint, keeps development and test closed, adds no
fallback, and does not change the default runtime.

## Frozen Attribution Rules

The analysis reports all 370 answerable train rows by the five frozen OOF
folds. It measures context and citation transition matrices, context-gain and
context-loss F1 outcomes, citation conversion, candidate gold-context rank,
and per-fold F1 deltas.

The primary bottleneck rules are fixed before formal replay:

```text
context-to-citation conversion:
  uncited context gains >= cited context gains

cited-evidence answer fidelity:
  at least one candidate gold-cited answer has lower F1

reranker context stability:
  at least one baseline gold context becomes a candidate miss
```

The first matching rule in that order identifies the recommended next
research direction. The result cannot authorize Stage 178B or a runtime
change.

## Formal Replay

The formal train-only replay used PID `14136`. The launching PowerShell
command called `Wait-Process` once for that PID and waited for natural
completion. The outer command completed successfully, but the Windows
`Start-Process` object's child `ExitCode` field was blank; this record does
not rewrite that blank value as zero. The complete report and all 16 process
guards were produced.

The authenticated inputs were:

```text
Stage 178 public report SHA-256:
e57e3f09bcc65657a3f8783e97e6767b690095e2cffd5d252d51e181eaf533c9

Stage 178 private report SHA-256:
6fffa820773dea8892dc1d441aff1c3ef3df54ff368b82bf1c9a09b961f0857a

corrected alignment report SHA-256:
e2398024edf128ad0628900d25eb1ccc9c83c437fb474921fe136e2603e47272
```

The replay used exactly 562 train questions and 9,714 authenticated OOF pair
scores. It built the live runtime once, completed 1,124 Agent turns, and made
562 score-provider calls. Both baseline and candidate completed all 562
invocations with no retry, fallback, failure, checkpoint load, model fit,
development access, test access, or default-runtime change. Stage 178 metrics
were reproduced exactly.

```text
source authentication:       0.024186 s
train/fold loading:          0.012835 s
runtime construction:       39.911538 s
paired Agent replay:         98.609738 s
attribution analysis:        0.114434 s
wall time:                  138.672731 s
```

The public Stage 179 report is
`artifacts/primeqa_hybrid_listwise_agent_failure_attribution_stage179.json`
with SHA-256
`80a7b82016eb54a480748466fabff7990d147843742e49114277e08155b45d8f`.

## Attribution Results

The answerable-train evidence path was:

```text
prefix gold hit:                 345
two-view union gold hit:         267
candidate Top10 context hit:     257
candidate gold citation:         183
candidate positive F1:           360
```

Thus 78 questions lost the gold document at the frozen union boundary, 10
more lost it during listwise Top10 selection, and 74 candidate contexts that
contained gold evidence did not produce a gold citation.

Context transitions were 171 hit-to-hit, 4 hit-to-miss, 86 miss-to-hit, and
109 miss-to-miss. The 86 context gains produced 47 improved, 6 tied, and 33
worsened F1 outcomes. Only 37 of those 86 gains converted to a gold citation,
for a conversion rate of `0.430233`; 49 remained uncited. Their mean F1 delta
was `+0.039837` and the summed delta was `+3.425977`.

Citation transitions were 143 hit-to-hit, 8 hit-to-miss, 40 miss-to-hit, and
179 miss-to-miss. Among all 183 candidate gold-cited answers, 60 improved, 63
tied, and 60 worsened, with mean F1 delta `+0.009791`. This confirms that
answer composition remains unstable even when the selected evidence is
cited.

Gold citation weakened sharply below rank 1:

```text
rank       cases   cited   citation rate   mean F1
1          175     152     0.868571        0.240844
2-3         53      22     0.415094        0.194706
4-5         12       5     0.416667        0.134394
6-10        17       4     0.235294        0.171287
```

The paired answerable outcome was 139 improved, 97 tied, and 134 worsened.
Fold F1 deltas were `-0.008159`, `-0.006910`, `+0.017779`, `+0.008320`, and
`+0.019563`. This explains why the aggregate `+0.005804` improvement from
Stage 178 was not statistically stable.

## Decision

All three frozen diagnostic flags are true: context-to-citation conversion,
cited-evidence answer fidelity, and reranker context stability. Fixed rule
priority selects `context_to_citation_conversion` as the primary bottleneck.
The next recommended research direction is
`design_runtime_visible_citation_aware_composition_oof`.

This result authorizes design work only. It does not authorize Stage 178B,
gold-dependent runtime features, a fallback, a default-runtime change, or any
development/test evaluation.

## Visual Verification And Corrections

Six SVG files were generated and XML-parsed. Four key charts were rendered
to PNG and visually inspected: pipeline waterfall, context-gain outcomes,
fold F1 deltas, and citation rate by context rank. They are nonblank, labels
fit, and values agree with the JSON report.

The first parallel image read occurred before two Edge screenshot writes had
fully settled, so their displayed titles appeared truncated. The same SVGs
were rendered again and then inspected sequentially; both titles were
complete. Edge also emitted an existing QQBrowser-profile lookup warning,
which did not affect the generated files.

Before the formal replay, implementation review found a redundant direct
score-provider call after the candidate workflow had already scored the same
query. It would have doubled diagnostic counters without changing answers.
The call was removed before the formal run, and the formal report records the
correct 562 provider calls.

## Current-Source Verification

```text
Stage 177-179 focused regression: 18 passed in 2.39s
three-file Ruff format check: passed
full repository Ruff lint: passed
full repository pytest: 1048 passed, 1 warning in 18.31s
full pytest PID / outer command exit: 32640 / 0
```

The full pytest command started one process and called `Wait-Process` once on
that PID until natural completion. `HasExited` was true, stderr was empty,
and the child `ExitCode` property printed blank. The warning is the existing
FastAPI/Starlette `TestClient` deprecation.
