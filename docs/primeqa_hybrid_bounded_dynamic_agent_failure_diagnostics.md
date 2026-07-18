# Stage 160: Bounded Dynamic Agent Failure Diagnostics

## Status

Stage 160 is a development-only diagnostic analysis of the exact Stage 159
warm multi-turn Agent service. It completed one formal 121-turn GPU and HTTP
run with 57/57 guards passing. It does not select a policy, tune a threshold,
change the production runtime, or open the test gate.

The formal public artifact SHA-256 is
`e17e5fe5bbc5fef4e25e41234e47b89daf19ea4ef18f3c7270601f0fee7d9377`.
The ignored private artifact byte SHA-256 is
`3f10cffe245a4405dfc56044f2a3c0d364fdd0f8723e6cc3ae401260199652db`;
its canonical JSON content SHA-256 is
`1c8aa4260be5427e13322cb3304e518dd3609c2e38f839cda4f10ce01c911a0d`.

## Frozen protocol

The only evaluation source is the exact Stage 68 development file:

```text
artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_dev.jsonl
SHA-256: 071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f
rows: 121
answerable / unanswerable: 76 / 45
```

The workload reuses the exact Stage 159 stable order and synthetic thread
grouping. The order SHA-256 is
`3b8a39cae397db4402080a2780178ade0fd4fc3a9ba5facb25d041510e8b69b7`;
the 30 four-turn plus one one-turn grouping SHA-256 is
`7aa271a775c2926b32226e0a4fccc96cff3a7bf98fc90246c8002d79561fd6d0`.
These generated groups are a service workload, not natural conversations.

Gold labels are loaded only by a validation observer after each real runtime
turn. Runtime requests receive only the question title and text. No gold field
is projected into retrieval, routing, generation, composition, or
verification. Train and test are not loaded.

Five-fold analysis is grouped by normalized question plus answer document, or
the unanswerable marker. It has 117 groups, fold row counts
`25/24/24/24/24`, and fold group counts `24/23/23/23/24`. The assignment
SHA-256 is
`79c5b8d805d00b36bb653bc1243b91985caf923b8c68ee39cfe54d48cd51f739`.
This is diagnostic stability analysis only: no model is fit, no policy is
selected, and no threshold is tuned.

## Implementation

Stage 160 adds a frozen protocol module, a validation-only runtime observer, a
CLI, focused tests, and ten SVG views. The observer decorates the exact Stage
158 runtime and records the private post-turn state needed to locate loss
between the candidate pool, generation Top10, model decision, and composition.
The production runtime and HTTP contract are unchanged.

The public artifact contains only aggregate metrics. The ignored private
artifact contains 121 hashed rows with numerical and categorical diagnostics.
It has no raw question, answer, document identifier, document text, citation
identifier, or model output. All 121 private sample and query identities are
valid SHA-256 values; 117 grouped query identities and 71 answer-document
identities are observed, matching the frozen input properties.

## Formal service result

The formal process started once on loopback port `18160` and ended naturally
with exit code 0. It completed all 31 threads and 121 development turns, then
released the port and left no Python process, listener, or server thread open.

```text
guards: 57 / 57
dev HTTP 200: 121 / 121
thread open / close: 31 / 31
selected compose / refuse: 34 / 87
resource factory builds: 1
model generations: 122 (1 warmup + 121 dev)
peak allocated GPU bytes: 7,344,342,016
queue / retry / fallback actions: 0 / 0 / 0
```

The development-only aggregate quality diagnostics are:

```text
answerable refusal rate: 68.4211% (52 / 76)
unanswerable refusal rate: 77.7778% (35 / 45)
unanswerable false-answer rate: 22.2222% (10 / 45)
answerable gold candidate-pool hit rate: 92.1053% (70 / 76)
answerable gold generation-Top10 hit rate: 47.3684% (36 / 76)
answerable gold verification-context hit rate: 90.7895% (69 / 76)
answerable gold citation rate: 19.7368% (15 / 76)
answerable token F1 over all answerable rows: 0.088213
answerable token F1 over completed answerable rows: 0.279342
```

The verification-context rate is an independent observation of the runtime's
verification inputs; it is not treated as proof that the generation prompt
contained the same document. The generation-Top10 metric is the relevant
boundary for the model decision studied here.

## Failure localization

The 52 answerable refusals split into mutually exclusive mechanisms:

```text
candidate-pool miss: 5
gold lost before generation Top10: 28
gold visible in generation Top10 but model refused: 19
post-compose refusal: 0
```

The dominant observed mechanism is therefore generation Top10 loss, not the
first-stage candidate pool. The candidate pool finds the gold document for
92.1% of answerable questions, but the generation Top10 retains it for only
47.4%. Enlarging or rebuilding the first-stage pool is not the primary next
experiment. A train/CV-only second-stage ranking or context-selection
intervention should be designed first, while separately studying the 19
gold-visible model refusals.

The grouped folds support the same conclusion. Answerable generation-Top10 hit
rate ranges only from `0.4375` to `0.5000` across the five folds, with
population standard deviation `0.023206`. Answerable refusal rate ranges from
`0.600000` to `0.764706`. No fold result is used for selection.

## Latency localization

```text
end-to-end median / p95 / max: 1582.611 / 9321.585 / 16350.131 ms
end-to-end average: 3460.434 ms
generation median / p95 / max: 1389.059 / 9038.853 / 16194.215 ms
generation average: 3219.194 ms
non-generation average: 241.240 ms
average generation share: 93.0286%
Stage159 p95 exceedances: 2 / 121
```

Spearman correlation with generation latency is `0.915809` for router input
tokens, `0.627997` for retained state bytes, `0.566652` for turn position,
`0.034743` for output tokens, and `0.0` for the fixed candidate-pool count.
This identifies input length as the strongest observed correlate; it does not
establish causality.

Average generation latency rises by synthetic turn position from
`1264.680ms` to `2007.382ms`, `3788.273ms`, and `5881.591ms`. The two requests
above the Stage 159 p95 reference both occur at turn position four. This
supports a future context-budget experiment, but the generated thread grouping
prevents claims about natural conversational behavior.

## Visual and privacy verification

Ten SVG files were generated and all ten parse as XML. Their explicit canvases
range from `1480x188` to `1480x2278`; titles, text nodes, bars, and view boxes
are present. The views cover action distribution, action latency, refusal flow,
failure buckets, grouped-fold refusal rates, all guards, latency correlations,
quality rates, and per-position end-to-end and generation latency.

The local image viewer could not process SVG input. A subsequent attempt to
render the local file in the in-app browser was blocked by browser URL policy,
and no policy workaround was attempted. Consequently Stage 160 proves SVG XML
and structural validity, but does not claim pixel-level screenshot review.

## Process history

Several non-formal checks failed and are retained as process facts:

- An initial parallel read-only audit returned exit 1 because one `rg` search
  had no matches. A later Windows wildcard `rg` emitted an invalid-filename
  error. Neither command changed files.
- The first query-uniqueness probe used an unavailable static .NET
  `SHA256.HashData` API and printed an invalid zero count alongside errors. A
  corrected `SHA256.Create` probe established 117 query digests, four duplicate
  groups, eight duplicate rows, 76 answerable rows, 45 unanswerable rows, and
  71 answer documents.
- The first Ruff pass formatted three files and reported five issues. After
  correction, one import-order issue remained and was fixed by Ruff. The next
  repository Ruff run passed.
- The first focused pytest run was `14 passed, 4 failed`. Three failures came
  from test fixtures hashing in-memory LF content while Windows wrote CRLF;
  one used an incorrect manually copied query digest. Tests were corrected to
  hash actual bytes and call the protocol digest function. The corrected run
  was `18 passed`; the Stage 157-160 set was `70 passed, 1 existing warning`.
- A post-document verification command mistakenly applied
  `ruff format --check` to the entire historical repository. It stopped before
  lint or pytest and reported 311 old files that the current formatter would
  change; it modified none. The intended Stage160 five-file format check and
  full-repository Ruff lint both pass.
- The next targeted pytest command named two nonexistent historical test
  paths. Ruff passed, but pytest collected zero items and exited 1 before any
  test ran. Reading the real repository file list and running the nine exact
  Stage157-160 files produced `70 passed, 1 existing warning in 2.00s`.
- The formal launcher returned without printing a PID. No success assumption
  or second launch was made; read-only process, progress, port, and exit-file
  checks confirmed the single wrapper/child run and followed it to natural
  completion.
- An initial report dump exceeded the tool output budget, so subsequent reads
  selected bounded aggregate fields. A private hash check first queried the
  wrong `query_sha256` field and printed a false zero; the corrected
  `query_digest_sha256` check validated all 121 rows.

The formal stderr is nonempty: 490 bytes of successful model weight-loading
progress. It contains no traceback, and the formal exit code is 0. The formal
workload took `419.539637s`; total runtime including authorization, resource
construction, model loading, warmup, shutdown, and audit was `484.320761s`.

Current-source verification after documentation is:

```text
Stage160 five-file Ruff format check: passed
full repository Ruff lint: passed
Stage157-160 targeted pytest: 70 passed, 1 existing warning in 2.00s
full repository pytest: 815 passed, 1 existing warning in 13.84s
full pytest exit code / stderr: 0 / empty
```

The warning is the existing Starlette deprecation warning emitted by FastAPI's
`TestClient` import. The full-test driver ran without a runtime deadline and
was observed read-only until its natural exit.

## Closed boundaries and next direction

Test evaluation, model fitting, policy selection, threshold tuning, default
runtime registration, remote exposure, persistent state, query rewrite,
second retrieval, queueing, retry, and fallback remain closed. Stage 160 does
not authorize any of them.

The next research stage should design a train plus grouped-CV experiment for a
fast second-stage reranker or generation-context selector targeted at the
measured Top10 loss, followed by a separate intervention for gold-visible model
refusals. Development evaluation remains independent of training; test stays
locked until a complete policy is frozen.
