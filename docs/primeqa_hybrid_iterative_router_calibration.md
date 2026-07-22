# Stage 169 Real GPU Iterative Router Calibration

## Objective

Stage 169 calibrates the Stage 168 A+C decision protocol with the real cached
`Qwen/Qwen3-VL-2B-Instruct` model on the RTX 5060. It measures strict schema
reliability, synthetic action following, train-only gold-visibility routing
proxies, latency, process memory, system memory, and CUDA memory.

The stage does not run answer generation, F1, citation scoring, development,
or test. It does not add retries, query rewriting, second retrieval, HTTP
integration, or default runtime activation.

## Frozen Calibration

Before the first model run, the protocol fixed 14 synthetic initial cases and
four synthetic final-phase cases. These cover direct compose, direct refuse,
all six clarification kinds, inspect then compose, and inspect then refuse.

The real train-only calibration deterministically selects ten rows from each
of five strata by ascending SHA-256 of the frozen sample identity:

```text
initial gold visible
alternate-only gold visible
union gold missing but candidate Top200 hit
candidate Top200 gold missing
unanswerable
```

Every selected train row receives one independent initial decision and one
counterfactual final-after-inspection decision. Together with the synthetic
matrix, the model performs exactly 118 greedy generations. The thresholds were
fixed before model execution and were not changed after observing results.

## Compact Prompt Redesign

The first formal run used the Stage 168 `600 chars x 20 documents` final prompt
and failed with CUDA OOM. A tokenizer-only estimate over real technote text
measured the following final prompt sizes:

```text
600 chars/document: 5252 tokens
300 chars/document: 3402 tokens
240 chars/document: 3075 tokens
200 chars/document: 2886 tokens
```

The user approved a compact redesign. The router now selects one query-aware
200-character window per document by locating at most 16 longest query tokens,
checking at most four occurrences per token, and scoring only that bounded
window set. It removes alternate documents already present in the initial view
and enforces 4,096 input tokens and 32 output tokens. This is deterministic
prompt construction, not an OOM retry or fallback.

The successful run observed 622 to 2,601 input tokens, with p50 1,551 and p95
2,502. All 118 model calls completed without OOM.

## Quality Results

The model obeyed the JSON schema on 116/118 calls, or 98.3051%. Both invalid
outputs were synthetic final compose cases. Synthetic action quality was well
below the frozen threshold:

```text
phase action accuracy:         7 / 18 = 38.8889%
clarification kind accuracy:   0 / 6  = 0%
exact synthetic path accuracy: 4 / 14 = 28.5714%
```

All six synthetic clarification cases were routed to inspect instead of
clarify. On the real train-only evidence, the initial action also had a strong
inspect bias:

```text
initial-gold-visible compose:       0 / 10 = 0%
alternate-only initial inspect:     9 / 10 = 90%
alternate-only final compose:       5 / 10 = 50%
alternate-only inspect->compose:    4 / 10 = 40%
insufficient-strata final compose:  3 / 30 = 10%
```

Only three of eight frozen quality gates passed: alternate-only inspect,
alternate-only complete-path success, and insufficient-evidence final compose
safety. The real model tends to inspect in the initial phase and clarify in the
final phase, even when the gold document is already visible. This makes the
current prompt/router unsuitable for runtime integration.

The formal decision is:

```text
stage169_router_requires_redesign
```

Development and test remain closed, and the runtime default remains unchanged.

## Resource Consumption

Resource observations are captured inside the process at phase boundaries and
after each generation. There is no external status polling or periodic monitor.

```text
total wall time:                    217.407039 s
source fingerprinting:               3.272942 s
train evidence build:               92.257149 s
model load:                           4.970529 s
synthetic calibration:                9.463150 s
train calibration:                  107.443268 s
total GPU generation time:          115.038575 s
generation throughput:                1.025743 calls/s
total input/output tokens:       194753 / 2148
generation latency p50/p95/max: 905.428 / 1603.777 / 1733.520 ms
```

Memory peaks were:

```text
process peak working set:             7.518 GiB
process peak private usage:           12.725 GiB
minimum system available memory:      3.370 GiB
CUDA peak allocated:                  5.323 GiB
CUDA peak reserved:                   6.496 GiB
```

Private usage is Windows committed virtual memory and can exceed the physical
working set. CUDA reserved memory includes PyTorch's allocator cache; CUDA
allocated memory is the closer measure of live tensor demand.

## Process Corrections

The first formal PID `56040` completed all 14 synthetic cases and nine train
cases before exiting with confirmed `CUDA error: out of memory`. It created no
report and performed no automatic retry. A single read-only diagnostic later
confirmed the process had ended with exit code 1.

The first compact replacement completed all 118 model calls without OOM, then
failed during report assembly. PowerShell had written the old stderr log in the
Windows `cp936` code page, while the audit reader assumed UTF-8. No report was
created. Byte inspection established the encoding; the reader was changed to
the Windows preferred encoding and a regression test was added.

An earlier standalone resource-sampler check failed because `ctypes` had not
declared the Win64 HANDLE and API argument types. The model was not loaded in
that check. Explicit `GetProcessMemoryInfo` and `GlobalMemoryStatusEx` signatures
were added and independently verified before the current-source run.

The current-source run started from model call one, completed all 118 calls,
and exited naturally. All formal processes were each started by one PowerShell
command that called `Wait-Process` once for the same PID. There was no polling,
PowerShell wait timeout, process kill, or partial-result continuation.

## Artifacts

The public report is
`artifacts/primeqa_hybrid_iterative_router_calibration_stage169.json`, SHA-256:

```text
aa1f66d64ecf901d811c8f4db436b88f3fd416f91f0d9078c8d37f2174b06ad1
```

Five SVGs cover synthetic quality, train proxy quality, initial inspect counts,
latency, and resource peaks. All parse as XML. The train quality and resource
charts were rendered with Edge headless and visually checked. A first shared
Edge profile screenshot showed missing title glyphs on the resource chart; an
independent profile rerender displayed the complete title and all labels.

## Next Boundary

Stage 170 should redesign and compare router prompts on the same frozen
synthetic and train calibration identities. It should address initial inspect
bias, final clarify bias, clarification-kind selection, and strict final JSON
reliability. Runtime E2E, development, test, and default activation stay closed
until the router passes the frozen gates.

## Repository Verification

```text
Stage 169 focused pytest: 30 passed in 2.14s
five-file Ruff format check: passed
full repository Ruff lint: passed
full repository pytest: 963 passed, 1 warning in 13.57s
full pytest exit code / stderr bytes: 0 / 0
```

The full suite ran once under one PowerShell-launched PID and one direct
`Wait-Process` until natural completion. The warning is the existing
FastAPI/Starlette `TestClient` deprecation warning.
