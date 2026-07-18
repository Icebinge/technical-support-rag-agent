# Stage 158 Bounded Dynamic Agent Local Service Validation

## Objective

Stage 158 integrates the Stage 157 bounded dynamic Agent runtime into a
separate, explicitly activated, loopback-only FastAPI/Uvicorn service. It does
not replace or modify the existing `/v1/agent/answers` service.

The user selected protocol A: three exact lifecycle endpoints instead of one
action-dispatch endpoint.

```text
POST /v1/bounded-agent/threads/open
POST /v1/bounded-agent/threads/turn
POST /v1/bounded-agent/threads/close
GET  /health/live
GET  /health/ready
```

No train, development, or test row is loaded. The real HTTP question is a
generated label-free serving query, and no answer-quality metric is computed.

## Activation And Startup

Both environment flags must be explicitly true:

```text
TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_RUNTIME
TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_HTTP_TRANSPORT
```

`TS_RAG_BOUNDED_DYNAMIC_AGENT_MODEL_SNAPSHOT` must point to the exact local
Qwen snapshot. The new runtime is mutually exclusive with the existing
optional and concurrent sidecar runtimes. All new settings default to false or
unset.

Startup order is fixed:

```text
configuration authorization
Stage157 artifact and current router/runtime source authorization
model snapshot fingerprint authorization
CPU retrieval resource construction
local Qwen GPU load
label-free warmup in an explicit temporary thread
warmup thread close
FastAPI composition
loopback listener startup
```

The source gate runs before resource construction. CPU retrieval resources are
built before loading the GPU model. The listener is not opened until warmup
completes and its temporary process-local thread is closed.

The Stage157 artifact gate uses exact SHA-256
`2351015d2c7447e6a5e1c2fe99b6583f0b9067e126ef2bfdd87b0b80c725c3e1`,
all `47/47` Stage157 guards, exact Stage157 status, exact router/runtime source
fingerprints, and exact model config/weights/tokenizer fingerprints.

## Admission And Lifecycle

The service coordinator owns the explicit open-handle set, active-turn set,
and one global whole-turn admission slot under one lock. A turn must acquire
admission before it is submitted to the single worker. This prevents the
executor from becoming an implicit application waiting queue.

Only `turn` consumes the GPU slot. `open` and `close` are short process-local
ledger operations. A second thread turn is immediately rejected with HTTP
`503`; a second simultaneous turn or close on the same thread is rejected with
HTTP `409`. Duplicate open is `409`, and an unknown thread is `404`.

Shutdown first enters draining state, rejects new operations, naturally waits
for an already admitted turn, then closes every remaining process-local thread.
There is no request timeout, shutdown timeout, queue, retry, parser repair,
fallback, implicit thread creation, eviction, truncation, checkpointer, or
persistent store.

Public events contain only route, status, coordinator and admission state,
aggregate thread counts, branch call counts, token counts, generation latency,
and zero recovery counters. They contain no thread handle, question, answer,
citation, document identifier, raw model output, headers, or client identity.

## Environment

The Stage157 GPU environment was extended with the existing project `[app]`
extra. The independent script verification completed with:

```text
torch: 2.11.0+cu128
torchvision: 0.26.0+cu128
transformers: 5.13.1
fastapi: 0.139.2
uvicorn: 0.51.0
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5060
capability: 12.0
pip check: no broken requirements
```

The environment check is implemented as
`scripts/verify_stage158_environment.py`; it avoids PowerShell inline-code
quoting and writes no artifact.

## Corrected Formal Real Service Result

The current-source formal process ran after an explicit user confirmation to
replace pre-correction evidence. It used port `18158`, built retrieval resources
once, loaded Qwen once, generated once for pre-listener warmup, generated once
for the real HTTP turn, shut down through Uvicorn's normal exit flag, joined the
non-daemon server thread without a timeout, and released the port.

```text
guards: 51 / 51
HTTP live / ready / open / turn / close: 200 / 200 / 201 / 200 / 200
public events: 5
resource factory builds: 1
model generations: 2
maximum observed in-flight turns: 1
opened threads after shutdown: 0
server thread alive after shutdown: false
port 18158 listener after shutdown: 0
port rebind after shutdown: true
queue / retry / fallback: 0 / 0 / 0
peak allocated GPU memory: 5,358,983,168 bytes
SVG: 10 / 10 parseable
```

The label-free pre-listener warmup selected refusal:

```text
selected action / terminal: refuse_insufficient_evidence / refuse
retrieval / model decision: 1 / 1
composition / verification / diagnostics: 0 / 0 / 0
input / output tokens: 2,190 / 11
generation latency: 1,883.706 ms
retained state before close: 1 turn / 158 bytes
thread opened after close: false
```

The generated label-free real HTTP turn also selected refusal:

```text
selected action / terminal: refuse_insufficient_evidence / refuse
retrieval / model decision: 1 / 1
composition / verification / diagnostics: 0 / 0 / 0
input / output tokens: 2,117 / 11
generation latency: 1,055.072 ms
retained state before close: 1 turn / 188 bytes
response citations: 0
```

Measured startup and lifecycle times were:

```text
source authorization: 3.563386 s
retrieval resource build: 53.588927 s
model load: 7.472638 s
pre-listener warmup: 2.435760 s
app composition: 0.001807 s
total prepare: 67.062519 s
server start: 0.025108 s
real HTTP sequence and shutdown: 1.409058 s
formal total: 68.564630 s
```

The real turn demonstrates serving and lifecycle compatibility, not routing
quality. Both model decisions were refusals on generated label-free queries;
there is no labeled evidence for decision accuracy or false-refusal rate.

The nonblocking capacity path was validated with deterministic synthetic
overlap in the coordinator and targeted tests. Stage158 did not issue two
simultaneous real Qwen HTTP turns, so it does not claim a measured real-GPU
capacity-rejection latency.

The formal artifact is
`artifacts/primeqa_hybrid_bounded_dynamic_agent_service_stage158.json` with
SHA-256 `12649c087c3140feeb4121837152b41ef4005922eb73931f3770a5fac83889b0`.

## Validation Process Corrections

The first targeted test process reached eight passing tests and then deadlocked
inside its concurrency fixture. One `TestClient` serialized the second request,
while the fixture waited for that response before releasing the first synthetic
turn. CPU and output remained unchanged. After explicit user approval, PID
`50464` was stopped. The product coordinator had not deadlocked. The test was
changed to drive coordinator admission directly, and HTTP `503` mapping was
tested separately. Corrected targeted results progressed from `41 passed` to
`44 passed`, then to the pre-audit `47 passed, 1 warning in 1.26s`.

After installing `[app]`, a foreground combined environment command was
externally stopped by the shell wrapper after approximately 14 seconds and
produced no verification result. The user approved a detached replacement, but
the first replacement incorrectly passed inline Python through PowerShell and
exited with `SyntaxError` before checking the environment. No formal process
had started. The inline approach was removed; after user confirmation, the
standalone environment script above ran once and passed. The subsequent formal
Stage158 process passed `51/51` guards. Its artifact SHA-256 was
`1358ce88bd494079dfc806ad3416e87c279f14947436122e89d6452e68d937b1`, and the
then-current full repository result was `783 passed, 1 warning in 15.94s`.

A later pre-commit audit found that a startup failure after source authorization
could emit a terminal event with `source_authorized=false`. The underlying
startup still failed closed, but the public terminal progress was inaccurate.
The entrypoint now tracks completed source, resource, and model stages
explicitly, and it also verifies `server.started` after `server.run`. New tests
cover source-, resource-, and model-stage failure progress. This changed the
entrypoint source after the first formal artifact, so that artifact and the
`783 passed` suite are retained only as pre-correction history.

No corrected formal run was started without approval. After the user explicitly
selected A, exactly one corrected formal process ran naturally, with no runtime
timeout, monitoring deadline, kill, restart, or automatic retry. It passed
`51/51`, produced the current hash above, left zero validation processes and
zero listeners on `18158`, and produced ten XML-parseable SVGs.

Current-source repository validation is:

```text
Ruff: passed
targeted pytest: 65 passed, 1 existing warning in 1.33 s
full pytest: 786 passed, 1 existing warning in 10.80 s
full pytest stderr: empty
```

The warning is the existing FastAPI `TestClient` Starlette deprecation warning.

## Next Boundary

Stage159 may measure warm multi-turn service behavior on the locked development
split only, including history growth, second-to-fourth-turn latency, answer versus
refusal distribution, and a real two-request admission rejection. Test remains
locked. Runtime defaultization, remote exposure, persistence, queues, retries,
fallback, query rewrite, and second retrieval remain closed.
