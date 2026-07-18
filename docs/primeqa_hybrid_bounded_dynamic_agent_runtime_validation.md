# Stage 157 Bounded Dynamic Agent Runtime Validation

## Objective

Stage 157 implements the Stage 156 protocol as an explicit, nondefault local
runtime. It adds one local-files-only Qwen structured decision after the frozen
retrieval and context-preparation steps, a conditional LangGraph branch, and a
process-local thread ledger with the user-selected limits of four completed
turns and 32 KiB.

The runtime remains outside the HTTP service and runtime default. No train,
development, or test question row is loaded, and no answer-quality metric is
computed.

## Selected Environment

The project-owned `.venv` is separate from the existing base CPU environment:

```text
virtual environment torch: 2.11.0+cu128
torchvision: 0.26.0+cu128
transformers: 5.13.1
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5060
capability: 12.0
base-environment torch: 2.13.0+cpu
```

The model is the already cached `Qwen/Qwen3-VL-2B-Instruct` snapshot at
revision `89644892e4d85e24eaac8bacfd4f463576704203`. Loading is strict
`local_files_only`; remote model access is not available. The model uses
BF16, SDPA, greedy decoding, at most 32 generated tokens, and one nonblocking
GPU generation slot.

PyTorch CUDA wheels remain an explicit environment provisioning step because
their index cannot be represented safely by a normal project extra. The
`local-router` extra pins `transformers==5.13.1`.

## Structured Router

The model may emit exactly one of these JSON objects:

```json
{"action":"compose_grounded_answer"}
```

```json
{"action":"refuse_insufficient_evidence"}
```

Pydantic rejects malformed JSON, markdown fences, trailing text, extra fields,
and unauthorized actions. There is no parser repair, retry, or fallback. The
model receives the current private question, completed private thread history,
and the Top10 generation context. The user-selected prompt profile exposes each
context title and at most 600 body characters, rejects inputs over 12,288
tokens before generation, and never silently truncates tokenized input.

Prompt text, history, model output, and model reasoning are not written to the
public artifact. The router metrics carried through private graph state contain
only token counts, latency, schema validity, and the selected action.

## Conditional Runtime

The nine-node graph has one conditional routing source and two targets:

```text
validate -> retrieve -> prepare -> select
  compose -> compose -> verify -> diagnostics -> verified finalize
  refuse  -> fixed system refusal
```

Retrieval and model selection each execute exactly once. The early-refusal
branch does not call composition, verification, or diagnostics. The compose
branch calls all three exactly once, and only the verifier-owned answer can be
returned. The graph has no checkpointer, cache, loop, query rewrite, second
retrieval, queue, retry, or fallback.

Each thread must be opened explicitly with an opaque handle. Only completed
user input, verified terminal response, sequence, and terminal state survive
to the next turn. A fifth turn or a byte overflow is rejected before ledger
mutation. Explicit close removes the state, and process restart loses it.

## Validation Results

The first real GPU probe used generated synthetic evidence and made one model
call. It selected `compose_grounded_answer` with a valid schema:

```text
input/output tokens: 696 / 9
generation latency: 905.026 ms
model load latency: 12.199471 s
peak allocated GPU memory: 4,463,856,128 bytes
```

The confirmed formal run used the real technote corpus and existing six-channel
retrieval graph, while forcing the two dense encoders to CPU so the Qwen router
remained the sole GPU workload. Its generated label-free runtime question is
not an evaluation row. Qwen selected the early-refusal action:

```text
candidate / generation / verification depth: 400 / 10 / 200
selected action / terminal state: refuse_insufficient_evidence / refuse
retrieval / model decision calls: 1 / 1
composition / verification / diagnostics: 0 / 0 / 0
input/output tokens: 2,190 / 11
generation latency: 1,793.562 ms
retained turns / bytes before close: 1 / 158
thread opened after close: false
peak allocated GPU memory: 5,358,983,168 bytes
```

Measured formal phase times were 3.354144 seconds for source fingerprinting,
13.835044 seconds for model loading, 40.714994 seconds for retrieval resource
construction, 2.686443 seconds for the real turn, and 60.651182 seconds total.

All five synthetic cases pass: compose branch, early-refuse branch, malformed
schema rejection, unauthorized-action rejection, and cross-thread isolation.
The formal result is:

```text
guards: 47 / 47
targeted tests: 46 passed
SVG: 10 / 10 parseable
forbidden public keys: []
train / dev / test loaded: false / false / false
test metrics run: false
```

The formal artifact is
`artifacts/primeqa_hybrid_bounded_dynamic_agent_runtime_stage157.json`, with
SHA-256 `2351015d2c7447e6a5e1c2fe99b6583f0b9067e126ef2bfdd87b0b80c725c3e1`.

Current-source repository validation completed after the formal run. All
Stage157 Python files pass Ruff formatting, the complete repository passes Ruff
lint, and one hidden full-suite pytest process exited naturally with
`758 passed, 1 warning in 15.73s`; stderr was empty. The warning is the existing
FastAPI `TestClient` Starlette deprecation warning. A separate repository-wide
format check reports 311 historical Python files that would be reformatted;
they are outside Stage157 and were not changed.

These results prove execution boundaries and local compatibility, not routing
quality. The actual model selected compose on generated synthetic evidence and
refuse on one real-corpus label-free question, but no labeled split was used to
estimate accuracy, false-refusal rate, answer F1, or citation quality.

## Process Corrections

The first installation attempt left `torch 2.13.0+cpu` and no `torchvision`.
After explicit selection, the environment was rebuilt manually and then
repaired in place with exact CUDA wheels. The user's terminal and an independent
background check both confirmed CUDA; one earlier combined independent check
was externally stopped by the shell's short default limit and produced no
result.

The first expanded test collection failed because the validation module
indirectly imported optional Uvicorn code. The unnecessary dependency was
removed. A later test found Python 3.11-only `hashlib.file_digest`; it was
replaced by Python 3.10-compatible streaming SHA-256.

The first formal run completed model loading, retrieval, the graph turn, and
thread close, but report assembly failed because `ContextVar` router metrics did
not cross the LangGraph node boundary. It produced no formal artifact. Metrics
were then carried explicitly in private graph state, targeted tests returned to
46 passing, and the user explicitly approved one complete formal rerun. That
rerun exited naturally and produced the accepted evidence above.

A separate monitoring command containing `Start-Sleep 10` was stopped by the
shell wrapper at ten seconds. It did not affect the detached formal process,
which continued under its original PID. No legitimate model or validation
process was stopped, restarted, or time-limited.

## Next Boundary

Stage 158 should design the explicit activation and lifecycle contract for
integrating this bounded runtime into the existing loopback service. Before
that integration, it must resolve GPU admission versus the existing four-call
CPU concurrency contract and define thread open/turn/close transport semantics.
Runtime defaultization, remote exposure, persistent state, and test evaluation
remain closed.
