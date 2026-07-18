# Stage 155 Agent Runtime Activation And Observability Validation

## Scope

Stage155 freezes the activation evidence required by the graph-integrated
local Agent service and implements content-free node-level operational
observation. It does not enable the runtime by default, expose a remote bind,
open the locked test split, add a queue, retry an Agent action, or add a
fallback strategy.

The validated implementation is centered in:

```text
src/ts_rag_agent/application/primeqa_hybrid_agent_runtime_observability.py
src/ts_rag_agent/application/primeqa_hybrid_agent_tool_workflow.py
src/ts_rag_agent/application/primeqa_hybrid_concurrent_sidecar_agent_runtime.py
src/ts_rag_agent/application/primeqa_hybrid_concurrent_runtime_activation.py
src/ts_rag_agent/application/primeqa_hybrid_agent_service_entrypoint.py
```

## Strict Activation Chain

The local service still requires both explicit settings:

```text
TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT=true
TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT=true
```

Stage155 adds a strict gate before Stage145 loading, resource construction,
warmup, and listener binding. The service must load an accepted Stage154 formal
report and independently match the report's four source fingerprints against:

```text
Stage153 orchestration protocol artifact
pyproject.toml
Agent tool workflow source
concurrent runtime source
```

A missing or malformed Stage154 report, a failed guard, a changed decision
boundary, or any stale size/SHA-256 pair exits with the dedicated code `9` and
the public outcome `stage154_workflow_authorization_rejected`. Synthetic tests
prove that tampered Stage154 evidence and a stale runtime source are both
rejected before resource construction or socket binding.

The accepted current-code service fingerprints eleven sources: Stage150,
Stage154, the four Stage154 source dependencies, Stage145, and the four
retrieval resource inputs.

## Observation Contract

Each workflow invocation emits synchronous validated JSON-line events through
an `AgentWorkflowObservationSink`. Observation is always attached to the
eligible runtime; there is no disable flag, sampling, batching, remote export,
or background observation queue.

One successful or refused invocation emits exactly nine events:

```text
workflow_started
validate_request node_completed
retrieve_candidate_pool node_completed
prepare_context node_completed
compose_grounded_answer node_completed
verify_grounded_answer node_completed
observe_diagnostics node_completed
finalize_response node_completed
workflow_completed
```

Each event has an invocation-local sequence, a process-local invocation
sequence, node and cumulative monotonic elapsed time, current workflow state,
transition and tool counts, candidate/context depths, failure stage, and the
observed in-flight count. The 22-field allowlist contains no wall-clock
timestamp, request handle, question, answer, document ID, citation ID, or raw
content.

The observer keeps request-local state behind an inherited context token and a
locked token map. This preserves invocation identity across LangGraph's copied
execution context while four graph calls run concurrently.

## Synthetic Results

```text
complete path: 9 events / 7 node events / terminal complete
refuse path:   9 events / 7 node events / terminal refuse
candidate/context depths: 400 / 10 / 200
tool calls: 3

retrieval failure: 4 events
completed nodes before failure: 1
failed node: retrieve_candidate_pool
retrieval calls: 1
same exception object propagated: true
retry / fallback actions: 0 / 0

four concurrent calls: 36 events / 28 node events
maximum observed in flight: 4
request-isolation failures: 0
delivery failures: 0
graph compile count: 1

synthetic HTTP live / ready / answer: 200 / 200 / 200
HTTP workflow timeline: 9 events / 7 nodes
```

No train, dev, or test row is loaded. Synthetic request and document content is
created only in memory and is not written to the report.

## Stage154 Compatibility Evidence

Stage155 changes the workflow and concurrent runtime source, so the Stage154
artifact completed with commit `9815a79` no longer described the current source
fingerprints. The Stage154 validator was rerun against the current source and
the already recorded Stage154 real support artifact. No new Stage154 real
resource lifecycle was executed.

```text
current compatibility formal: 54 / 54
current compatibility preflight: 46 / 54

formal SHA-256:
af80d1dd9ba6b5ee7bc0e4f767182e922536d0f1547c86df1310a0d73b9bb416

preflight SHA-256:
bfebe4f38b5e0fa27a0c72882623a4f6ae8c3b8992e85137e66848db22c2a504
```

These are Stage155 compatibility-support artifacts. They do not retroactively
replace the Stage154 completion hashes recorded at the time of commit
`9815a79`.

## Preflight

The Stage155 unconfirmed preflight passes `48/57` guards. Its nine failures are
the explicit confirmation guard and the eight unavailable real lifecycle or
real observation checks. Every source, activation, synthetic workflow,
failure, concurrency, HTTP, privacy, and closed-boundary check passes.

```text
preflight SHA-256:
72e512c243fc7bb97a9a1e3e844508cfc05e78e09db59c688ce29e399911c4ec
```

## Real Lifecycle

The first formal command was launched without an explicit timeout, but the
shell tool imposed its own approximately 14-second default limit and returned
exit `124`. That attempt was externally interrupted, produced no Stage155
formal artifact, left no listener on port `18155`, and is not counted as a
validation pass.

The user explicitly selected option A to rerun. A single hidden process was
then launched on the same fixed port and observed until it exited naturally.
The orchestration layer initially failed to parse the returned PID text, so it
checked process state before doing anything else. Exactly one target Python
process was present; no duplicate process was started. That existing process
was then allowed to finish.

```text
formal validation total: 40.741661s
guard checks: 57 / 57
service exit code: 0
HTTP/1.1 live / ready / answer: 200 / 200 / 200
source fingerprints: 11
listener released: true
transport closed: true
port 18155 rebind after exit: true
test metrics run: false
```

The lifecycle includes one built-in label-free warmup and one real HTTP answer
request, both through the same compiled graph and observer:

```text
invocations: 2
events: 18
node events: 14
failed events: 0

warmup total: 679.789ms
  validate_request: 0.016ms
  retrieve_candidate_pool: 614.917ms
  prepare_context: 48.781ms
  compose_grounded_answer: 6.958ms
  verify_grounded_answer: 0.131ms
  observe_diagnostics: 0.241ms
  finalize_response: 0.006ms

answer request total: 59.595ms
  validate_request: 0.011ms
  retrieve_candidate_pool: 46.777ms
  prepare_context: 2.143ms
  compose_grounded_answer: 6.684ms
  verify_grounded_answer: 0.065ms
  observe_diagnostics: 0.205ms
  finalize_response: 0.008ms
```

The warmup is dominated by first retrieval at `614.917ms`. Once resources and
caches are warm, the answer request is dominated by retrieval at `46.777ms`;
answer composition is the second-largest measured node at `6.684ms`.

```text
formal SHA-256:
25ac42e3573f6c6d86fb367cabef7fe83955f17ed79bb86b53585823fb423eef
```

Formal and preflight each contain ten parseable SVG charts. Artifacts and
visuals remain under the ignored `artifacts/` directory.

Final repository validation after the Stage155 source was frozen is:

```text
Ruff: passed
pytest: 712 passed, 1 existing warning in 11.37s
```

The warning is the existing FastAPI `TestClient` Starlette deprecation
warning. It is not hidden or counted as a Stage155 failure.

## Decision

The graph runtime now has a strict current-evidence activation chain and
request-isolated node-level operational observation. It remains an explicit,
nondefault, loopback-only local service. Remote serving, test evaluation,
queues, retries, fallback, LLM-selected tools, and multi-turn memory remain
closed.

The next research direction is to design the boundary for local Agent tool
selection and multi-turn state before implementing autonomous routing.
