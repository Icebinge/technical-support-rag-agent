# PrimeQA Hybrid Agent Request-Facade Validation

## Scope

Stage148 implements the transport-neutral application Agent request facade
authorized by Stage147. It does not implement FastAPI or another network
surface, register a default runtime, access test, or add queues, retries, or
fallback.

The formal validator reads only the saved Stage147 public aggregate and uses
synthetic runtimes. It loads no dataset rows, documents, models, indexes, or
candidate pools. A separate unit integration test builds a 400-document
synthetic in-memory pool to prove that the new label-free query can traverse
the real concurrent runtime, retriever, entrypoint, orchestrator, answer
generator, and verifier without any gold fields.

## Label-Free Query

The serving path now consumes the structural `PrimeQAQuery` protocol. Offline
evaluation continues to use `PrimeQAQuestion`, while the facade creates this
separate model:

```text
PrimeQARuntimeQuery
  id
  title
  text
```

It has no answer, answerability label, answer document reference, candidate
document membership, offsets, or test membership. The real online integration
test completes with this object; any attempted gold-field read would raise
instead of silently receiving a fabricated placeholder.

## Facade Contract

The facade is constructible only from an eligible, warm, active Stage146
bootstrap result. It does not own or destroy runtime resources.

Private request:

```text
request_handle
title (optional)
text
cancellation_signal (optional)
```

Private response:

```text
request_handle
text
refused
citations(document_reference, title, rank, evidence_score)
```

The private payload has no public serializer. `AgentRequestFacadeRun` exposes a
separate public view containing the exact six-field facade event and the
existing fourteen-field concurrent runtime trace. Request content, response
content, handles, and document references do not enter that public view.

## Errors And Cancellation

```text
invalid input -> AgentRequestFacadeInvalidRequestError
pre-dispatch cancellation -> AgentRequestFacadeCancelledError
runtime capacity -> AgentRequestFacadeCapacityExceededError
draining -> AgentRequestFacadeDrainingError
closed -> AgentRequestFacadeClosedError
inactive bootstrap -> AgentRequestFacadeNotActiveError
other downstream exception -> same exception object is re-raised
```

Invalid and pre-cancelled calls make zero runtime calls. Capacity rejection
preserves `PrimeQAHybridConcurrentCapacityExceededError` as the cause and its
public `rejected_capacity` trace. It is not queued, retried, converted into an
answer, or sent into retrieval/generation. Unknown downstream failures are
recorded in request-local `ContextVar` telemetry and propagated unchanged.

Cancellation is checked immediately before synchronous runtime dispatch. The
facade does not claim in-flight hard cancellation.

## Lifecycle

The implemented lifecycle is monotonic:

```text
accepting -> draining -> closed
```

The concurrency test holds one request inside a blocking synthetic runtime,
starts shutdown, waits for the observable `draining` state without a timeout,
and confirms that a new call is rejected. Releasing the original request lets
it finish naturally; shutdown then reaches `closed` with zero in-flight work.
Closed calls are rejected and repeated shutdown is idempotent. No implicit
timeout, force-cancel, or facade-owned resource cleanup exists.

## Source Gate

The real unconfirmed CLI preflight stopped before synthetic validation:

```text
source gate: false
failed check: stage148_user_confirmed
synthetic validation executed: false
questions / models / candidate pools: false / false / false
```

The confirmed formal run re-read the complete Stage147 JSON after validation
and proved it unchanged. Its saved source SHA-256 is
`3f72d7a8bd89d1b791c89d41d038e25c38e14ab053cc162c0fc0792fcf4dc860`,
which matches the actual file.

## Formal Result

```text
Stage147 source guards: 34 / 34
Stage148 guards: 37 / 37
source gate passed: true
synthetic validation executed: true
runtime query fields: id / text / title
forbidden runtime query attributes: []
public facade / runtime trace fields: 6 / 14
invalid / pre-cancel runtime calls: 0 / 0
capacity facade code / runtime state: capacity_exceeded / rejected_capacity
downstream same-object propagation: true
shutdown in-flight before release: 1
natural completion / closed: true / true
implicit timeout / force-cancel: false / false
formal time: 0.001964s
network / default / test: false / false / false
queue / retry / fallback: false / false / false
public forbidden keys: []
```

## Process Corrections

The initial implementation/refactor suite passed all 61 relevant tests. The
facade and formal validator suite then passed 13 tests. After adding the real
online-chain integration test, the first run was `13 passed / 1 failed`: the
synthetic documents produced a valid verified refusal, while the test had
incorrectly required a completed answer. No threshold was changed and no
evidence was tailored. The assertion was corrected to verify the real refusal,
Top400 depth, and application trace, after which all 14 Stage148 tests passed.

Final repository verification:

```text
changed/new Python format check: 16 files already formatted
full repository Ruff: passed
full repository pytest: 564 passed in 6.67s
formal Stage148 guards: 37 / 37
formal / preflight SVG XML parse: 10 / 10 and 10 / 10
formal / preflight artifact ignore checks: passed
```

## Visualizations

```text
stage148_source_gate.svg
stage148_runtime_query_boundary.svg
stage148_private_response_mapping.svg
stage148_public_telemetry_fields.svg
stage148_error_outcomes.svg
stage148_dispatch_counts.svg
stage148_lifecycle.svg
stage148_closed_boundaries.svg
stage148_decision_flags.svg
stage148_guard_check_status.svg
```

## Next Step

Stage149 should freeze the network transport protocol around this facade:
serialization schema, HTTP error/status mapping, request-size policy,
disconnect semantics, lifespan ownership, health/readiness behavior, and
public logging boundary. FastAPI implementation follows only after that
protocol passes. Test, defaultization, queues, retries, and fallback remain
closed.
