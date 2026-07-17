# PrimeQA Hybrid Agent Request-Facade Protocol

## Scope

Stage147 freezes an executable, transport-neutral protocol for the application
Agent request facade. It reads only the saved Stage146 public aggregate and
runs synthetic policy cases. It does not instantiate the facade, start a
network service, load any data or model, build a candidate pool, change the
runtime default, or access the locked test split.

This separation is deliberate: the existing `PrimeQAQuestion` is an evaluation
sample containing gold labels. A serving facade must accept a private,
label-free request and create the runtime adapter internally. It cannot expose
evaluation labels or reuse the evaluation schema as a public API.

## Private Call Contract

The private request contains:

```text
request_handle
title (optional)
text
cancellation_signal (optional)
```

Gold answer, answerability label, gold document reference, and test membership
are forbidden. The private response contains the request handle, answer text,
refusal state, and citations. Each citation contains the document reference,
title, retrieval rank, and evidence score already produced by the current
domain model. Candidate pools and raw Agent action traces are not returned.

The contract intentionally does not invent a request-size limit. That limit,
along with serialization and validation details, belongs to the later network
transport protocol.

## Public Telemetry

Public telemetry is allowlist-only. The runtime trace may expose activation and
admission state, concurrency limit, candidate depth, retrieval/end-to-end
latency, latency-budget outcome, and terminal state. Facade events may expose
facade state, outcome code, whether downstream work was dispatched, and zero
queue/retry/fallback counters.

Question content, answer content, request handles, document identifiers,
candidate rows, and raw action traces remain private.

## Error Contract

```text
invalid input -> invalid_request, rejected before downstream
runtime capacity error -> capacity_exceeded, exact typed mapping
draining facade -> facade_draining
closed facade -> facade_closed
downstream failure -> propagate unchanged
```

Errors are not converted into answer payloads. Capacity rejection remains
nonblocking and occurs before downstream retrieval. No queue, automatic retry,
or fallback is introduced.

## Cancellation And Shutdown

Cancellation is cooperative and checked immediately before runtime dispatch.
If already cancelled, the call terminates without downstream work. Once the
synchronous runtime call has begun, Stage147 does not claim hard cancellation:
the runtime completes or raises and releases its permit in `finally`.

The facade lifecycle is:

```text
accepting -> draining -> closed
```

Shutdown first enters `draining`, rejects new calls, and then waits naturally
for all in-flight calls to finish. There is no implicit timeout, force-cancel,
or facade-owned resource destruction. Process bootstrap retains ownership of
the shared runtime resources. A closed facade cannot reopen.

## Formal Result

```text
source Stage146 guards: 43 / 43
Stage147 guards: 34 / 34
canonical policy cases: 1 eligible / 4 rejected
SVG visualizations: 10
formal protocol time: 0.001102s
train / dev / test loaded by Stage147: false / false / false
models / indexes loaded: false / false
candidate pools built: false
public forbidden keys: []
```

The four negative cases reject source/default drift, content or label leakage,
unsafe capacity/error behavior, and unsafe cancellation/shutdown behavior.

## Process Corrections

The first targeted command used a repository-local virtual environment path
that does not exist, so Python never started. The real environment then
reported `7 passed / 1 failed`; the failed assertion expected 32 guards while
the implementation correctly produced 33 at that point, and Ruff found one
unused import.
After correction, the targeted suite passed all eight tests.

The first formal CLI run passed 32 of 33 guards and was rejected because the
Stage147 reader expected `eligible_runtime_full_workload_passed`, while the real
Stage146 artifact uses `eligible_runtime_full_workload_validation_passed`. The
fixture and reader were aligned to the real saved contract, and the second run
passed all then-current 33 guards. A final contract audit added one explicit
guard proving that all 14 existing Stage146 runtime-trace fields are retained,
so the final result is 34/34. The rejected artifact was not accepted as final.

## Visualizations

```text
stage147_source_activation_boundary.svg
stage147_private_call_contract.svg
stage147_public_telemetry_allowlist.svg
stage147_error_mapping.svg
stage147_cancellation_boundary.svg
stage147_lifecycle.svg
stage147_shutdown_contract.svg
stage147_policy_cases.svg
stage147_decision_flags.svg
stage147_guard_check_status.svg
```

## Next Step

Stage148 should implement the transport-neutral facade against this frozen
protocol and validate request conversion, exact private response mapping,
public telemetry, capacity rejection, lifecycle races, cooperative
pre-dispatch cancellation, downstream exception propagation, and natural
shutdown using synthetic runtimes. FastAPI, test evaluation, defaultization,
queues, retries, and fallback remain closed.
