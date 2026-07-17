# PrimeQA Hybrid Concurrent Runtime Validation Protocol

## Scope

Stage144 freezes the user-selected strict practical profile B for future
concurrent runtime implementation and validation. It reads only the saved
public-safe Stage143 aggregate and executes synthetic policy cases. It does not
load train, dev, test, questions, documents, models, indexes, or candidate
pools, and it does not execute concurrent runtime requests.

The concurrent implementation, concurrent performance result, runtime
defaultization, final locked-test evaluation, queues, retries, and fallback
strategies all remain closed in this stage.

## Confirmed Profile B

```text
profile id: strict_practical_b_concurrency4_v1
process state: one warm process
hardware scope: current benchmark machine only
maximum in-flight requests: 4
accepted-request end-to-end P95: <= 0.800s
accepted-request end-to-end P99: <= 1.500s
startup included in request SLO: false
startup reported separately: true
```

The Stage144 machine record is Windows 10, AMD64, 16 logical CPUs, and CPython
3.10.20. This is a local benchmark scope. The protocol forbids generalizing a
future result to other hardware without a separate run.

## End-To-End Measurement

The primary clock starts before the nonblocking admission attempt and stops at
the request's `complete`, `refuse`, or capacity-rejection terminal outcome.
For admitted requests it includes admission, candidate retrieval, answer
generation, verification, observation, and terminal trace construction.

Retrieval latency remains a separately reported diagnostic. Stage144 does not
invent an additional retrieval-only gate and does not substitute retrieval
latency for the confirmed end-to-end SLO. Accepted-request and capacity-
rejection latency distributions must be reported separately.

## Arrival Patterns

Every measured train pass uses cohorts with a maximum size of four.

| Pattern | Deterministic offsets inside each cohort |
| --- | --- |
| Synchronized four-request burst | `0, 0, 0, 0ms` |
| Deterministic jitter | `0, 7, 13, 20ms` |

The final partial cohort uses the corresponding prefix of the offset vector.
The jitter schedule is fixed rather than sampled so future repetitions are
directly comparable.

## Train Grouped-CV Workload

Each arrival pattern runs three complete 562-row train passes:

```text
complete passes per pattern: 3
complete measured passes total: 6
accepted requests per pattern: 1,686
accepted train requests total: 3,372
full four-request cohorts per pass: 140
final cohort size per pass: 2
```

The grouped five-fold assignments remain those already frozen by the project.
The validation matrix contains 39 latency scopes:

| Scope | Count |
| --- | ---: |
| Fold x pattern x repetition | 30 |
| Complete-pass aggregate | 6 |
| Pattern-pooled aggregate | 2 |
| Global pooled aggregate | 1 |

Every scope must independently satisfy both P95 and P99. Any failed or missing
percentile rejects the concurrency evidence. Dev cannot be loaded before all
train gates pass.

Train validation must also preserve Stage143 behavior across all 3,372
accepted requests: zero runtime/entrypoint trace violations, zero cross-request
contamination, candidate-pool depth 400, exact recall hit counts, terminal
counts, verified F1 and gold citations, and zero retry/fallback actions.

## Overload Contract

The overload probe submits five simultaneous attempts against capacity four.
A validation-only barrier holds the first four admitted requests after
admission and before retrieval while the fifth attempts admission. The exact
required outcome is:

```text
attempted: 5
admitted: 4
rejected: 1
rejection type: PrimeQAHybridConcurrentCapacityExceededError
rejected request downstream retrieval/Agent calls: 0
queue actions: 0
retry actions: 0
fallback actions: 0
```

Stage144 does not invent a numeric threshold for the word "immediate". It
defines immediate structurally: the rejected request must stop at admission,
before retrieval, Agent execution, or any other downstream call. Its measured
rejection latency is reported but is not mixed into accepted-request SLOs.

## Resource And State Safety

Future Stage145 implementation must preserve the exact Stage143 process-owned
inventory and factory build count:

```text
dense models: 2
dense embedding caches: 2
lexical indexes: 4
derived route: 1
candidate-pool retriever: 1
optional entrypoint: 1
resource factory builds: 1
resources built or loaded per request: false
```

Heavy retrieval resources remain shared and long-lived. Mutable retrieval
profiling, Agent state machines, entrypoint execution state, results, and
traces must be request-local. The Stage143 shared pending retrieval profile is
explicitly forbidden under concurrency. Cross-request trace or result
contamination rejects validation.

## Dev Report-Only Pass

After the complete train gate passes, Stage145 may load 121 dev rows once. Dev
uses 31 cohorts, alternating synchronized and deterministic-jitter patterns,
beginning with synchronized. There are 30 full cohorts and one final request.

Dev P95 and P99 must satisfy the same profile B limits, but dev remains report-
only: it cannot select an implementation, alter admission behavior, or retune
the workload or latency thresholds. Its trace, contamination, retrieval, and
Agent aggregate invariants must also match Stage143.

## Executable Policy

Stage144 adds an aggregate-only evidence model and fail-closed policy. The
policy checks profile identity, all train gate scopes, overload behavior,
resource ownership, request-local state, dev ordering, test lock, default
stability, and zero queue/retry/fallback actions.

Four synthetic cases prove the decision logic:

| Case | Result |
| --- | --- |
| Exact P95/P99 boundaries and all contracts satisfied | `eligible` |
| Train P95 exceeded and dev P99 missing | `rejected` |
| Five admitted, no rejection, and queue observed | `rejected` |
| Test/default opened and retry/fallback observed | `rejected` |

An `eligible` policy result never activates runtime by itself.

## Formal Stage144 Result

```text
status: primeqa_hybrid_concurrent_runtime_validation_protocol_frozen
guard checks: 29 / 29 passed
failed checks: []
concurrency validation policy executable: true
concurrent runtime implemented now: false
concurrent runtime validation run: false
concurrent runtime activation allowed now: false
runtime registered as default: false
runtime defaultization allowed now: false
test loaded / metrics run: false / false
queue / retry / fallback enabled: false / false / false
default runtime policy: unchanged
public forbidden keys: []
```

The final formal protocol freeze took `0.014430s`. That timing is only JSON loading,
synthetic policy evaluation, guard evaluation, and report construction. It is
not a concurrent runtime performance measurement.

## Visualizations

```text
stage144_end_to_end_latency_slo.svg
stage144_train_request_budget.svg
stage144_arrival_pattern_offsets.svg
stage144_latency_gate_matrix.svg
stage144_overload_contract.svg
stage144_process_resource_inventory.svg
stage144_decision_flags.svg
stage144_guard_check_status.svg
```

The JSON and SVG artifacts are local ignored aggregate-only files.

## Next Step

Stage145 is complete and recorded in:

```text
docs/primeqa_hybrid_concurrent_runtime_validation.md
```

It implements the bounded four-request research runtime, adds request-local
profiling and state, proves typed pre-downstream capacity rejection, and passes
the complete frozen train-CV/dev workload. Stage146 may now wire that validated
runtime behind a separate explicit non-default application activation path.
Test, defaultization, queues, retries, fallback, and network serving remain
separate closed gates.
