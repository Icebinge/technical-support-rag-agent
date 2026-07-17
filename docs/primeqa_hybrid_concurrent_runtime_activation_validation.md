# PrimeQA Hybrid Concurrent Runtime Application Activation Validation

## Scope

Stage146 wires the Stage145 concurrency-four research runtime into an explicit,
disabled-by-default application bootstrap. It validates configuration parsing,
source-evidence recomputation, fail-closed startup, one-time resource ownership,
warmup, overload behavior, and the complete Stage145 train/dev workload through
the runtime returned by the application bootstrap.

This stage does not register a default runtime, add a network service, evaluate
test, queue overload, retry requests, or add fallback behavior.

## Configuration Contract

```text
settings field: enable_concurrent_sidecar_agent
environment flag: TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT
default: false
accepted values: literal true / false
mutually exclusive with: enable_optional_sidecar_agent
```

If both the existing single-request flag and the new concurrent flag are true,
`ProjectSettings` rejects the configuration. The application cannot
accidentally initialize two process-owned retrieval graphs.

## Source Evidence

An explicit `true` is necessary but insufficient. The bootstrap recomputes the
complete Stage144 policy evidence from the saved Stage145 aggregate, evaluates
it again, and compares the recomputed evidence with the evidence stored in that
report. It also requires the Stage145 identity, all `36/36` source guards,
wiring authorization, public-safety contract, test lock, and unchanged default.

The formal rejected case uses an in-memory deep copy of Stage145 and changes
only train global P95 to `0.800001s`. It is rejected before resource
initialization because both the recomputed policy and saved-evidence comparison
fail. The original Stage145 JSON is then reloaded and compared in full with the
pre-mutation object; equality is true. The synthetic copy is never persisted.

## Startup States

| Case | Requested | Source state | Resource builds | Warmup | Active |
| --- | ---: | --- | ---: | ---: | ---: |
| Disabled | no | not evaluated | 0 | 0 | no |
| Rejected | yes | rejected | 0 | 0 | no |
| Eligible | yes | eligible | 1 | 1 | yes |

Eligible startup builds two dense models, two dense caches, four lexical
indexes, one derived route, one candidate-pool retriever, and one Agent
entrypoint. No resource is built per request. The final warmup produced Top400
with retrieval/end-to-end latency `256.477/302.702ms`.

## Formal Workload

The runtime returned by eligible application startup runs the complete frozen
workload rather than a smoke test:

```text
train rows: 562
complete train passes: 6
accepted train requests: 3,372
latency gate scopes: 39
overload attempts: 5
dev rows after train gate: 121
test rows loaded: 0
```

## Train Result

| Scope | Requests | P95 | P99 | Max | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Synchronized pooled | 1,686 | `0.574038s` | `0.779176s` | `0.974289s` | pass |
| Deterministic-jitter pooled | 1,686 | `0.538408s` | `0.730639s` | `0.850078s` | pass |
| Global pooled | 3,372 | `0.559442s` | `0.755804s` | `0.974289s` | pass |

The worst P95 scope was synchronized repetition 1 fold 3 at `0.687264s`.
The worst P99 scope was synchronized repetition 1 fold 2 at `0.866313s`.
All `39/39` scopes meet profile B P95 `<=0.800s` and P99 `<=1.500s`.
Retrieval-only train P95/P99 was `0.485895/0.691303s`.

## Arrival Fidelity

Synchronized repetition maximum offsets were `1.4664ms`, `0.3473ms`, and
`0.6197ms`. Across every train request, target-offset absolute error was
`3.762444ms` on average, `14.007355ms` at P95, `15.737221ms` at P99, and
`303.5484ms` at maximum.

The maximum deviations occurred in deterministic-jitter workers after their
target sleeps: `303.5484ms`, `279.8377ms`, and `277.2259ms` across the three
repetitions. The target schedule remains exact, but Windows/Python scheduling
is not physically deterministic under CPU load. Stage144 did not define this
diagnostic as an acceptance gate, so it is reported as a machine-specific load
generator limitation without changing the frozen decision rule.

## Overload And Behavior

```text
attempted / admitted / rejected: 5 / 4 / 1
completed admitted / failed admitted: 4 / 0
maximum in flight: 4
rejected downstream calls: 0
typed rejection latency: 0.004ms
queue / retry / fallback actions: 0 / 0 / 0
```

Application activation preserves Stage145 behavior:

```text
train Recall@10/50/100/200/400: 0.6892 / 0.8189 / 0.8973 / 0.9324 / 0.9568
train verified F1 / gold citations: 0.1946 / 906
train terminal complete / refuse: 3,360 / 12
cross-request contamination: 0
dev end-to-end P95 / P99: 0.539259 / 0.695918s
dev Recall@10/50/100/200/400: 0.7237 / 0.8421 / 0.8684 / 0.9079 / 0.9211
dev verified F1 / gold citations: 0.1873 / 33
dev terminal complete / refuse: 121 / 0
```

## Formal Decision

```text
status: primeqa_hybrid_concurrent_runtime_application_activation_validation_passed
guard checks: 43 / 43 passed
latency scopes: 39 / 39 passed
application activation bootstrap implemented: true
disabled / rejected / eligible startup validated: true
eligible runtime full workload passed: true
explicit non-default concurrent activation available: true
activation requires explicit true and Stage145 evidence: true
single and concurrent flags mutually exclusive: true
runtime registered as default: false
runtime defaultization allowed: false
network service implemented: false
test loaded / metrics run: false / false
queue / retry / fallback enabled: false / false / false
public forbidden keys: []
```

The final validator phases through guard evaluation completed naturally in
`487.615s`. JSON serialization and SVG generation happen after that internal
timer. An earlier `486.508s` run also passed, but it only rechecked one source
field after the synthetic rejection audit. It is retained as process history,
not used as the final result.

## Visualizations

```text
stage146_startup_states.svg
stage146_startup_resource_builds.svg
stage146_train_pattern_latency.svg
stage146_train_scope_maxima.svg
stage146_dev_latency.svg
stage146_overload_outcome.svg
stage146_arrival_fidelity.svg
stage146_decision_flags.svg
stage146_guard_check_status.svg
```

All nine SVGs parse successfully. JSON and SVG outputs remain local ignored
artifacts.

## Next Step

Stage147 should freeze a non-default application Agent request-facade protocol
before implementing FastAPI or another network surface. It should define the
call contract, lifecycle ownership, capacity-error mapping, public response and
trace allowlists, cancellation/error propagation, and shutdown behavior. Test,
defaultization, queues, retries, and fallback remain separate closed gates.
