# PrimeQA Hybrid Concurrent Runtime Validation

## Scope

Stage145 implements the bounded-concurrency research runtime authorized by the
Stage144 strict practical profile B protocol and validates it on the current
benchmark machine. The implementation shares process-owned models, embedding
caches, lexical indexes, the candidate-pool retriever, and one Agent entrypoint
while keeping retrieval profiling and Agent state request-local.

This stage does not register the concurrent runtime for application use, expose
a network service, change the default runtime, or evaluate test. Queues,
retries, and fallback strategies remain prohibited.

## Runtime Architecture

```text
process-owned heavy resource graph: built once
maximum in-flight requests: 4
admission: nonblocking BoundedSemaphore
capacity outcome: typed pre-downstream rejection
request-local retrieval profile: ContextVar
request-local Agent state machine: yes
shared mutable pending request state: prohibited
resources built or loaded per request: false
```

Each question still requires online retrieval to produce its question-specific
ordered Top400 result set. That is not an index or model rebuild: all expensive
corpus-level resources are already warm and shared. The measured request path
includes admission, online candidate retrieval, answer generation,
verification, observation, and public terminal trace construction.

## Arrival Harness

Each train cohort has at most four requests. A per-cohort four-party barrier
makes every worker ready before one shared clock origin is recorded. Workers
then target either:

| Pattern | Target offsets |
| --- | --- |
| Synchronized burst | `0, 0, 0, 0ms` |
| Deterministic jitter | `0, 7, 13, 20ms` |

The barrier fixed an audit defect in the first implementation, where the clock
started before all worker tasks were ready. In the final run, the maximum
synchronized-burst offset was `0.6999ms` across all three train repetitions.

The target jitter schedule is deterministic, but Windows thread scheduling is
not. Across all train requests, arrival-offset absolute error was
`3.421241ms` on average, `14.249230ms` at P95, `15.815290ms` at P99, and had one
`286.9019ms` maximum outlier in jitter repetition 2. This is reported as a
workload-fidelity diagnostic, not silently treated as exact physical arrival.
Stage144 froze exact target offsets but did not define an actual-offset error
gate, so this diagnostic does not alter the frozen decision rule.

## Formal Workload

```text
train rows: 562
train grouped folds: 5
complete repetitions per pattern: 3
complete train passes: 6
accepted train requests: 3,372
latency gate scopes: 39
overload attempts: 5
dev rows after train gate: 121
test rows loaded: 0
```

The 39 independent train scopes are 30 fold-pattern-repetition scopes, 6
complete-pass scopes, 2 pattern-pooled scopes, and 1 global-pooled scope. Every
scope must satisfy end-to-end P95 `<= 0.800s` and P99 `<= 1.500s`.

## Train Result

| Scope | Requests | P95 | P99 | Max | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Synchronized pooled | 1,686 | `0.574487s` | `0.760026s` | `0.874202s` | pass |
| Deterministic-jitter pooled | 1,686 | `0.564124s` | `0.764513s` | `1.055940s` | pass |
| Global pooled | 3,372 | `0.569697s` | `0.763205s` | `1.055940s` | pass |

The worst P95 scope was synchronized repetition 3 fold 5 at `0.682807s`.
The worst P99 scope was deterministic-jitter repetition 2 fold 3 at
`0.875067s`. All `39/39` latency scopes passed.

Retrieval-only train P95/P99 was `0.497690/0.693310s`. It remains diagnostic;
the formal gate uses the full end-to-end request latency.

## Overload Result

```text
attempts: 5
admitted: 4
capacity rejected: 1
completed admitted: 4
failed admitted: 0
maximum observed in flight: 4
rejected downstream calls: 0
typed error: PrimeQAHybridConcurrentCapacityExceededError
rejected end-to-end latency: 0.004ms
queue / retry / fallback actions: 0 / 0 / 0
```

The four admitted requests were held after admission and before downstream
work, proving that the fifth request was rejected against real full capacity
rather than a timing coincidence.

## Behavior Preservation

The final run built the shared resource graph once and used one label-free
warmup. Warmup retrieval/end-to-end latency was `59.704/109.715ms`.

Train aggregates repeat the Stage143 562-row behavior six times:

```text
Recall@10/50/100/200/400: 0.6892 / 0.8189 / 0.8973 / 0.9324 / 0.9568
verified F1: 0.1946
verified gold citations: 906 = 151 x 6
terminal complete / refuse: 3,360 / 12 = (560 / 2) x 6
candidate depth: always 400
runtime / entrypoint trace violations: 0 / 0
cross-request contamination count: 0
```

Dev was loaded only after the train gate and run once as report-only evidence:

```text
end-to-end P95 / P99: 0.591977 / 0.695942s
Recall@10/50/100/200/400: 0.7237 / 0.8421 / 0.8684 / 0.9079 / 0.9211
verified F1 / gold citations: 0.1873 / 33
terminal complete / refuse: 121 / 0
candidate depth: always 400
runtime / entrypoint trace violations: 0 / 0
Stage143 behavior match: true
```

## Formal Decision

```text
status: primeqa_hybrid_concurrent_runtime_train_cv_dev_validation_passed
guard checks: 36 / 36 passed
failed latency scopes: []
profile B end-to-end SLO passed: true
overload admission contract validated: true
request-local state isolation validated: true
Stage143 behavior preserved: true
can wire explicit non-default concurrent runtime now: true
concurrent runtime registered for application use: false
concurrent runtime activation allowed now: false
runtime registered as default: false
runtime defaultization allowed now: false
test loaded / metrics run: false / false
queue / retry / fallback enabled: false / false / false
public forbidden keys: []
```

The final validator phases through guard evaluation completed naturally in
`489.895s`. This timing includes public source checks, one shared-resource
build, one warmup, the overload probe, six complete train passes, one gated dev
pass, summarization, and guards. JSON serialization and SVG generation happen
after that internal timer and are not included in `489.895s`.

## Visualizations

```text
stage145_train_pass_end_to_end_latency.svg
stage145_pattern_pooled_latency.svg
stage145_latency_scope_maxima.svg
stage145_dev_end_to_end_latency.svg
stage145_request_budget.svg
stage145_pass_throughput.svg
stage145_overload_outcome.svg
stage145_behavior_invariants.svg
stage145_decision_flags.svg
stage145_guard_check_status.svg
```

All ten SVG files parse successfully. The aggregate JSON and SVG files remain
local ignored artifacts.

## Next Step

Stage146 may wire the validated concurrency-four runtime behind a separate,
explicit, disabled-by-default application activation path. That stage must
validate disabled, rejected, and eligible startup states without rebuilding
heavy resources per request. It must not infer permission for defaultization,
test evaluation, queues, retries, fallback, or a network service from this
research-runtime result.
