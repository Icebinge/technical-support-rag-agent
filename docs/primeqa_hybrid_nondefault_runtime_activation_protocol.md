# PrimeQA Hybrid Non-Default Runtime Activation Protocol

## Scope

Stage141 freezes the user-confirmed strict-C latency SLO and an executable,
fail-closed policy for a future non-default sidecar-agent runtime. It reads only
the saved public-safe Stage140 aggregate report. It does not load split rows,
questions, documents, candidate pools, models, indexes, or test data.

The protocol is frozen, but runtime is not wired. The repository still has no
registered FastAPI or other service entrypoint, and `ProjectSettings` does not
yet expose the future activation field. The protocol names the future
environment interface without claiming that it already exists:

```text
TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT=false
```

An explicit `true` will eventually be required. The default remains disabled,
and even a policy result of `eligible` must not activate or register runtime by
itself.

## Confirmed Strict-C SLO

The user selected the strict profile:

```text
profile: strict_c_warm_single_request_v1
warm single-request P95: <= 0.300 seconds
warm single-request P99: <= 1.000 seconds
```

The scope is one warm candidate-pool retrieval request. Startup and refresh
costs are excluded from request latency and must be reported separately. This
protocol does not authorize concurrent serving. A later concurrent runtime
requires a separately confirmed concurrency SLO and protocol.

The percentile method remains the Stage140 linear interpolation at
`(n - 1) * p`. Stage142 must execute three complete warm train measurement
passes. The frozen five grouped train folds and the pooled train aggregate must
all meet both thresholds. Only after the train decision is fixed may dev run
once as a locked report-only gate. Test remains unloaded and unmeasured.

## Resource Ownership

The future process-scoped bootstrap owns these long-lived resources:

```text
2 dense models
2 dense embedding caches
4 lexical indexes
1 candidate-pool retriever
```

The special-token route is derived from the full-document BM25 results and
does not own another index. One deterministic train-only request, selected
without labels, warms all routes and is excluded from timing. That row remains
part of every complete measured train pass so the sample set is unchanged.

No model, cache, or lexical index may be loaded or built inside a measured
request. A measured request may only perform query encoding, long-lived index
search, frozen Top200/Top400 RRF fusion, deduplication, and candidate
materialization.

## Executable Policy

The Stage141 policy has three outcomes:

| State | Meaning |
| --- | --- |
| `disabled` | The explicit activation flag is absent or false. |
| `rejected` | Activation was requested but one or more guards failed. |
| `eligible` | Aggregate evidence satisfies every single-request guard; runtime is still not automatically activated. |

The guards require Stage140 source validation, warm resources, exact candidate
sequence identity, exact Stage127 recall preservation, five train folds, train
aggregate and dev report-only SLO compliance, and a locked test split. Any
failure rejects before request serving. There is no retry, fallback, silent
route substitution, or default-route change.

The public request trace contract permits only:

```text
runtime_mode
activation_requested
activation_state
slo_profile_id
warm_resources_ready
candidate_pool_depth
retrieval_latency_ms
latency_budget_passed
terminal_state
```

Question/document content, sample or document identifiers, and candidate rows
are forbidden.

## Current Evidence

Stage140 measured:

| Split | P95 | Strict P95 result | P99 evidence |
| --- | ---: | --- | --- |
| Train | 0.450798s | fail | unavailable |
| Dev | 0.293909s | pass | unavailable |

Therefore the current Stage140 evidence is `rejected`, not eligible. The
specific gaps are the train P95 violation, missing train/dev P99, absent
strict-fold validation, and resources not initialized by this aggregate-only
protocol run. Stage141 does not invent missing P99 values or reinterpret the
Stage140 maximum as P99.

## Real Stage141 Result

```text
status: primeqa_hybrid_nondefault_runtime_activation_protocol_frozen
guard checks: 19 / 19 passed
failed checks: []
strict SLO currently satisfied: false
runtime settings flag implemented: false
runtime entrypoint registered: false
runtime activation allowed now: false
runtime activated now: false
concurrent runtime activation allowed: false
runtime defaultization allowed now: false
test split loaded / metrics run: false / false
retry / fallback enabled: false / false
public-safe forbidden keys: []
```

Protocol success means the decision rules are frozen and internally
consistent. It does not mean latency passed or runtime was activated.

## Visualizations

```text
stage141_source_p95_vs_strict_slo.svg
stage141_percentile_evidence_availability.svg
stage141_activation_case_states.svg
stage141_runtime_permission_flags.svg
stage141_guard_check_status.svg
```

The JSON report and SVG files are local ignored artifacts and contain only
public-safe aggregate or synthetic policy data.

## Next Step

Stage142 is complete and recorded in:

```text
docs/primeqa_hybrid_strict_latency_validation.md
```

It replaced full eligible-row sorting with exact Top-K boundary selection,
compared optimized pools directly with a historical full-sort reference, and
passed three complete warm train runs plus one dev report-only run. Combined
train P95/P99 is `0.111715s / 0.322262s`; dev P95/P99 is
`0.094916s / 0.120182s`. Identity violations are zero and recall is unchanged.

Stage143 may now implement and validate the explicit non-default
single-request runtime wiring. Test stays locked. Concurrent serving,
defaultization, retries, and fallback strategies remain out of scope.
