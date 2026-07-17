# PrimeQA Hybrid Optional Sidecar Runtime Wiring Validation

## Scope

Stage143 implements and validates the explicit non-default single-request
runtime wiring authorized by Stage142. It adds the real project setting,
process-scoped bootstrap, long-lived retrieval and Agent resources, strict
activation policy integration, and the Stage141 public request trace.

This is not a web service and does not register a default runtime. Concurrent
serving, runtime defaultization, test evaluation, retries, and fallback
strategies remain closed.

## Runtime Setting

The setting is:

```text
ProjectSettings.enable_optional_sidecar_agent
TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT
default: false
accepted environment values: true / false
```

Values such as `1`, `yes`, or `on` are rejected. The default path does not
initialize models, caches, indexes, the candidate-pool retriever, or the
optional Agent entrypoint.

## Startup State Machine

Stage143 executes three startup cases:

| Case | Requested | State | Resources | Warmup | Active |
| --- | ---: | --- | ---: | ---: | ---: |
| Default disabled | no | `disabled` | no | no | no |
| Concurrent request | yes | `rejected` | no | no | no |
| Strict single request | yes | `eligible` | yes | one | yes |

The rejected case uses the real Stage142 evidence but requests concurrent
support. It stops before resource construction with
`concurrent_runtime_not_authorized_by_single_request_protocol`.

The eligible case requires all Stage142 SLO, identity, recall, five-fold, dev,
test-lock, and public-safety evidence. It then initializes resources once and
runs one deterministic train-only warmup question with labels removed. The
warmup completes the full optional entrypoint path and is excluded from the
measured train/dev request distributions.

## Process-Owned Resources

```text
dense models: 2
dense embedding caches: 2
lexical indexes: 4
derived special-token route: 1
candidate-pool retriever: 1
optional Agent entrypoint: 1
resources built or loaded per request: false
resource factory build count: 1
```

The runtime rejects a second bootstrap in the same process. It also rejects a
second simultaneous request instead of queueing, retrying, or substituting a
different route.

## Public Request Trace

Every successful request exposes exactly the Stage141 allowlisted fields:

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

`latency_budget_passed` is a request diagnostic comparing the current
retrieval latency with the strict P99 numeric limit. It does not replace the
aggregate P95/P99 validation. No question, answer, document, sample ID,
candidate row, content handle, label, or test membership is written.

## Train-First Validation

The final run order is:

1. Read public Stage141, Stage142, and final Stage139 regression reports.
2. Validate disabled and rejected startup without resource initialization.
3. Load train, build five grouped folds, initialize eligible resources once,
   and run one label-free warmup.
4. Run all 562 train rows through the actual runtime.
5. Require all train and fold trace, depth, recall, latency, and Stage139 Agent
   parity checks to pass.
6. Only then load 121 dev rows and run one report-only pass.

Test is not loaded or measured.

## Final Runtime Results

```text
status: primeqa_hybrid_optional_sidecar_runtime_wiring_validation_passed
guard checks: 28 / 28 passed
failed checks: []
disabled resource builds: 0
rejected resource builds: 0
eligible resource builds: 1
eligible warmup depth / latency: 400 / 64.084ms
train/dev runtime trace violations: 0 / 0
train/dev entrypoint trace violations: 0 / 0
train/dev exact five-transition trace rate: 1.0 / 1.0
train/dev candidate-pool depth: 400 / 400
test loaded / metrics run: false / false
retry / fallback: false / false
```

## Runtime Retrieval Latency

| Split | Average | P50 | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.061893s | 0.053957s | 0.104243s | 0.152497s | 0.371683s |
| Dev report-only | 0.059910s | 0.053435s | 0.094431s | 0.123178s | 0.295469s |

Every train fold also passed P95 <= `0.300s` and P99 <= `1.000s`. Fold P95
ranged from `0.094989s` to `0.108273s`; fold P99 ranged from `0.123359s` to
`0.331931s`.

The complete run took `126.371s`, including `45.049s` for process resource
initialization plus eligible warmup, `67.066s` for train, and `14.193s` for
dev. Startup cost is not repeated per request.

## Retrieval And Agent Parity

Recall exactly matches Stage142:

| Split | Recall@10 | Recall@50 | Recall@100 | Recall@200 | Recall@400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.6892 | 0.8189 | 0.8973 | 0.9324 | 0.9568 |
| Dev | 0.7237 | 0.8421 | 0.8684 | 0.9079 | 0.9211 |

Agent aggregates exactly match the final Stage139 regression:

```text
train/dev verified F1: 0.1946 / 0.1873
train/dev verified gold citations: 151 / 33
train terminal complete/refuse: 560 / 2
dev terminal complete/refuse: 121 / 0
```

Stage143 does not claim an answer-quality improvement. It validates that the
new runtime wiring preserves the already validated answer path.

## Decision Boundary

```text
optional runtime wiring implemented: true
optional runtime activation validated: true
single-request runtime validated: true
runtime registered as default: false
runtime defaultization allowed: false
concurrent runtime activation allowed: false
test gate opened: false
retry actions enabled: false
fallback strategies enabled: false
default runtime policy: unchanged
```

Activation is process-local and requires explicit `true` plus compliant source
evidence. The repository still has no FastAPI or other network service.

## Visualizations

```text
stage143_startup_states.svg
stage143_process_resource_inventory.svg
stage143_train_fold_latency.svg
stage143_split_latency_vs_slo.svg
stage143_recall_at_k.svg
stage143_decision_flags.svg
stage143_guard_check_status.svg
```

The JSON report and SVGs are local ignored aggregate-only artifacts.

## Next Step

Any Stage144 concurrency work requires a new user-confirmed protocol specifying
the workload model, concurrent request count, arrival pattern, hardware,
startup state, P95/P99 limits, rejection behavior, and resource safety rules.
The current single-request evidence cannot authorize concurrent serving.
Runtime defaultization and final test evaluation remain separate future gates.
