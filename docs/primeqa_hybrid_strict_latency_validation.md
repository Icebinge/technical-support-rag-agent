# PrimeQA Hybrid Strict Warm Latency Validation

## Scope

Stage142 optimizes and validates the Stage141 strict-C warm single-request
latency SLO without changing retrieval scores, tokenization, tie breaks, route
weights, the seven-channel graph, the immutable Top200 prefix, the Top400
append pool, or RRF. It uses train for optimization and selection. Dev is
loaded only after all train gates pass and is run once as report-only. Test is
not loaded or measured.

The confirmed limits are:

```text
warm single-request P95 <= 0.300 seconds
warm single-request P99 <= 1.000 seconds
```

Runtime wiring, concurrent-request validation, defaultization, retries, and
fallback strategies remain out of scope.

## Bottleneck And Optimization

Stage140 vectorized BM25 scoring but still fully sorted every eligible document
to return Top400. That left Python sorting on as many as 28,482 documents in
each document channel and all touched parent documents in the section channel.
Dense retrieval also fully sorted every score.

Stage142 replaces those full eligible-row sorts with an exact boundary method:

1. Use NumPy partition to find the Top-K score boundary.
2. Retain every row above or equal to that boundary, including all ties.
3. Apply the original score-descending and secondary tie break only to that
   reduced set.
4. Return the unchanged Top400 order.

BM25 length normalization is query-independent, so its `k1`-scaled values are
also precomputed during index construction. Query tokenization, term order,
duplicate-term accumulation, float precision, IDF, BM25 scores, dense scores,
and fusion are unchanged.

The train-only screening run measured:

```text
rows: 562
P50: 0.054075s
P95: 0.104895s
P99: 0.235705s
maximum: 0.393015s
dev loaded: false
test loaded: false
```

This passed the strict SLO without parallel channel execution. The selected
runtime shape therefore remains a sequential seven-channel graph. No thread
pool or request-internal parallelism was added.

## Historical Full-Sort Reference

Stage140 did not persist per-question candidate sequence hashes. Stage142 does
not pretend that a saved historical digest exists. Instead, it keeps the old
full eligible-row sort as a validation-only method on document BM25, section
BM25, mapped field-weighted BM25, and dense retrieval.

The final run built train and dev reference pools through that historical
full-sort path, then compared every optimized pool with the corresponding
reference sequence. The special-token route reused the full-sort baseline
result, matching the dependency-aware online graph. The reference construction
time is reported as validation/startup work and is excluded from request SLO
latency.

Unit tests also compare the boundary selector against full sort across random
tied scores and compare optimized versus reference results for document BM25,
section BM25, and dense retrieval.

## Train-First Protocol

The final validation order is enforced in code:

1. Load only the frozen train split and build five grouped folds.
2. Build long-lived models and indexes.
3. Build full-sort train reference pools.
4. Run one deterministic train-only warmup request, excluded from timing.
5. Run three complete 562-row warm train passes.
6. Require every pass, all five folds in every pass, the pooled 1,686-request
   distribution, and all pooled folds to meet both P95 and P99.
7. Only after that train gate passes, load dev, build its full-sort references,
   and run one 121-row report-only pass.

## Final Latency

| Scope | Average | P50 | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train pass 1 | 0.063797s | 0.054924s | 0.115416s | 0.319545s | 0.354894s |
| Train pass 2 | 0.063589s | 0.055092s | 0.107201s | 0.231092s | 0.401503s |
| Train pass 3 | 0.063618s | 0.053562s | 0.107752s | 0.327095s | 0.377087s |
| Combined train | 0.063668s | 0.054662s | 0.111715s | 0.322262s | 0.401503s |
| Dev report-only | 0.059702s | 0.053795s | 0.094916s | 0.120182s | 0.311350s |

Every pass and every fold passed. Combined train fold P95 ranged from
`0.096234s` to `0.120056s`; combined fold P99 ranged from `0.146212s` to
`0.337555s`.

Compared with Stage140, combined train P95 improved from `0.450798s` to
`0.111715s` (4.04x lower), and dev P95 improved from `0.293909s` to
`0.094916s` (3.10x lower). These ratios compare observed runs; they are not
claims about other hardware or concurrent workloads.

## Identity And Recall

```text
warmup full-sort identity violations: 0 / 1
three train passes identity violations: 0 / 1686
dev identity violations: 0 / 121
candidate-pool depth: 400
```

Recall counts exactly match Stage140 and the frozen Stage127 selected config in
every train pass and the dev pass:

| Split | Recall@10 | Recall@50 | Recall@100 | Recall@200 | Recall@400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.6892 | 0.8189 | 0.8973 | 0.9324 | 0.9568 |
| Dev | 0.7237 | 0.8421 | 0.8684 | 0.9079 | 0.9211 |

## Agent Regression

The final shared-retrieval implementation was followed by a complete Stage139
train/dev Agent regression:

```text
status: passed
guards: 45 / 45
train/dev candidate identity violations: 0 / 0
train/dev exact five-transition trace rate: 1.0 / 1.0
train/dev verified F1: 0.1946 / 0.1873
train/dev verified gold citations: 151 / 33
Stage137 aggregate parity: true
test loaded: false
retry/fallback: false / false
```

## Run History

- The first Stage139 regression attempt was blocked at preflight because its
  confirmation note said `Stage142` but the inherited guard requires the
  literal `Stage139`. It completed no Agent evaluation and reported 17/18
  preflight guards. The corrected run started from scratch and passed.
- An initial Stage142 formal run passed strict latency and same-run reference
  guards. Review then found that Stage140 had no persisted historical sequence
  digest, so the evidence was strengthened with a direct full-sort reference.
- The first strengthened run stopped after index build because the mapped
  field-weighted BM25 wrapper did not expose the new reference method. It did
  not complete train/dev measurement or write a new completed report.
- The wrapper was given an explicit reference delegation method. The final run
  restarted from model/index initialization and passed all guards.

## Final Result

```text
status: primeqa_hybrid_strict_warm_latency_validation_passed
guard checks: 25 / 25 passed
failed checks: []
strict SLO evidence state: eligible
can implement non-default runtime wiring now: true
runtime flag implemented: false
runtime entrypoint registered: false
runtime activated: false
concurrent activation allowed: false
runtime defaultization allowed: false
test loaded / metrics run: false / false
retry / fallback enabled: false / false
public forbidden keys: []
```

`eligible` means the frozen single-request performance and integrity evidence
is sufficient to implement the non-default runtime wiring. It does not mean
runtime is already wired, enabled, or activated.

## Visualizations

```text
stage142_train_pass_latency_vs_slo.svg
stage142_train_fold_worst_latency.svg
stage142_stage140_latency_comparison.svg
stage142_train_channel_p95_latency.svg
stage142_dev_latency_vs_slo.svg
stage142_decision_flags.svg
stage142_guard_check_status.svg
```

The JSON and SVG artifacts are local ignored outputs and contain aggregate
data only.

## Next Step

Stage143 should implement and validate the explicit non-default single-request
runtime wiring frozen in Stage141. It may expose the disabled-by-default
`TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT` setting, process-scoped bootstrap, and
public trace contract. It must test disabled, rejected, and eligible startup
behavior without opening concurrent serving, changing defaults, loading test,
or adding retries or fallback strategies.
