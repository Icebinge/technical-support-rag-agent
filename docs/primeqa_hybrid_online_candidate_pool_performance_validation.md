# PrimeQA Hybrid Online Candidate-Pool Performance Validation

Stage: Stage 140
Status: completed
Date: 2026-07-17

## Purpose

Stage139 validated the optional sidecar-agent entrypoint but spent 7600.407
seconds constructing candidate pools for 683 train/dev rows. That batch average
of 11.127975 seconds per row was not an acceptable online retrieval design.

Stage140 therefore postpones runtime activation and validates an exact-semantics
performance redesign. The stage must improve latency without changing any
candidate-pool document order or any frozen Recall@10/50/100/200/400 result.
Test remains locked and runtime defaults remain unchanged.

## Command

```text
python scripts/run_primeqa_hybrid_online_candidate_pool_performance_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage140 online candidate-pool performance diagnosis, vectorized retrieval optimization, full train five-fold grouped-CV aggregate validation and dev single-pass report-only validation; exact Stage127 recall and legacy candidate-pool identity required; prior completed run was correctly blocked because the online retriever fused Top400 directly instead of preserving the separately fused Stage116 Top200 prefix; this is a full clean rerun after restoring exact two-depth fusion semantics; test locked; runtime defaults unchanged; no retries or fallback strategies"
```

## Bottleneck Diagnosis

The first deterministic ten-row train probe measured the original seven frozen
routes separately:

| Route | P50 seconds | P95 seconds | Maximum seconds |
| --- | ---: | ---: | ---: |
| Full-document BM25 | 0.8595 | 2.7012 | 4.0109 |
| Section BM25 rollup | 5.8998 | 52.9634 | 89.6734 |
| Title/heading/body BM25 | 3.1801 | 4.8381 | 5.2615 |
| Title/heading-only BM25 | 0.3830 | 0.5339 | 0.5670 |
| Special-token BM25 | 2.0762 | 4.0376 | 4.9387 |
| E5-small dense | 0.5352 | 0.8913 | 0.9757 |
| MiniLM dense | 0.1729 | 0.5126 | 0.7543 |

The dominant cost was Python-loop lexical scoring, especially section BM25,
not dense query encoding. The Stage139 harness also evaluated four query-variant
routes that the selected Stage128 config did not consume, and the special-token
route repeated the full-document BM25 search.

## Implementation

- `BM25Retriever` now stores postings and lengths in NumPy arrays and applies
  the unchanged BM25 formula with vectorized scoring.
- `SectionBM25Retriever` vectorizes section scoring and uses parent-document
  `maximum.at` rollup while retaining score-descending/document-id tie breaks.
- Scalar-reference unit tests verify ranking and score equivalence, including
  repeated query terms and deterministic ties.
- `PrimeQAHybridOnlineCandidatePoolRetriever` owns no model or index bootstrap.
  It receives six initialized independent routes and one derived special-token
  route, so indexes and dense models live outside the request path.
- The derived special-token route reuses the full-document BM25 result instead
  of repeating the base search.
- Every request still builds its own query-specific candidate pool.
- The frozen two-depth semantics are explicit: each route is truncated to
  Top200 before Stage116 prefix RRF, while Top400 route results feed Stage128
  append RRF. The Top200 prefix is then preserved exactly.
- The online run records per-channel, fusion, materialization, and total latency
  without writing question text, sample IDs, document IDs, gold labels, or
  per-row traces.

NumPy is now an explicit project dependency because both lexical vectorization
and the existing dense retriever import it directly.

## Real Run History

The first complete calculation reached report assembly after 509.1 seconds but
failed because the split summarizer received one list instead of the full split
mapping. It wrote no Stage140 report and is not counted as a completed result.

The next complete run wrote a blocked report. Performance passed, but all 683
candidate-pool identity checks failed because the first online implementation
fused Top400 results directly. Train and dev shallow recall shifted while
Recall@400 stayed unchanged. Runtime activation remained closed.

The implementation was corrected to build the Top200 prefix and Top400 append
source separately. The final full run rebuilt every input and passed all guards.
No failed-run intermediate pool was reused.

## Final Latency

Long-lived index construction is a startup/refresh cost, not a request cost:

```text
load dense models and cached embeddings: 29.036 seconds
build long-lived lexical indexes: 100.816 seconds
```

Online per-query candidate-pool latency:

| Split | Rows | Average | P50 | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 562 | 0.296455s | 0.222661s | 0.450798s | 5.051437s |
| Dev | 121 | 0.234170s | 0.185714s | 0.293909s | 2.751323s |

The weighted train/dev mean is 0.285421 seconds. The complete online retriever
pass over 683 rows took 195.427 seconds, versus the Stage139 source candidate
pool time of 7600.407 seconds: a 38.89x batch speedup.

The historical Stage139 builder, which still evaluates all eleven routes and
does not use the dependency-aware online graph, was rerun after vectorization.
Its candidate construction time fell to 682.299 seconds and its total time fell
from 7834.841 to 959.343 seconds. All 45 Stage139 guards still passed.

## Recall And Identity

Every Stage140 pool contains 400 documents. Exact sequence comparison against
the optimized legacy builder covered all 683 rows:

```text
train identity violations: 0 / 562
dev identity violations: 0 / 121
```

Final recall exactly matches the frozen Stage127 selected config:

| Split | Recall@10 | Recall@50 | Recall@100 | Recall@200 | Recall@400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.6892 | 0.8189 | 0.8973 | 0.9324 | 0.9568 |
| Dev | 0.7237 | 0.8421 | 0.8684 | 0.9079 | 0.9211 |

Train uses the existing five grouped folds for aggregate latency and recall
reporting. Dev is one report-only pass and is not used for selection or tuning.

## Agent Regression

The complete Stage139 entrypoint validation was rerun to a separate ignored
artifact after the infrastructure change:

```text
guard checks: 45 / 45 passed
candidate-pool identity violations: 0
Stage137 aggregate parity: true
train/dev exact five-transition trace rate: 1.0000 / 1.0000
train/dev verified F1: 0.1946 / 0.1873
train/dev verified gold citations: 151 / 33
retry actions: 0
fallback actions: 0
test split loaded: false
```

## Visualizations

```text
stage140_candidate_pool_wall_time.svg
stage140_online_latency_distribution.svg
stage140_train_channel_p95_latency.svg
stage140_recall_at_k.svg
stage140_guard_check_status.svg
```

The JSON report and SVGs are local ignored artifacts and contain aggregate data
only.

## Decision

Stage140 passes the exact candidate identity, Stage127 recall, performance,
public-safety, train/dev-only, test-lock, retry, fallback, and runtime-default
guards. The online candidate-pool core is validated, but it is not activated or
registered as a runtime default.

No user-confirmed production latency SLO currently exists. Stage140 therefore
reports the observed distribution but does not invent a product SLO or use it
to authorize runtime activation.

## Next Step

Stage141 should freeze a user-confirmed latency SLO and explicit non-default
runtime activation protocol. It should define startup ownership of the two
dense models, cached embeddings, and four lexical indexes; the disabled-by-
default runtime flag; one-request timing and public trace fields; concurrency
and warmup validation; refusal behavior; and activation guards. Test remains
locked. Runtime defaultization, retries, and fallback strategies remain
disabled.
