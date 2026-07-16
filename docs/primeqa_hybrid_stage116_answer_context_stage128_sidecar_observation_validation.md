---
Stage: Stage 135
Title: PrimeQA hybrid Stage116 answer-context plus Stage128 sidecar-observation validation
Status: completed
---

# PrimeQA Hybrid Stage116 Answer-Context Plus Stage128 Sidecar-Observation Validation

Stage135 implements the frozen Stage134 sidecar adapter and validates it on
train grouped cross-validation plus dev report-only.

The result has two separate conclusions:

1. The two-channel interface is safe and structurally valid. Stage116 remains
   the only answer-context source, while Stage128 append candidates remain
   isolated metadata-only sidecar observations.
2. The current three-slot query-overlap sidecar has not demonstrated evidence
   recovery value. It captured none of the known train/dev incremental gold
   opportunities in the Stage128 append region.

Stage135 does not load test, run final test metrics, change runtime defaults,
enable fallback strategies, or permit sidecar answer generation or prefix
replacement.

## Command

```text
python scripts\run_primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage135 real train grouped-CV and dev report-only sidecar observation validation after Stage134 protocol freeze; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Data Boundary

```text
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
train folds: 5
test split loaded: false
final test metrics run: false
dev used for selection: false
dev used for retuning: false
candidate selection performed: false
threshold tuning performed: false
```

Runtime content handles, questions, documents, candidates, and gold labels
were used only in memory. The public artifact contains aggregate summaries and
does not contain raw observation records, sample identifiers, document
identifiers, question text, answer text, document text, or candidate rows.

## Adapter Contract

```text
adapter id: stage116_primary_plus_stage128_sidecar_observation_adapter_v1
scoring policy: runtime_visible_query_overlap_plus_retrieval_prior_v1
primary context depth: 10
primary source: Stage116 immutable prefix ranks 1-200
sidecar observation slots: 3
sidecar source: Stage128 append ranks 201-400
answer generator receives primary channel only: true
sidecar contains document text: false
sidecar contains gold labels: false
sidecar can generate answer text: false
sidecar can replace primary context: false
citation-verification signal thresholded: false
```

The online sidecar signal uses only runtime-visible query/document token
overlap and retrieval rank. Gold labels are used only after observation
construction to calculate aggregate train/dev diagnostic counts.

## Train Grouped-CV

All five train folds preserved the primary context and sidecar isolation:

| Fold | Rows | Primary identity violations | Sidecar leaks | Observation availability | Query-overlap coverage | Novel-query coverage | Append gold opportunities | Sidecar captures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold_1 | 113 | 0 | 0 | 1.0000 | 1.0000 | 0.3540 | 2 | 0 |
| fold_2 | 113 | 0 | 0 | 1.0000 | 1.0000 | 0.3628 | 3 | 0 |
| fold_3 | 112 | 0 | 0 | 1.0000 | 1.0000 | 0.3304 | 2 | 0 |
| fold_4 | 112 | 0 | 0 | 1.0000 | 1.0000 | 0.4821 | 0 | 0 |
| fold_5 | 112 | 0 | 0 | 1.0000 | 1.0000 | 0.4196 | 2 | 0 |

Train aggregate:

```text
rows with full three-slot observations: 562 / 562
sidecar observation count: 1686
primary-context identity violations: 0
answer-generation context identity violations: 0
primary record field violations: 0
sidecar record field violations: 0
sidecar rank-region violations: 0
sidecar answer-context leaks: 0
sidecar/primary overlaps: 0
sidecar slot overflows: 0
query-overlap signal rate: 1.0000
rows with novel-query coverage signal: 219 / 562 (0.3897)
append-pool incremental gold opportunities: 9
sidecar incremental gold captures: 0
sidecar capture rate: 0.0000
```

## Dev Report-Only

```text
rows with full three-slot observations: 121 / 121
sidecar observation count: 363
primary-context identity violations: 0
answer-generation context identity violations: 0
primary record field violations: 0
sidecar record field violations: 0
sidecar rank-region violations: 0
sidecar answer-context leaks: 0
sidecar/primary overlaps: 0
sidecar slot overflows: 0
query-overlap signal rate: 1.0000
rows with novel-query coverage signal: 52 / 121 (0.4298)
append-pool incremental gold opportunities: 1
sidecar incremental gold captures: 0
sidecar capture rate: 0.0000
```

Dev is reported once and was not used to select, retune, or threshold the
adapter.

## Answer Invariance

Stage135 independently found zero primary-context identity violations on all
683 train/dev rows. The saved Stage132 measured answer results also remain
unchanged for the selected sidecar profile:

```text
train verified F1 delta vs Stage116: +0.0000
train gold citation count delta vs Stage116: +0
train changed verified answers: 0

dev verified F1 delta vs Stage116: +0.0000
dev gold citation count delta vs Stage116: +0
dev changed verified answers: 0
```

These are safety/invariance results, not answer-quality gains.

## Interpretation

The Stage135 adapter solves the channel-boundary problem:

- The answer generator receives exactly the Stage116 primary context.
- Sidecar records are complete, available, append-only, and isolated.
- The online sidecar signal does not use gold labels or dev-tuned thresholds.
- The interface is ready for a train/dev agent orchestrator and public-safe
  trace contract.

The adapter does not yet solve evidence recovery:

- Stage128 exposed 9 train and 1 dev incremental gold opportunities in ranks
  201-400.
- The current three-slot query-overlap sidecar selected 0 of those 10
  opportunities.
- Query-overlap signal availability at 100% therefore means that the signal is
  populated, not that it identifies the missing answer document.
- Citation-verification effectiveness remains unproven and must not be claimed
  as validated answer quality or retrieval improvement.

## Guard Checks

```text
guard checks: 30 / 30 passed
test split loaded: false
final test metrics run: false
runtime observation records written: false
runtime defaultization allowed now: false
fallback strategies enabled: false
default runtime policy: unchanged
public_safe_contract.forbidden_keys_found: []
```

## Timing

```text
load protocols: 0.010 seconds
load splits and build train folds: 0.046 seconds
load documents and sections: 4.219 seconds
dense preflight: 21.653 seconds
build indexes: 63.703 seconds
build candidate pools: 2867.530 seconds
build and validate observations: 20.423 seconds
summarize and guard: 0.082 seconds
total: 2977.666 seconds
```

Candidate-pool construction dominates the runtime. Stage135 records this as an
engineering cost; it is not interpreted as an algorithm-quality result.

## Visualizations

```text
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_stage135_visuals\stage135_split_sidecar_signal_coverage.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_stage135_visuals\stage135_train_fold_signal_coverage.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_stage135_visuals\stage135_gold_observation_opportunities.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_stage135_visuals\stage135_isolation_violation_counts.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_stage135_visuals\stage135_decision_flags.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_stage135_visuals\stage135_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_sidecar_observation_validation_passed
sidecar observation protocol validated: true
can implement train/dev agent orchestrator now: true
validated interface consumers:
  sidecar_observation_rendering
  citation_verification_probe
  evidence_gap_explanation
direct Stage128 all-400 answer context remains blocked: true
sidecar can generate answer text: false
sidecar can replace primary context: false
can open final test gate now: false
can run final test metrics now: false
can use test for tuning: false
runtime defaultization allowed now: false
fallback strategies enabled: false
default runtime policy: unchanged
```

The artifact's validated consumers refer to interface availability and
isolation. They do not mean that citation-verification effectiveness has been
demonstrated. The derived Stage135 interpretation from the observed 0/9 train
and 0/1 dev opportunity capture is:

```text
current sidecar evidence-recovery effectiveness demonstrated: false
```

## Next Step

Stage136 should implement the train/dev-only Stage116-primary plus
Stage128-sidecar agent orchestrator and public-safe trace contract. It should
render the sidecar as observation metadata and preserve the current answer
path exactly. Because the current three-slot selector captured none of the
known append-region gold opportunities, Stage136 must keep citation
verification in diagnostic status and include explicit trace evidence for
sidecar selection/miss analysis. Test remains locked, runtime defaults remain
unchanged, and fallback strategies remain disabled.
