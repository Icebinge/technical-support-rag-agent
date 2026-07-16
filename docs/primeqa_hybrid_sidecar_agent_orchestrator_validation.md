---
Stage: Stage 137
Title: PrimeQA hybrid sidecar agent orchestrator train-CV/dev validation
Status: completed
---

# PrimeQA Hybrid Sidecar Agent Orchestrator Train-CV/Dev Validation

Stage137 executes the fixed Stage136 orchestrator on all frozen train/dev rows.
For every row, it independently runs a Stage116 control and the Stage136 agent,
records the exact generator and verifier inputs in memory, and compares answer,
verification, trace, and sidecar-isolation behavior.

The integration passed. The sidecar remains diagnostically available but
effectively neutral: it captured none of the known append-region answer-document
opportunities and did not change any answer or metric.

## Command

```text
python scripts/run_primeqa_hybrid_sidecar_agent_orchestrator_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage137 real fixed-orchestrator train five-fold grouped-CV and dev single-pass report-only validation after Stage136 protocol freeze; no candidate selection; no dev retuning; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

The process was allowed to complete naturally. It was not sampled, restarted,
or stopped early.

## Data Boundary

```text
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
train folds: 5
candidate selection performed: false
threshold tuning performed: false
dev used for selection: false
dev used for retuning: false
test split loaded: false
final test metrics run: false
```

Stage137 loads only the frozen train/dev splits, local training/dev corpus,
existing local dense caches, and public-safe Stage125/128/135/136 artifacts.
Private control/agent executions, document handles, answers, and per-row trace
objects exist only in memory. The saved report contains aggregate summaries.

## Validation Harness

The Stage137 harness wraps the real answer generator and answer verifier with
recording decorators. It therefore checks the actual dependency inputs rather
than trusting only the orchestrator's declared trace flags.

For every row it compares:

```text
Stage116 answer-context records vs actual agent generator input
Stage116 prefix ranks 1-200 vs actual agent verifier input
adapter bundle answer context vs actual agent generator input
control original answer vs agent original answer
control verified answer vs agent verified answer
control verification reasons vs agent verification reasons
sidecar handles vs generator, verifier, and primary-context handles
serialized public trace vs its frozen field/permission contract
```

Gold answer-document labels are read only after online orchestration for
aggregate train/dev opportunity and capture diagnostics. They do not enter
sidecar scoring, answer generation, verification, or runtime trace fields.

## Candidate Pools

```text
train prefix identity violations: 0
dev prefix identity violations: 0
train append budget exceeded: 0
dev append budget exceeded: 0
train append count per row: 200
dev append count per row: 200
```

The Stage116 prefix remains immutable and Stage128 adds exactly 200 append
candidates per row.

## Train Grouped-CV

| Fold | Rows | Generation context violations | Verification context violations | Original answer changes | Verified answer changes | Sidecar answer-path leaks | Public trace violations | Append opportunities | Sidecar captures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold_1 | 113 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| fold_2 | 113 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| fold_3 | 112 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| fold_4 | 112 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fold_5 | 112 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |

All nine frozen train-CV integrity checks passed.

## Control Versus Agent

Train verified results:

```text
Stage116 control average token F1: 0.1946
Stage137 agent average token F1: 0.1946
F1 delta: +0.0000
Stage116 control gold citation count: 151
Stage137 agent gold citation count: 151
gold citation count delta: +0
changed original answers: 0 / 562
changed verified answers: 0 / 562
changed verification reasons: 0 / 562
```

Dev report-only results:

```text
Stage116 control average token F1: 0.1873
Stage137 agent average token F1: 0.1873
F1 delta: +0.0000
Stage116 control gold citation count: 33
Stage137 agent gold citation count: 33
gold citation count delta: +0
changed original answers: 0 / 121
changed verified answers: 0 / 121
changed verification reasons: 0 / 121
```

All original and verified citation/refusal metric deltas are zero on both
splits. This proves answer-path invariance, not answer-quality improvement.

## Saved-Control Audit

After the Stage137 run completed, the Stage137 control aggregate was compared
with the saved public-safe Stage132 `stage116_top200_agent_pool_control`
profile:

```text
train verified F1: 0.1946 vs 0.1946
train gold citation count: 151 vs 151
dev verified F1: 0.1873 vs 0.1873
dev gold citation count: 33 vs 33
```

This was a post-run public-aggregate audit, not an additional Stage137 guard and
not a rerun. It confirms that the independently executed Stage137 control did
not drift from the previously saved Stage116 control metrics.

## Sidecar Diagnostics

```text
train rows with sidecar observations: 562 / 562
train sidecar observations: 1686
train query-overlap row coverage: 1.0000
train novel-query row coverage: 0.3897
train append opportunities: 9
train sidecar captures: 0

dev rows with sidecar observations: 121 / 121
dev sidecar observations: 363
dev query-overlap row coverage: 1.0000
dev novel-query row coverage: 0.4298
dev append opportunities: 1
dev sidecar captures: 0
```

Train public evidence-gap statuses:

```text
novel_query_coverage_observed_not_answer_evidence: 219
sidecar_observed_without_novel_query_coverage: 341
verified_answer_refused_sidecar_diagnostic_only: 2
```

Dev public evidence-gap statuses:

```text
novel_query_coverage_observed_not_answer_evidence: 52
sidecar_observed_without_novel_query_coverage: 69
```

The Stage135 negative boundary is reproduced exactly: train `0/9` and dev
`0/1`. Query-overlap and novel-query signals are observable diagnostics, but
the current three-slot selector has not demonstrated citation recovery.

## Isolation And Public Safety

Across all 683 rows:

```text
generation context identity violations: 0
verification context identity violations: 0
bundle/generator context identity violations: 0
sidecar generation leaks: 0
sidecar verification leaks: 0
sidecar/primary overlaps: 0
public trace serialization violations: 0
public trace forbidden keys: 0
public trace contract violations: 0
```

The public report does not contain raw question text, answer text, document
text, document identifiers, runtime handles, candidate rows, sample IDs, gold
labels, test membership, or per-row train-CV group values.

## Guard Checks

```text
status: primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_passed
guard checks: 36 / 36 passed
failed checks: []
agent orchestrator integration validated: true
sidecar effectiveness status: safe_but_neutral
sidecar citation-verification effectiveness demonstrated: false
can claim answer-quality improvement: false
can claim retrieval improvement: false
can freeze optional agent entrypoint protocol now: true
can open final test gate now: false
can run final test metrics now: false
can use test for tuning: false
runtime defaultization allowed now: false
fallback strategies enabled: false
default runtime policy: unchanged
```

## Timing

```text
load protocols: 0.020 seconds
load splits and folds: 0.046 seconds
load documents and sections: 3.145 seconds
dense preflight: 20.526 seconds
build indexes: 41.605 seconds
build candidate pools: 1516.081 seconds
run control and agent traces: 19.767 seconds
summarize and guard: 0.112 seconds
total: 1601.302 seconds
```

Candidate-pool construction remains the dominant engineering cost. This timing
is not an algorithm-quality result.

## Visualizations

```text
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_validation_stage137_visuals\stage137_train_fold_identity_violations.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_validation_stage137_visuals\stage137_split_answer_metric_deltas.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_validation_stage137_visuals\stage137_sidecar_isolation_violations.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_validation_stage137_visuals\stage137_sidecar_opportunity_capture.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_validation_stage137_visuals\stage137_sidecar_signal_coverage.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_validation_stage137_visuals\stage137_decision_flags.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_validation_stage137_visuals\stage137_guard_check_status.svg
```

## Decision

The Stage136 orchestrator is safe to expose behind a future optional agent
entrypoint protocol: its real dependency inputs and answers are identical to
Stage116, its sidecar remains isolated, and its public trace is safe.

This does not authorize runtime defaultization. The current sidecar adds
diagnostic visibility but no demonstrated citation-recovery or answer-quality
benefit.

## Next Step

Stage138 should freeze an optional sidecar-agent entrypoint and action-state
protocol. It should define explicit retrieve, answer, verify, observe, and
refuse states around the validated Stage137 orchestrator while keeping the
existing runtime default unchanged. Test remains locked and fallback strategies
remain disabled.
