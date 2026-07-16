---
Stage: Stage 136
Title: PrimeQA hybrid sidecar agent orchestrator and public-safe trace protocol
Status: completed
---

# PrimeQA Hybrid Sidecar Agent Orchestrator And Public-Safe Trace Protocol

Stage136 implements the first executable agent orchestrator around the frozen
Stage134/135 two-channel contract. It keeps the Stage116 answer path unchanged
and exposes Stage128/Stage132 sidecar observations only as diagnostic metadata.

This stage freezes implementation and validation boundaries. It does not run
the orchestrator over train or dev rows, and it does not demonstrate citation,
answer-quality, or retrieval improvement.

## Command

```text
python scripts\freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage136 sidecar agent orchestrator implementation and public-safe trace protocol freeze after Stage135 validation; public-safe aggregate only; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies; citation-verification effectiveness unproven"
```

## Source Boundary

Stage136 reads only the saved public-safe Stage135 aggregate report:

```text
Stage135 status: primeqa_hybrid_sidecar_observation_validation_passed
Stage135 guard checks: 30 / 30 passed
train rows represented by source aggregate: 562
dev rows represented by source aggregate: 121
train append gold opportunities / sidecar captures: 9 / 0
dev append gold opportunities / sidecar captures: 1 / 0
split files loaded by Stage136: false
corpus documents loaded by Stage136: false
test split loaded: false
final test metrics run: false
```

The train/dev counts above are inherited aggregate facts from Stage135. Stage136
does not reopen the rows, documents, candidate pools, answers, or gold labels.

## Orchestrator Contract

```text
orchestrator id: stage116_primary_plus_stage128_sidecar_agent_orchestrator_v1
answer generation channel: stage116_primary_answer_context
answer generation depth: 10
answer verification channel: stage116_prefix_verification_context
answer verification maximum rank: 200
sidecar observation channel: stage128_stage132_sidecar_observation
sidecar candidate-pool maximum rank: 400
sidecar observation slots: 3
sidecar can generate answer text: false
sidecar can enter answer verification context: false
sidecar can replace primary context: false
fallback strategy allowed: false
```

The orchestrator builds the Stage135 observation bundle first, sends only
`answer_context_for_generation()` to the answer generator, and sends only the
Stage116 prefix records at ranks 1-200 to the verifier. Sidecar observations
are copied into the public-safe diagnostic trace after answer generation and
verification; they are never supplied to either answer path.

The implementation rejects duplicate ranks, non-positive ranks, and candidate
ranks above the frozen top400 pool boundary.

## Public-Safe Trace

`PublicSafeSidecarAgentTrace` records only aggregate and numeric runtime
metadata needed to inspect channel routing and sidecar selection:

```text
primary candidate count and rank list
verification candidate count and maximum rank
sidecar observation rank and source region
query-overlap and retrieval-prior scores
novel-query coverage indicators
selected-for-generation flag: always false
selected-for-verification flag: always false
answer and verification refusal flags
verification reason and checked-candidate count
```

It does not serialize raw question text, answer text, document text, document
identifiers, runtime content handles, sample identifiers, candidate rows, gold
labels, or test membership. Runtime traces are gold-free, so they can explain
why a sidecar record was selected and prove channel isolation, but cannot label
an answer-document miss. Offline Stage137 train/dev diagnostics remain required
for that interpretation.

## Frozen Validation Plan

Stage137 is frozen as a real train/dev validation rather than another design
step:

```text
selection split: train
selection mode: fixed-orchestrator grouped-CV integrity validation
minimum train folds: 5
candidate selection performed: false
threshold tuning performed: false
validation split: dev
dev mode: single pass report-only, no retuning
test split: locked and unloaded
```

Stage137 must compare the orchestrator with the Stage116 control and verify:

1. Generated and verified answers remain identical to Stage116.
2. Sidecar observations never enter generation or verification context.
3. Public-safe traces serialize without private fields.
4. Selection and offline miss diagnostics preserve the Stage135 effectiveness
   boundary instead of turning signal availability into a quality claim.

## Guard Checks

```text
status: primeqa_hybrid_sidecar_agent_orchestrator_protocol_frozen
guard checks: 21 / 21 passed
failed checks: []
public forbidden keys found: []
can run Stage137 train/dev validation now: true
can claim citation-verification effectiveness: false
can claim answer-quality improvement: false
can claim retrieval improvement: false
can open final test gate now: false
can run final test metrics now: false
can use test for tuning: false
runtime defaultization allowed now: false
fallback strategies enabled: false
default runtime policy: unchanged
```

The guards fail closed when a zero-valued Stage135 safety or capture field is
missing. A missing violation count therefore cannot be silently interpreted as
zero.

## Visualizations

```text
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_protocol_stage136_visuals\stage136_stage135_safety_counts.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_protocol_stage136_visuals\stage136_sidecar_opportunity_capture.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_protocol_stage136_visuals\stage136_channel_permission_flags.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_protocol_stage136_visuals\stage136_decision_flags.svg
artifacts\primeqa_hybrid_sidecar_agent_orchestrator_protocol_stage136_visuals\stage136_guard_check_status.svg
```

## Decision

The orchestrator and public-safe trace contract are implemented and frozen.
The code can proceed to Stage137 train grouped-CV plus dev report-only
validation. This is an implementation-readiness decision, not an effectiveness
decision: the current sidecar still captured none of Stage135's 9 train and 1
dev append-region opportunities.

## Next Step

Stage137 should run the fixed Stage116-primary plus Stage128-sidecar agent
orchestrator over train five-fold grouped cross-validation and one dev
report-only pass. It must not tune on dev, load test, change runtime defaults,
enable fallback strategies, or claim sidecar effectiveness without measured
evidence.
