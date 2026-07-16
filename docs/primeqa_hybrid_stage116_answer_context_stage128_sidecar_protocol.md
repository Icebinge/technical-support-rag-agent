---
Stage: Stage 134
Title: PrimeQA hybrid Stage116 answer-context plus Stage128 sidecar-observation agent protocol
Status: frozen
---

# PrimeQA Hybrid Stage116 Answer-Context Plus Stage128 Sidecar-Observation Agent Protocol

Stage134 freezes a train/dev-only agent protocol that keeps the primary answer
context on the stable Stage116 behavior while exposing Stage128/Stage132
append candidates only as a sidecar observation channel.

This stage reads only saved public-safe aggregate reports from Stage128,
Stage129, and Stage133. It does not load split files, corpus documents, raw
candidate rows, raw questions, raw answers, raw document identifiers, or test
data. It does not run retrieval, answering, validation metrics, final metrics,
runtime defaultization, or fallback strategies.

## Command

```text
python scripts\freeze_primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage134 Stage116 answer-context plus Stage128 sidecar-observation agent protocol freeze after Stage133 selected sidecar review; public-safe aggregate only; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source Boundary

Stage128 source contract:

```text
source status: primeqa_hybrid_agent_retrieval_integration_protocol_frozen
source protocol id: primeqa_hybrid_agent_retrieval_integration_protocol_v1
selected candidate-pool config: prefix_existing_dense_broad_append200_v1
candidate pool output depth: 400
candidate pool is not automatic answer context: true
rank region 1: Stage116 immutable prefix, ranks 1-200
rank region 2: Stage128 append expansion, ranks 201-400
source guard checks: 17 / 17 passed
```

Stage129 risk carried forward:

```text
source status: primeqa_hybrid_agent_retrieval_integration_validation_blocked_or_failed
failed guard: gold_citation_count_delta_vs_stage116_non_negative
train F1 delta vs Stage116: +0.0003
train gold citation count delta vs Stage116: -1
train target-depth hit delta vs Stage116: +9
train changed verified answers vs Stage116: 221
dev changed verified answers vs Stage116: 50
source guard checks: 20 / 21 passed
```

Stage133 selected sidecar review:

```text
source status: primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review_completed
source review id: primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review_v1
selected config: prefix10_append_sidecar_probe_v1
selected profile: stage132_prefix10_append_sidecar_probe_v1
classification: safe_but_neutral_sidecar
source guard checks: 14 / 14 passed
```

## Frozen Protocol

```text
protocol id: primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_v1
route name: stage116_answer_context_plus_stage128_sidecar_observation
primary answer context source: Stage116 top200 evidence shortlist behavior
sidecar observation source: Stage128 top400 candidate pool via Stage132 sidecar config
```

Primary answer-context channel:

```text
channel: stage116_primary_answer_context
answer context depth: 10
rank source: Stage116 immutable prefix ranks 1-200
sidecar candidates included: false
may be reordered by sidecar: false
may be replaced by sidecar: false
allowed to generate answer text: true
```

Sidecar observation channel:

```text
channel: stage128_stage132_sidecar_observation
candidate pool depth: 400
source candidate-pool config: prefix_existing_dense_broad_append200_v1
selected sidecar config: prefix10_append_sidecar_probe_v1
selected sidecar profile: stage132_prefix10_append_sidecar_probe_v1
append region ranks: 201-400
append budget: 200
observation slots: 3
allowed to generate answer text: false
allowed to replace primary context: false
allowed to support agent observation: true
allowed to support future citation verification: true
```

## Agent Observation Interface

Primary context fields to validate:

```text
runtime_content_handle
primary_context_rank
primary_context_source_region
retrieval_score_summary
```

Sidecar observation fields to validate:

```text
runtime_content_handle
sidecar_observation_rank
sidecar_source_region
sidecar_route_family
sidecar_score_summary
citation_verification_signal
```

Forbidden runtime signals:

```text
test membership
gold labels
answer document labels
source-provided candidate labels
dev-selected thresholds
raw private rows in public artifacts
```

## Agent Consumer Policy

Allowed only after Stage135 validation:

```text
sidecar_observation_rendering
citation_verification_probe
evidence_gap_explanation
```

Blocked:

```text
sidecar_answer_text_generation
sidecar_primary_context_replacement
direct_stage128_all400_answer_context
runtime_default_retrieval_route
fallback_strategy_route
```

## Stage135 Validation Plan

```text
next stage: Stage135
action: run_stage116_answer_context_stage128_sidecar_observation_train_cv_dev_validation
selection split: train
selection mode: train_grouped_cross_validation_sidecar_observation_integrity
minimum train folds: 5
validation split: dev
dev mode: single_pass_report_only_no_retuning
```

Primary checks:

```text
Stage116 primary answer context remains byte-for-byte unchanged at the policy level
sidecar observation records are isolated from answer-text generation
sidecar observation records are isolated from prefix replacement
direct Stage128 all-400 answer context remains blocked
test split remains unloaded
```

Expected metrics:

```text
primary answer-context identity status
sidecar observation availability count
sidecar citation-verification signal coverage
answer F1 delta expected to remain zero
gold citation delta expected to remain zero
changed answer count expected to remain zero
```

Rules:

```text
test access allowed: false
final test metrics allowed: false
test tuning allowed: false
default runtime policy: unchanged
runtime defaultization allowed in Stage134: false
fallback strategies enabled: false
```

## Guard Checks

```text
guard checks: 19 / 19 passed
test split loaded: false
final test metrics run: false
runtime defaultization allowed now: false
fallback strategies enabled: false
default runtime policy: unchanged
public_safe_contract.forbidden_keys_found: []
```

## Visualizations

```text
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_stage134_visuals\stage134_protocol_components.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_stage134_visuals\stage134_sidecar_train_dev_signals.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_stage134_visuals\stage134_channel_permission_flags.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_stage134_visuals\stage134_risk_boundary_flags.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_stage134_visuals\stage134_protocol_decision_flags.svg
artifacts\primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol_stage134_visuals\stage134_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_stage116_answer_context_stage128_sidecar_agent_protocol_frozen
recommended_next_direction: run_stage116_answer_context_stage128_sidecar_observation_train_cv_dev_validation
can_continue_agent_design_implementation: true
can_build_sidecar_observation_adapter_now: true
sidecar_can_generate_answer_text: false
sidecar_can_replace_primary_context: false
direct_stage128_all400_answer_context_remains_blocked: true
replacement_append_answer_context_route_remains_stopped: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
runtime_defaultization_allowed_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Follow-Up

Stage135 completed the train grouped-CV plus dev report-only adapter validation
in:

```text
docs/primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation.md
```

The two-channel interface passed all integrity and isolation checks, but the
current three-slot sidecar captured none of the 9 train and 1 dev incremental
gold opportunities in the Stage128 append region. The interface can move into
train/dev agent orchestration, while citation-verification effectiveness remains
unproven. Test remains locked, runtime defaults remain unchanged, and fallback
strategies remain disabled.
