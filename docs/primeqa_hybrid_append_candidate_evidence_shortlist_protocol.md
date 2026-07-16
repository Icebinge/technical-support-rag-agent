---
Stage: Stage 131
Title: PrimeQA hybrid append-candidate evidence shortlist redesign protocol
Status: completed
---

# PrimeQA Hybrid Append-Candidate Evidence Shortlist Redesign Protocol

Stage131 freezes a train/dev-only protocol for redesigning the evidence
shortlist after Stage130 showed that the Stage128 top400 candidate pool improves
recall but is not citation-safe as direct answer context.

This stage reads only the public-safe Stage130 aggregate failure review. It
does not load split files, corpus documents, raw candidate rows, raw questions,
raw answers, raw document identifiers, or test data. It does not run retrieval,
answering, validation metrics, final metrics, runtime defaultization, or
fallback strategies.

## Command

```text
python scripts\freeze_primeqa_hybrid_append_candidate_evidence_shortlist_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage131 append-candidate evidence shortlist redesign protocol freeze after Stage130 failure review; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source

```text
source stage: Stage 130
source review id: primeqa_hybrid_stage129_agent_integration_failure_review_v1
source status: primeqa_hybrid_stage129_agent_integration_failure_review_completed
source next direction: freeze_append_candidate_evidence_shortlist_redesign_protocol
stage128 direct agent integration path blocked: true
source guard checks: 12 / 12 passed
```

## Source Failure Pressure

```text
train gold hit delta: +9
train gold citation delta: -1
train verified F1 delta: +0.0003
train changed answer rate: 0.3932
train append selected citations: 42
train prefix-like selected citation delta: -42

dev gold hit delta: +1
dev gold citation delta: -2
dev verified F1 delta: -0.0036
dev changed answer rate: 0.4132
dev append selected citations: 12
dev prefix-like selected citation delta: -12
```

## Frozen Protocol

```text
protocol_id: primeqa_hybrid_append_candidate_evidence_shortlist_redesign_protocol_v1
route_name: append_candidate_evidence_shortlist_redesign
source candidate pool: Stage116 prefix ranks 1-200 + Stage128 append ranks 201-400
candidate pool role: recall_pool_not_unrestricted_answer_context
Stage116 prefix must remain available: true
append candidates are supplemental: true
```

Frozen candidate shortlist configs:

```text
prefix10_append_sidecar_probe_v1:
  role: conservative_control
  answer context depth: 10
  protected prefix slots: 10
  replacement append slots: 0
  append sidecar slots: 3
  append sidecar can generate answer text: false
  append sidecar can support citation verification: true

prefix9_append1_high_precision_v1:
  role: high_precision_single_append
  answer context depth: 10
  protected prefix slots: 9
  replacement append slots: 1
  append sidecar slots: 2
  append sidecar can generate answer text: false
  append sidecar can support citation verification: true

prefix8_append2_balanced_probe_v1:
  role: bounded_balanced_append
  answer context depth: 10
  protected prefix slots: 8
  replacement append slots: 2
  append sidecar slots: 2
  append sidecar can generate answer text: false
  append sidecar can support citation verification: true
```

The configs are validation candidates only. They are not runtime defaults and
are not fallback strategies.

## Validation Plan

```text
next stage: Stage132
action: run_append_candidate_evidence_shortlist_train_cv_dev_validation
selection split: train
selection mode: train_grouped_cross_validation_append_shortlist_config_selection
minimum train folds: 5
validation split: dev
dev mode: single_pass_report_only_no_retuning
baseline: stage116_top200_agent_pool_control
candidate pool: stage128_prefix_append_top400_agent_pool
primary train-CV guard: gold_citation_count_delta_vs_stage116_non_negative
```

Secondary train-CV guards:

```text
verified_f1_delta_vs_stage116_non_negative
answerable_refusal_rate_delta_vs_stage116_non_positive
unanswerable_refusal_rate_delta_vs_stage116_non_positive
changed_verified_answer_rate_not_above_stage129_candidate
append_selected_citations_do_not_displace_prefix_like_citations_without_gold_gain
```

## Guard Checks

```text
guard checks: 14 / 14 passed
public_safe_contract.forbidden_keys_found: []
test split loaded: false
final test metrics run: false
runtime defaults changed: false
fallback strategies enabled: false
```

## Visualizations

```text
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_protocol_stage131_visuals\stage131_source_failure_pressure.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_protocol_stage131_visuals\stage131_shortlist_candidate_budgets.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_protocol_stage131_visuals\stage131_validation_guard_thresholds.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_protocol_stage131_visuals\stage131_protocol_decision_flags.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_protocol_stage131_visuals\stage131_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_append_candidate_evidence_shortlist_redesign_protocol_frozen
can_run_append_shortlist_validation_now: true
can_continue_train_dev_development: true
stage128_direct_agent_integration_path_remains_blocked: true
recommended_next_direction: run_append_candidate_evidence_shortlist_train_cv_dev_validation
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
runtime_defaultization_allowed_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Next Step

Stage132 should run the frozen append-candidate evidence shortlist train-CV/dev
validation. Train grouped cross-validation is the selection surface; dev remains
single-pass report-only. Test remains locked, runtime defaults stay unchanged,
and fallback strategies stay disabled.
