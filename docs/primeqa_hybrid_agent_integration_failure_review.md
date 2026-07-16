---
Stage: Stage 130
Title: PrimeQA hybrid agent integration failure review
Status: completed
---

# PrimeQA Hybrid Agent Integration Failure Review

Stage130 reviews the Stage129 agent retrieval integration validation failure
using only the saved public-safe Stage129 aggregate report.

This stage does not load split files, corpus documents, raw candidate rows, raw
questions, raw answers, raw document identifiers, or test data. It does not run
retrieval, answer generation, final metrics, runtime defaultization, or fallback
strategies.

## Command

```text
python scripts\review_primeqa_hybrid_agent_integration_failure_patterns.py --user-confirmed-review --confirmation-note "user confirmed Stage130 agent integration failure-pattern review after Stage129 validation blocked; public-safe aggregate only; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source

```text
source stage: Stage 129
source analysis id: primeqa_hybrid_agent_retrieval_integration_validation_v1
source status: primeqa_hybrid_agent_retrieval_integration_validation_blocked_or_failed
source next direction: review_stage129_agent_integration_failure_patterns
source guard checks: 20 / 21 passed
source failed guard: stage129_agent_answer_quality_train_cv_guard
source failed sub-check: gold_citation_count_delta_vs_stage116_non_negative
```

## Train-CV Failure Review

Stage128 top400 versus Stage116 top200 control:

```text
verified F1 delta: +0.0003
gold citation rate delta: -0.0027
gold citation count delta: -1
answerable refusal rate delta: +0.0000
unanswerable refusal rate delta: +0.0000
gold hit count at profile depth delta: +9
gold hit rate at profile depth delta: +0.0244
changed verified answers: 221 / 562 = 0.3932
```

Selected citation region shift:

```text
rank 1-10 delta: -16
Stage116 prefix ranks 11-200 delta: -26
Stage128 append ranks 201-400 delta: +42
prefix-like selected citation delta: -42
append selected citations: 42
```

Direct metric conclusion: Stage128 improved target-depth recall but failed the
gold-citation safety requirement.

Aggregate inference: append-region citations are active, and they appear to
replace stable prefix-like citations without a gold-citation-safe net gain.

## Dev Report-Only Review

Stage128 top400 versus Stage116 top200 control:

```text
verified F1 delta: -0.0036
gold citation count delta: -2
gold hit count at profile depth delta: +1
changed verified answers: 50 / 121 = 0.4132
```

Selected citation region shift:

```text
rank 1-10 delta: -10
Stage116 prefix ranks 11-200 delta: -2
Stage128 append ranks 201-400 delta: +12
prefix-like selected citation delta: -12
append selected citations: 12
```

Dev remains report-only and was not used for selection or retuning. Its
aggregate direction is consistent with the train-CV citation-risk signal.

## Failure Patterns

```text
recall_gain_not_citation_safe:
  basis: direct_metric
  severity: blocking
  score: 19.0
  train gold hit delta: +9
  train gold citation delta: -1
  dev gold hit delta: +1
  dev gold citation delta: -2

append_region_displaces_prefix_evidence:
  basis: aggregate_region_mix_inference
  severity: high
  score: 84.0
  train append selected citations: 42
  train prefix-like selected citation delta: -42
  dev append selected citations: 12
  dev prefix-like selected citation delta: -12

changed_answer_churn_too_high:
  basis: direct_metric
  severity: high
  score: 39.32
  train changed verified answers: 221 / 562 = 0.3932
  dev changed verified answers: 50 / 121 = 0.4132

dev_report_confirms_risk_direction:
  basis: dev_report_only_metric
  severity: medium
  score: 2.0
  dev gold citation delta: -2
  dev verified F1 delta: -0.0036
  dev gold hit delta: +1
```

## Action Boundary

```text
stage128 direct agent integration path blocked: true
stage128 runtime defaultization allowed now: false
stage128 final test gate allowed now: false
test remains locked: true
runtime default policy: unchanged
fallback strategies enabled: false
```

Stage128 top400 remains useful as a recall candidate pool, but it is not safe as
the current agent integration route. The next redesign should preserve Stage116
evidence stability and treat append candidates as supplemental evidence, not
unrestricted replacements.

## Guard Checks

```text
guard checks: 12 / 12 passed
public_safe_contract.forbidden_keys_found: []
test split loaded: false
final test metrics run: false
runtime defaults changed: false
fallback strategies enabled: false
```

## Visualizations

```text
artifacts\primeqa_hybrid_agent_integration_failure_review_stage130_visuals\stage130_train_cv_key_deltas.svg
artifacts\primeqa_hybrid_agent_integration_failure_review_stage130_visuals\stage130_dev_key_deltas.svg
artifacts\primeqa_hybrid_agent_integration_failure_review_stage130_visuals\stage130_changed_answer_churn.svg
artifacts\primeqa_hybrid_agent_integration_failure_review_stage130_visuals\stage130_region_displacement.svg
artifacts\primeqa_hybrid_agent_integration_failure_review_stage130_visuals\stage130_failure_pattern_scores.svg
artifacts\primeqa_hybrid_agent_integration_failure_review_stage130_visuals\stage130_decision_flags.svg
artifacts\primeqa_hybrid_agent_integration_failure_review_stage130_visuals\stage130_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_stage129_agent_integration_failure_review_completed
stage128_direct_agent_integration_path_blocked: true
recommended_next_direction: freeze_append_candidate_evidence_shortlist_redesign_protocol
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
runtime_defaultization_allowed_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Next Step

Stage131 froze the train/dev-only append-candidate evidence shortlist redesign
protocol in:

```text
docs/primeqa_hybrid_append_candidate_evidence_shortlist_protocol.md
```

The next step is Stage132: run the frozen train-CV/dev append-shortlist
validation. Test remains locked.
