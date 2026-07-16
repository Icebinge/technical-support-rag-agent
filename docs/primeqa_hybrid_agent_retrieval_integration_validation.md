---
Stage: Stage 129
Title: PrimeQA hybrid agent retrieval integration validation
Status: blocked_or_failed
---

# PrimeQA Hybrid Agent Retrieval Integration Validation

Stage129 runs the train-CV/dev validation for the Stage128 frozen agent retrieval
integration protocol.

This stage validates a two-step agent consumer:

1. Stage128 candidate pool: preserve Stage116 ranks 1-200, append ranks 201-400.
2. Agent evidence shortlisting: select 10 runtime-visible evidence-context
   documents from the candidate pool before sentence evidence selection.

The final result is blocked/failed for runtime approval because train-CV gold
citation count regressed by 1 versus the Stage116 top200 agent-pool control.

## Command

```text
python scripts\run_primeqa_hybrid_agent_retrieval_integration_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage129 agent retrieval integration train-CV/dev validation after Stage128 protocol freeze; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Process Corrections

The first implementation attempted to let the sentence evidence selector scan
the full 400-document candidate pool directly. That run exceeded the tool-side
20-minute execution window and was stopped. The implementation was then changed
to the intended agent shape: high-recall candidate pool first, cheap document
shortlist second, sentence evidence selection third.

The next completed run revealed a reporting bug: Stage116/Stage128 candidate
pools were built only for answerable rows, so unanswerable answerability metrics
were artificially improved. That artifact was not used as the final result. The
bug was fixed by constructing candidate pools for all train/dev rows, and the
final Stage129 report below is from the corrected rerun.

## Profiles

```text
baseline profile:
  id: stage102_bm25_top10_verified_baseline
  candidate depth: 10
  answer context depth: 10
  verifier max citation rank: 3

stage116 control:
  id: stage116_top200_agent_pool_control
  candidate depth: 200
  answer context depth: 10
  verifier max citation rank: 200

stage128 candidate:
  id: stage128_prefix_append_top400_agent_pool
  candidate depth: 400
  answer context depth: 10
  verifier max citation rank: 400
```

## Candidate Pool Checks

```text
train rows: 562
dev rows: 121
train append count average/median/p95/max: 200.0 / 200.0 / 200.0 / 200
dev append count average/median/p95/max: 200.0 / 200.0 / 200.0 / 200
prefix identity violations: 0
append budget exceeded: 0
```

## Train-CV Results

Baseline:

```text
profile: stage102_bm25_top10_verified_baseline
verified F1: 0.2015
gold citation rate: 0.5014
gold citation count: 176
answerable refusal rate: 0.0514
unanswerable refusal rate: 0.0677
gold hit at profile depth: 245 / 370 = 0.6622
```

Stage116 top200 control:

```text
profile: stage116_top200_agent_pool_control
verified F1: 0.1946
gold citation rate: 0.4092
gold citation count: 151
answerable refusal rate: 0.0027
unanswerable refusal rate: 0.0052
gold hit at profile depth: 345 / 370 = 0.9324
```

Stage128 top400 candidate:

```text
profile: stage128_prefix_append_top400_agent_pool
verified F1: 0.1949
gold citation rate: 0.4065
gold citation count: 150
answerable refusal rate: 0.0027
unanswerable refusal rate: 0.0052
gold hit at profile depth: 354 / 370 = 0.9568
changed verified answers vs Stage116 control: 221
```

Stage128 vs Stage116 control:

```text
verified F1 delta: +0.0003
gold citation rate delta: -0.0027
gold citation count delta: -1
answerable refusal rate delta: +0.0000
unanswerable refusal rate delta: +0.0000
gold hit count at profile depth delta: +9
gold hit rate at profile depth delta: +0.0244
```

Train-CV checks:

```text
verified_f1_delta_vs_stage116_non_negative: passed
gold_citation_count_delta_vs_stage116_non_negative: failed
answerable_refusal_rate_delta_vs_stage116_non_positive: passed
target_depth_recall_delta_vs_stage116_positive: passed
```

## Dev Report-Only Results

```text
profile: stage128_prefix_append_top400_agent_pool
verified F1: 0.1837
gold citation rate: 0.4079
answerable refusal rate: 0.0000
gold hit at profile depth: 70 / 76 = 0.9211
changed verified answers vs Stage116 control: 50
dev selection allowed: false
dev retuning allowed: false
```

Dev was reported only. It was not used for selection, retuning, guard
thresholding, runtime defaultization, or test-gate decisions.

## Selected Evidence Region Mix

Train-CV Stage128 selected citations:

```text
rank 1-10: 1027
Stage116 prefix ranks 11-200: 611
Stage128 append ranks 201-400: 42
```

Dev Stage128 selected citations:

```text
rank 1-10: 210
Stage116 prefix ranks 11-200: 141
Stage128 append ranks 201-400: 12
```

The append region was used by the evidence shortlister, but this did not
translate into a gold-citation-safe improvement.

## Guard Checks

```text
guard checks: 20 / 21 passed
failed guard: stage129_agent_answer_quality_train_cv_guard
failed train-CV sub-check: gold_citation_count_delta_vs_stage116_non_negative
public_safe_contract.forbidden_keys_found: []
test split loaded: false
final test metrics run: false
runtime defaultization allowed now: false
fallback strategies enabled: false
default runtime policy: unchanged
```

## Visualizations

```text
artifacts\primeqa_hybrid_agent_retrieval_integration_validation_stage129_visuals\stage129_train_cv_verified_f1.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_validation_stage129_visuals\stage129_train_cv_gold_citation_rate.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_validation_stage129_visuals\stage129_train_cv_answer_quality_deltas.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_validation_stage129_visuals\stage129_target_depth_recall.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_validation_stage129_visuals\stage129_selected_evidence_region_mix.svg
artifacts\primeqa_hybrid_agent_retrieval_integration_validation_stage129_visuals\stage129_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_agent_retrieval_integration_validation_blocked_or_failed
selected_profile_id: stage128_prefix_append_top400_agent_pool
train_cv_validation_passed: false
train_cv_failed_checks:
  gold_citation_count_delta_vs_stage116_non_negative
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
runtime_defaultization_allowed_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_direction: review_stage129_agent_integration_failure_patterns
```

## Next Step

Stage130 should review Stage129 agent-integration failure patterns before any
new runtime or final-test gate. The review should focus on the 221 train-CV
changed verified answers versus Stage116 control and the -1 train-CV gold
citation regression.

Stage130 completed this review in:

```text
docs/primeqa_hybrid_agent_integration_failure_review.md
```

The review blocks the direct Stage128 agent-integration path and recommends a
new append-candidate evidence shortlist redesign protocol.
