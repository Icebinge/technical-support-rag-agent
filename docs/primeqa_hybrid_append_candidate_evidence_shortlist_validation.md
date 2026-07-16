---
Stage: Stage 132
Title: PrimeQA hybrid append-candidate evidence shortlist train-CV/dev validation
Status: completed
---

# PrimeQA Hybrid Append-Candidate Evidence Shortlist Validation

Stage132 runs the frozen Stage131 append-candidate evidence shortlist protocol
on train grouped cross-validation plus dev report-only validation.

This stage may load train/dev split files, the local corpus, existing local
dense cache metadata, and public-safe Stage125/128/131 protocol artifacts. It
does not load the test split, does not run final test metrics, does not change
runtime defaults, and does not add fallback strategies.

## Command

```text
python scripts\run_primeqa_hybrid_append_candidate_evidence_shortlist_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage132 append-candidate evidence shortlist train-CV/dev validation after Stage131 protocol freeze; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source

```text
stage131 status: primeqa_hybrid_append_candidate_evidence_shortlist_redesign_protocol_frozen
stage131 next direction: run_append_candidate_evidence_shortlist_train_cv_dev_validation
stage128 selected config: prefix_existing_dense_broad_append200_v1
candidate configs: 3
train rows: 562
dev rows: 121
test split loaded: false
```

## Candidate Pool

```text
train prefix identity violation count: 0
dev prefix identity violation count: 0
train append budget exceeded count: 0
dev append budget exceeded count: 0
train append count average: 200.0
dev append count average: 200.0
```

## Train-CV Selection

```text
candidate count: 3
eligible config count: 1
selected config: prefix10_append_sidecar_probe_v1
selected profile: stage132_prefix10_append_sidecar_probe_v1
selection split: train
dev used for selection: false
dev used for retuning: false
```

Train-CV ranking:

```text
prefix10_append_sidecar_probe_v1:
  guard: passed
  verified F1 delta vs Stage116: +0.0000
  gold citation count delta vs Stage116: +0
  target-depth gold hit delta vs Stage116: +9
  changed answer rate vs Stage116: 0.0000
  failed checks: []

prefix9_append1_high_precision_v1:
  guard: failed
  verified F1 delta vs Stage116: +0.0015
  gold citation count delta vs Stage116: +0
  target-depth gold hit delta vs Stage116: +9
  changed answer rate vs Stage116: 0.3932
  failed checks:
    append_selected_citations_do_not_displace_prefix_like_citations_without_gold_gain

prefix8_append2_balanced_probe_v1:
  guard: failed
  verified F1 delta vs Stage116: -0.0001
  gold citation count delta vs Stage116: -1
  target-depth gold hit delta vs Stage116: +9
  changed answer rate vs Stage116: 0.3932
  failed checks:
    verified_f1_delta_vs_stage116_non_negative
    gold_citation_count_delta_vs_stage116_non_negative
    append_selected_citations_do_not_displace_prefix_like_citations_without_gold_gain
```

## Dev Report-Only

Dev was not used for selection or retuning.

```text
prefix10_append_sidecar_probe_v1:
  verified F1 delta vs Stage116: +0.0000
  gold citation count delta vs Stage116: +0
  target-depth gold hit delta vs Stage116: +1
  changed answer rate vs Stage116: 0.0000

prefix9_append1_high_precision_v1:
  verified F1 delta vs Stage116: -0.0036
  gold citation count delta vs Stage116: -2
  target-depth gold hit delta vs Stage116: +1
  changed answer rate vs Stage116: 0.4132

prefix8_append2_balanced_probe_v1:
  verified F1 delta vs Stage116: -0.0053
  gold citation count delta vs Stage116: -2
  target-depth gold hit delta vs Stage116: +1
  changed answer rate vs Stage116: 0.4132
```

## Interpretation

`prefix10_append_sidecar_probe_v1` passed because it preserves the Stage116
answer context exactly while keeping the Stage128 top400 candidate pool
available for retrieval coverage and verification. It does not improve answer
F1 or citation count, and it should not be treated as runtime defaultization.

The two configs that allowed append candidates to replace prefix slots failed:
they selected append-region citations but still displaced prefix-like citations
without positive gold-citation gain. This confirms the Stage130 failure pattern.

## Guard Checks

```text
guard checks: 22 / 22 passed
public_safe_contract.forbidden_keys_found: []
test split loaded: false
final test metrics run: false
runtime defaults changed: false
fallback strategies enabled: false
```

## Visualizations

```text
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132_visuals\stage132_train_cv_verified_f1_delta.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132_visuals\stage132_train_cv_gold_citation_delta.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132_visuals\stage132_train_cv_changed_answer_rate.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132_visuals\stage132_selected_evidence_region_mix.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132_visuals\stage132_train_config_guard_status.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132_visuals\stage132_decision_flags.svg
artifacts\primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132_visuals\stage132_guard_check_status.svg
```

## Decision

```text
status: primeqa_hybrid_append_candidate_evidence_shortlist_validation_completed
selected_config_id: prefix10_append_sidecar_probe_v1
selected_profile_id: stage132_prefix10_append_sidecar_probe_v1
eligible_config_count: 1
recommended_next_direction: review_append_candidate_evidence_shortlist_selected_config
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
runtime_defaultization_allowed_now: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Next Step

Stage133 reviewed the selected Stage132 sidecar config and recorded the result
in:

```text
docs/primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review.md
```

The next step is Stage134: freeze a train/dev-only `Stage116 answer context +
Stage128 sidecar observation` agent protocol. Test remains locked.
