# PrimeQA Hybrid Retrieval/Index Redesign Stop Decision

## Scope

Stage 115 records the stop decision for the frozen Stage113 retrieval/index
redesign family after Stage114 selected no train-CV-safe config.

This stage reads only the public-safe Stage114 report. It does not load
train/dev/test split files, does not load corpus documents, does not run new
retrieval or answer metrics, does not run final metrics, does not select from
dev-only observations, does not change runtime defaults, and does not add
fallback strategies.

## Command

```text
python scripts\decide_primeqa_hybrid_retrieval_index_redesign_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage115 retrieval/index redesign stop decision on 2026-07-16 after Stage114 selected 0 of 8 configs; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Source

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114.json
```

## Decision

```text
status: primeqa_hybrid_retrieval_index_redesign_family_stopped
stopped_family_id: retrieval_index_redesign_candidate_family
stopped_protocol_id: primeqa_hybrid_retrieval_index_redesign_protocol_v1
stopped_analysis_id: primeqa_hybrid_retrieval_index_redesign_train_cv_dev_validation_v1
current_route_defaultization: blocked
recommended_next_direction: user_confirmed_next_research_direction_required
```

This stop decision does not open any runtime or final-test gate:

```text
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

## Stop Reason

Stage114 found no train-CV-selectable config in the frozen Stage113
retrieval/index redesign family.

The best train-CV retrieval movement recovered only four
`retrieval_context_miss` cases and improved recall@10 by `+0.0108`, but those
configs violated downstream answer-quality or changed-answer guards. Dev was
report-only and cannot rescue configs that failed train-CV selectability.

Therefore this family provides no runtime-defaultization or final-test gate
justification.

## Family Summary

```text
entity_version_error_code_handling_candidate_v1:
  configs: 2
  selectable: 0
  best train-CV objective delta: -14.0
  best retrieval_context_miss delta: -4

title_heading_weighted_bm25_candidate_v1:
  configs: 3
  selectable: 0
  best train-CV objective delta: -10.5
  best retrieval_context_miss delta: -3

section_level_index_rollup_candidate_v1:
  configs: 3
  selectable: 0
  best train-CV objective delta: +31.5
  best retrieval_context_miss delta: +9
```

## Improved But Blocked

Four configs improved train-CV retrieval recall but were blocked:

```text
evc_special_token_exact_boost_v1:
  retrieval_context_miss delta: -4
  recall@10 delta: +0.0108
  changed answer rate: 0.1833
  failed guards:
    train_cv_answerability_false_answer_delta_within_guard
    train_cv_evidence_selection_miss_delta_within_guard

evc_special_token_title_heading_boost_v1:
  retrieval_context_miss delta: -4
  recall@10 delta: +0.0108
  changed answer rate: 0.7278
  failed guards:
    train_cv_evidence_selection_miss_delta_within_guard
    train_cv_gold_span_beats_selected_delta_within_guard
    train_cv_changed_answer_rate_within_guard

thw_title2_heading2_body1_doc_bm25_v1:
  retrieval_context_miss delta: -3
  recall@10 delta: +0.0081
  changed answer rate: 0.7171
  failed guards:
    train_cv_evidence_selection_miss_delta_within_guard
    train_cv_changed_answer_rate_within_guard

thw_title3_heading2_body1_doc_bm25_v1:
  retrieval_context_miss delta: -3
  recall@10 delta: +0.0081
  changed answer rate: 0.8327
  failed guards:
    train_cv_evidence_selection_miss_delta_within_guard
    train_cv_gold_span_beats_selected_delta_within_guard
    train_cv_changed_answer_rate_within_guard
```

## Dev Observation

Dev was report-only. It was not used for selection or retuning.

```text
dev_gate_status: report_only_no_frozen_pass_threshold
best dev F1 delta: +0.0067 from evc_special_token_title_heading_boost_v1
lowest dev changed-answer rate: 0.2231 from evc_special_token_exact_boost_v1
```

These observations do not override the train-CV stop decision.

## Visualizations

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_train_cv_retrieval_context_miss_deltas.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_train_cv_gold_doc_recall_deltas.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_train_cv_changed_answer_rates.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_train_cv_guard_failure_reasons.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_selectability_by_family.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_stop_decision_flags.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_stop_guard_check_status.svg
```

## Guard Checks

```text
25 / 25 passed
```

## Next Step

Stage116 requires user confirmation before choosing the next train/dev-only
research direction. Test remains locked, runtime defaults remain unchanged, and
no fallback strategies are added.
