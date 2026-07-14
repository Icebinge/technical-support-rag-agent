# MSQA Stage51 Changed Case Review

This document records the Stage65 review of Stage64 changed cases and
source-citation tradeoffs.

Stage65 does not run a new Stage51 comparison, does not rebuild the candidate
pool, does not tune guards, and does not change the default runtime.

## Boundary

Stage65 rebuilds the full Stage64 case view from the same inputs and checks that
the aggregate metrics match the saved Stage64 report. It analyzes only the
already-approved Stage64 capped MSQA adapter comparison.

This is still an MSQA answer-source-row proxy review, not a PrimeQA verified RAG
document-citation metric.

## Inputs

```text
artifacts/msqa_stage51_adapter_comparison_stage64.json
artifacts/msqa_evaluation_split_stage57.jsonl
artifacts/msqa_stage51_candidate_adapter_stage63_capped.json
artifacts/msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl
artifacts/msqa_stage51_candidate_distribution_stage63_capped.json
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json
```

## Command

```powershell
python scripts\review_msqa_stage51_changed_cases.py `
  --stage64-report artifacts\msqa_stage51_adapter_comparison_stage64.json `
  --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl `
  --candidate-jsonl artifacts\msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl `
  --adapter-report artifacts\msqa_stage51_candidate_adapter_stage63_capped.json `
  --distribution-report artifacts\msqa_stage51_candidate_distribution_stage63_capped.json `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage31-summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json `
  --output artifacts\msqa_stage51_changed_case_review_stage65.json `
  --visualization-dir artifacts\msqa_stage51_changed_case_review_stage65_visuals
```

## Rebuild Check

```text
candidate_pool_rebuilt: false
case_count: 3301
model_name: logistic_best_candidate
train_split: train
max_answer_candidates: 3
max_citation_rank: 3
consistency_checks_passed: true
```

The rebuilt metrics matched Stage64 for:

```text
question_count
changed_answer_count
replacement_count
top3_improved_count
top3_regressed_count
gold_source_citation_delta
citation_lost_count
citation_gained_count
baseline_top3_average_answer_token_f1
stage51_top3_average_answer_token_f1
top3_average_delta_vs_baseline
```

## Results

Changed-case summary:

```text
question_count: 3301
changed_answer_count: 719
changed_answer_rate: 0.2178

top3_regression_count: 57
top3_improvement_count: 20
regression_to_improvement_count_ratio: 2.85

regression_loss_sum: -10.5636
improvement_gain_sum: 1.3434
net_top3_delta_sum: -9.2202

citation_gained_count: 3
citation_lost_count: 0
citation_delta: +3
regressions_with_citation_gain: 0
improvements_with_citation_gain: 3
changed_without_f1_or_citation_gain: 642
```

Changed cases by route:

```text
other: 488
error_or_log: 136
install_upgrade_config: 77
limitation_or_restriction: 18
```

Regressions by route:

```text
other: 28
install_upgrade_config: 16
error_or_log: 12
limitation_or_restriction: 1
```

Changed cases by selected rank:

```text
rank_2: 352
rank_3: 289
rank_4_5: 78
```

Source transitions among changed cases:

```text
same_source_sentence_rewrite: 631
leading_source_changed: 85
gold_source_added: 3
```

Concentration findings:

```text
regression_selected_rank_share:
  rank_4_5: 57 / 57

regression_source_transition_share:
  leading_source_changed: 57 / 57

citation_gain_source_transition_share:
  gold_source_added: 3 / 3
```

Largest regression samples:

```text
493945: route error_or_log, delta -0.6668, rank 4, source_transition leading_source_changed
288802: route install_upgrade_config, delta -0.6393, rank 4, source_transition leading_source_changed
1126961: route other, delta -0.6255, rank 4, source_transition leading_source_changed
```

Citation gained samples:

```text
23845: route other, delta +0.3385, rank 4, source_transition gold_source_added
539049: route other, delta +0.2757, rank 5, source_transition gold_source_added
842129: route other, delta +0.2700, rank 5, source_transition gold_source_added
```

## Interpretation

Stage65 sharpens the Stage64 conclusion:

- The citation gain is real within the MSQA answer-source-row proxy: 3 gained,
  0 lost.
- The answer-risk signal is stronger than the citation gain: 57 top3
  regressions vs 20 improvements.
- Every top3 regression came from a rank 4-5 leading-source change.
- Most changed answers were same-source sentence rewrites, but those did not
  account for the regressions.
- The 3 citation-gained cases were also improvements, but they are too few to
  offset the regression pattern.

The current Stage51 adapter remains non-default research evidence only.

## Decision

```text
status: msqa_stage51_changed_case_review_blocks_defaultization
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
stage51_adapter_comparison_run_performed: false
candidate_pool_rebuilt: false
consistency_checks_passed: true
regression_count_exceeds_improvement_count: true
citation_lost_count: 0
citation_gained_count: 3
```

## Artifacts

```text
artifacts/msqa_stage51_changed_case_review_stage65.json
artifacts/msqa_stage51_changed_case_review_stage65_visuals/stage65_msqa_changed_outcomes.svg
artifacts/msqa_stage51_changed_case_review_stage65_visuals/stage65_msqa_regressions_by_route.svg
artifacts/msqa_stage51_changed_case_review_stage65_visuals/stage65_msqa_changed_by_selected_rank.svg
artifacts/msqa_stage51_changed_case_review_stage65_visuals/stage65_msqa_source_transitions.svg
```

Stage65 report checksum:

```text
089003101663b7b2880ad359b5eb2f6065778a6ece3983e154f3ba59b8e69def
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Stage66 should explicitly choose the next evaluation route:

1. Find another external dataset and run a fresh schema/license/leakage
   qualification flow.
2. Design an MSQA-specific rank-4/5 leading-source risk guard and run one new
   frozen MSQA experiment.
3. Freeze Stage51 as non-default research evidence and keep top-k as the runtime
   default.

Until that route is chosen, do not defaultize Stage51.
