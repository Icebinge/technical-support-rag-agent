# MSQA Stage 51 Candidate Distribution Review

This document records the Stage 62 review of the Stage 61 MSQA candidate
adapter distribution.

Stage 62 does not run Stage 51, does not tune policies, does not change
candidate rows, and does not change the default runtime.

## Inputs

Stage 61 adapter report:

```text
artifacts/msqa_stage51_candidate_adapter_stage61.json
```

Stage 61 candidate JSONL:

```text
artifacts/msqa_stage51_candidate_adapter_stage61_candidates.jsonl
```

Stage 31 training candidate summary:

```text
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json
```

## Stage 61 Distribution

Candidate count per MSQA query:

```text
count: 3301
min: 14
p10: 51
p25: 64
median: 79
p75: 95
p90: 113
p95: 125
p99: 151
max: 189
average: 80.7776
```

Source and gold-candidate coverage:

```text
candidate_jsonl_rows: 266647
queries_seen_in_candidate_jsonl: 3301
rows_with_question_key: 0
queries_with_gold_source_candidate: 2023
gold_source_candidate_rate: 0.6128
queries_with_top_candidate_from_gold_source: 1369
top_candidate_gold_source_rate: 0.4147
```

## Stage 31 Training-Pool Reference

Stage 31 candidate-reranker training contract:

```text
retrieval_top_k: 5
max_candidates_per_document: 3
candidate_limit: 25
effective_max_candidates: 15
average_rows_per_question: 13.2131
```

Candidate count per Stage 31 question:

```text
count: 610
min: 2
p10: 5
p25: 15
median: 15
p75: 15
p90: 15
p95: 15
p99: 15
max: 15
average: 13.2131
```

## Comparison

```text
average_candidate_count_ratio_stage61_vs_stage31: 6.1134
median_candidate_count_ratio_stage61_vs_stage31: 5.2667
stage61_median_exceeds_stage31_max: true
stage61_p10_exceeds_stage31_max: true
gold_candidate_rate_delta_stage61_minus_stage31: -0.0069
```

Interpretation:

- Gold-source availability is close to the Stage 31 training pool.
- Candidate-pool size is not close. Stage 61 median candidate count is 79, while
  Stage 31 max is 15.
- The mismatch affects almost every MSQA query because Stage 61 p10 is already
  51 candidates.

## Fairness Checks

| Check | Status | Severity |
| --- | --- | --- |
| `stage61_adapter_contract_passed` | pass | info |
| `candidate_jsonl_has_no_question_text_field` | pass | info |
| `all_stage61_samples_have_candidates` | pass | info |
| `gold_source_candidate_rate_matches_training_pool` | pass | info |
| `candidate_pool_size_aligned_with_stage31` | blocked | blocker |
| `stage61_candidate_volume_within_training_limit` | blocked | blocker |
| `direct_stage51_adapter_comparison_fair_now` | blocked | blocker |

## Decision

Current status:

```text
msqa_stage51_adapter_comparison_blocked_by_candidate_pool_mismatch
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
stage51_candidate_run_performed: false
```

Reason:

```text
The Stage 61 adapter contract passed, but the uncapped MSQA candidate pool is
much larger than the Stage 31 training candidate pool. A direct Stage 51 adapter
comparison would mix protocol effects with policy effects.
```

## Artifacts

```text
artifacts/msqa_stage51_candidate_distribution_stage62.json
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/stage62_candidate_count_percentiles.svg
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/stage62_stage31_vs_stage61_candidate_pool.svg
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/stage62_candidate_rows_by_retrieval_rank.svg
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/stage62_fairness_checks.svg
```

Stage 62 report checksum:

```text
1948ddf4101a35c5229fe6c79e50a21d956d8dabaa6dfeed029b83afe4629c79
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Stage 63 should design a Stage31-aligned MSQA candidate-pool cap and rerun the
adapter dry run before any Stage 51 comparison.
