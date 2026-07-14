# MSQA Stage31-Aligned Candidate-Pool Cap

This document records the Stage 63 MSQA candidate-pool cap dry run and
distribution review.

Stage 63 does not run Stage 51, does not tune policies, does not change
candidate rows after artifact generation, and does not change the default
runtime.

## Confirmed Cap Rule

The user confirmed option A on 2026-07-14:

```text
For each retrieved source row, generate all normalized answer-sentence
candidates, score them with the existing dry-run candidate_score, then keep the
top 3 candidates per source row.
```

Stage63 uses the Stage31-aligned retrieval shape:

```text
top_k: 5
max_candidates_per_source_row: 3
effective_candidate_pool_cap: 15
```

`candidate_score` remains a dry-run adapter score only. It combines answer-only
BM25 source retrieval score, retrieval-rank prior, and query-sentence token
overlap. It is not a tuned Stage 51 model score.

## Inputs

```text
artifacts/msqa_evaluation_split_stage57.jsonl
artifacts/msqa_stage51_protocol_stage60.json
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json
```

## Commands

Adapter dry run:

```powershell
python scripts\dry_run_msqa_stage51_candidate_adapter.py `
  --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl `
  --protocol-report artifacts\msqa_stage51_protocol_stage60.json `
  --output artifacts\msqa_stage51_candidate_adapter_stage63_capped.json `
  --candidate-output artifacts\msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl `
  --visualization-dir artifacts\msqa_stage51_candidate_adapter_stage63_capped_visuals `
  --confirmed-protocol `
  --top-k 5 `
  --max-candidates-per-source-row 3 `
  --min-sentence-chars 1 `
  --sample-limit 20 `
  --stage-name "Stage 63"
```

Distribution review:

```powershell
python scripts\review_msqa_stage51_candidate_distribution.py `
  --adapter-report artifacts\msqa_stage51_candidate_adapter_stage63_capped.json `
  --candidate-jsonl artifacts\msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl `
  --stage31-summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json `
  --output artifacts\msqa_stage51_candidate_distribution_stage63_capped.json `
  --visualization-dir artifacts\msqa_stage51_candidate_distribution_stage63_capped_visuals `
  --stage-name "Stage 63"
```

## Adapter Dry-Run Result

```text
evaluation_samples: 3301
candidate_rows: 47342
samples_with_candidates: 3301
samples_without_candidates: 0
samples_with_gold_source_candidate: 1850
average_candidates_per_sample: 14.3417
median_candidates_per_sample: 15.0
max_candidates_per_sample_contract: 15
unique_source_rows_in_candidates: 2624
```

Source retrieval summary:

```text
hit@1: 0.4147
hit@5: 0.5604
mrr: 0.4692
gold_source_missing_at_5: 1451
```

Contract checks:

```text
passed: 7 / 7
rows_with_question_key: 0
no_answer_field_fallback_used: true
no_external_fetch_used: true
stage51_candidate_run_performed: false
```

## Candidate-Pool Distribution

Stage63 candidate count per MSQA query:

```text
count: 3301
min: 7.0
p10: 13.0
p25: 14.0
median: 15.0
p75: 15.0
p90: 15.0
p95: 15.0
p99: 15.0
max: 15.0
average: 14.3417
```

Stage31 candidate count per training question:

```text
count: 610
min: 2.0
p10: 5.0
p25: 15.0
median: 15.0
p75: 15.0
p90: 15.0
p95: 15.0
p99: 15.0
max: 15.0
average: 13.2131
```

Comparison:

```text
average_candidate_count_ratio_adapter_vs_stage31: 1.0854
median_candidate_count_ratio_adapter_vs_stage31: 1.0
adapter_median_exceeds_stage31_max: false
adapter_p10_exceeds_stage31_max: false
gold_candidate_rate_delta_adapter_minus_stage31: -0.0593
```

## Warning

The candidate-pool size is aligned, but source availability is lower than the
Stage31 training reference:

```text
Stage63 gold-source candidate rate: 0.5604
Stage31 gold-document candidate rate: 0.6197
delta: -0.0593
```

This is a real source-retrieval availability tradeoff, not a candidate-size
blocker. Stage64 must interpret any capped Stage51 comparison under this
top5-source-row boundary.

## Fairness Checks

| Check | Status | Severity |
| --- | --- | --- |
| `adapter_contract_passed` | pass | info |
| `candidate_jsonl_has_no_question_text_field` | pass | info |
| `all_adapter_samples_have_candidates` | pass | info |
| `gold_source_candidate_rate_matches_training_pool` | warn | info |
| `candidate_pool_size_aligned_with_stage31` | pass | info |
| `adapter_candidate_volume_within_training_limit` | pass | info |
| `direct_stage51_adapter_comparison_fair_now` | pass | info |

## Decision

```text
status: msqa_stage51_adapter_comparison_ready_for_user_confirmation
can_run_stage51_candidate_now: false
can_run_stage51_candidate_next_with_user_confirmation: true
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
stage51_candidate_run_performed: false
blocker_checks: []
```

Stage63 allows one capped Stage51 adapter comparison after user confirmation.
It does not defaultize Stage51 and does not change the runtime policy.

## Artifacts

```text
artifacts/msqa_stage51_candidate_adapter_stage63_capped.json
artifacts/msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl
artifacts/msqa_stage51_candidate_adapter_stage63_capped_visuals/stage63_adapter_candidate_counts.svg
artifacts/msqa_stage51_candidate_adapter_stage63_capped_visuals/stage63_adapter_source_hit_rates.svg
artifacts/msqa_stage51_candidate_adapter_stage63_capped_visuals/stage63_adapter_contract_checks.svg
artifacts/msqa_stage51_candidate_distribution_stage63_capped.json
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/stage63_candidate_count_percentiles.svg
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/stage63_stage31_vs_adapter_candidate_pool.svg
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/stage63_candidate_rows_by_retrieval_rank.svg
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/stage63_fairness_checks.svg
```

Checksums:

```text
adapter report: 56bef0e0f78365dc18c8ee1f2c63d25a2c17015123c6a78d0014e05669e0934a
candidate JSONL: 317c4502edb7a34eba97b5b5045fed63f05228a82e271bee0804af74467248d2
distribution report: c036e67c48f3dbd800fc8fbb81e174d830f9785080146ddd99cc62f8b5df69cf
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Stage64 ran one capped Stage51 adapter comparison against the same Stage63
capped candidate pool:

```text
docs/msqa_stage51_adapter_comparison.md
```

The comparison preserved the source-availability warning and did not change the
default runtime. It found gold-source citation gain but answer-F1 regression, so
the next step is changed-case and tradeoff review, not defaultization.
