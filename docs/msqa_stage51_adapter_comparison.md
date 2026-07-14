# MSQA Stage51 Capped Adapter Comparison

This document records the Stage64 one-time Stage51 adapter comparison against
the unchanged Stage63 capped MSQA candidate pool.

Stage64 does not rebuild the candidate pool, does not fetch external pages,
does not tune Stage51, and does not change the default runtime.

## Boundary

The user confirmed option A for Stage64:

```text
Read question text from the Stage57 frozen split only for runtime feature
computation.
```

This boundary is important:

- Question text is not written back to the Stage63 candidate JSONL.
- Question text is not used for retrieval indexing.
- The Stage63 candidate JSONL is reused unchanged.
- The comparison remains an MSQA answer-source-row proxy, not a PrimeQA
  verified RAG document-citation metric.

## Inputs

```text
artifacts/msqa_evaluation_split_stage57.jsonl
artifacts/msqa_stage51_candidate_adapter_stage63_capped.json
artifacts/msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl
artifacts/msqa_stage51_candidate_distribution_stage63_capped.json
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json
```

## Stage51 Adapter Contract

```text
model: logistic_best_candidate
train_split: train
selector_name_runtime_feature: hybrid_routing_answer_aware_mcpd3_section_span_mcpd1
composition_policy: candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker
runtime_guard: candidate_score_gte_60_all_selected_citations_rank_lte_max_citation_rank_preserve_baseline_out_of_rank_docs
max_answer_candidates: 3
rank_contained_max_retrieval_rank: 3
candidate_pool_rebuilt: false
candidate_pool_rows: 47342
```

Within each query, Stage64 ranks the unchanged capped candidates by:

```text
candidate_score descending
retrieval_rank ascending
candidate_id ascending
```

This gives the already-capped pool a Stage31-like global `candidate_rank`.

## Command

```powershell
python scripts\compare_msqa_stage51_capped_adapter.py `
  --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl `
  --candidate-jsonl artifacts\msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl `
  --adapter-report artifacts\msqa_stage51_candidate_adapter_stage63_capped.json `
  --distribution-report artifacts\msqa_stage51_candidate_distribution_stage63_capped.json `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage31-summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json `
  --output artifacts\msqa_stage51_adapter_comparison_stage64.json `
  --visualization-dir artifacts\msqa_stage51_adapter_comparison_stage64_visuals `
  --model logistic_best_candidate `
  --train-split train `
  --max-answer-candidates 3 `
  --max-citation-rank 3 `
  --sample-limit 20
```

## Results

Answer-source proxy F1:

```text
baseline_top1_average_token_f1: 0.2881
stage51_top1_average_token_f1: 0.2863
top1_average_delta_vs_baseline: -0.0018

baseline_top3_average_answer_token_f1: 0.4199
stage51_top3_average_answer_token_f1: 0.4171
top3_average_delta_vs_baseline: -0.0028

oracle_best_single_average_token_f1: 0.4156
```

Top3 outcomes:

```text
question_count: 3301
top3_improved_count: 20
top3_regressed_count: 57
top3_tied_count: 3224
changed_answer_count: 719
changed_answer_rate: 0.2178
replacement_count: 719
replacement_rate: 0.2178
```

Gold-source citation:

```text
baseline_gold_source_citation_count: 1394
stage51_gold_source_citation_count: 1397
baseline_gold_source_citation_rate: 0.4223
stage51_gold_source_citation_rate: 0.4232
gold_source_citation_delta: +3
citation_lost_count: 0
citation_gained_count: 3
```

Decision reasons:

```text
candidate_score_gte_60_accepted: 719
candidate_score_gte_60_blocked: 479
model_selected_top_candidate: 1050
score_margin_below_min: 555
selected_rank_exceeds_limit: 498
```

Selected-rank distribution:

```text
rank_1: 1050
rank_2: 717
rank_3: 585
rank_4_5: 451
rank_6_10: 415
rank_11_15: 83
```

## Stage63 Warning Preserved

Stage64 preserves the Stage63 source-availability warning:

```text
Stage63 gold-source candidate rate: 0.5604
Stage31 gold-document candidate rate: 0.6197
delta: -0.0593
```

This warning means the Stage64 comparison is performed under a top5 source-row
availability boundary. Missing gold source rows cannot be recovered by Stage51.

## Route Metrics

```text
other:
  question_count: 2402
  top3_average_delta_vs_baseline: -0.0017
  changed_answer_count: 488
  replacement_count: 488
  top3_improved_count: 13
  top3_regressed_count: 28
  gold_source_citation_delta: +3

error_or_log:
  question_count: 452
  top3_average_delta_vs_baseline: -0.0050
  changed_answer_count: 136
  replacement_count: 136
  top3_improved_count: 4
  top3_regressed_count: 12
  gold_source_citation_delta: 0

install_upgrade_config:
  question_count: 351
  top3_average_delta_vs_baseline: -0.0078
  changed_answer_count: 77
  replacement_count: 77
  top3_improved_count: 3
  top3_regressed_count: 16
  gold_source_citation_delta: 0

limitation_or_restriction:
  question_count: 95
  top3_average_delta_vs_baseline: -0.0007
  changed_answer_count: 18
  replacement_count: 18
  top3_improved_count: 0
  top3_regressed_count: 1
  gold_source_citation_delta: 0
```

## Interpretation

Stage64 found a mixed signal:

- Stage51 gained 3 gold-source citations and lost none.
- Stage51 reduced top3 answer-source proxy F1 by 0.0028.
- Stage51 changed 719 answers, but only 20 improved and 57 regressed by top3
  answer F1.
- The largest positive examples are source-correction cases, but several
  regressions replace a strong same-source baseline sentence with a weaker
  cross-source leading sentence.

This result is not suitable for defaultization. It is evidence that Stage51 may
help MSQA source citation in a few cases, while introducing answer-text risk on
this external adapter.

## Decision

```text
status: msqa_stage51_capped_adapter_comparison_f1_regressed
stage51_adapter_comparison_run_performed: true
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
candidate_pool_reused_without_rebuild: true
source_availability_warning_preserved: true
```

## Artifacts

```text
artifacts/msqa_stage51_adapter_comparison_stage64.json
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/stage64_msqa_answer_f1.svg
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/stage64_msqa_answer_f1_delta.svg
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/stage64_msqa_gold_source_citation.svg
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/stage64_msqa_decision_reasons.svg
```

Stage64 report checksum:

```text
2dd3427631028248c552c1ec68983ff3e271d7332940bb7768a58b63c0aaa8b3
```

These artifacts are local ignored outputs and are not committed by git policy.

## Stage65 Follow-Up

Stage65 reviewed the Stage64 changed cases and source-citation tradeoffs:

```text
docs/msqa_stage51_changed_case_review.md
```

Stage65 confirmed that consistency checks pass, citation gain is real within the
MSQA proxy, but top3 regressions outnumber improvements. Stage51 remains
non-default.
