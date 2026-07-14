# PrimeQA Hybrid BM25 Top10 Miss Analysis

This document records Stage 75.

## Scope

Stage 75 analyzes BM25 top10 misses on the project-owned
`primeqa_hybrid_stage68_v1` train/dev split.

This stage diagnoses answerable train/dev questions whose gold answer document
is not retrieved in the BM25 top10 window. It does not load the frozen test
split, does not run final test metrics, does not tune on test, and does not
change the default runtime policy.

The report is public-safe for project documentation: it omits raw question
text, raw answer text, document titles, and document body text. It stores IDs,
counts, route labels, rank buckets, overlap features, and aggregate summaries.

## Command

```text
python scripts\analyze_primeqa_hybrid_bm25_top10_misses.py ^
  --output artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75.json ^
  --visualization-dir artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals ^
  --top-k 10 ^
  --search-depth 50
```

The actual run used PowerShell line continuations and completed in `120.201s`.

## Inputs

```text
train split:
  artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_train.jsonl
  sha256: cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155

dev split:
  artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_dev.jsonl
  sha256: 071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f

documents:
  data\raw\primeqa_techqa\TechQA\training_and_dev\training_dev_technotes.sections.json
  sha256: f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b

candidate routes:
  artifacts\primeqa_hybrid_rebuild_stage69_candidates.jsonl
  sha256: d379d59f5172394a40bcd1852aa8188f2dec18d4abcae20d08acd992a802da4d
```

## Configuration

```text
top_k: 10
search_depth: 50
bm25_k1: 1.5
bm25_b: 0.75
```

## Results

```text
train:
  evaluated_questions: 370
  hit@10: 0.6622
  miss_count: 125
  miss_rate: 0.3378

dev:
  evaluated_questions: 76
  hit@10: 0.6974
  miss_count: 23
  miss_rate: 0.3026

cross split:
  evaluated_questions: 446
  hit_count: 298
  miss_count: 148
  hit@10: 0.6682
  miss_rate: 0.3318
```

## Miss Drivers

Across train/dev misses:

```text
gold_doc_not_found_within_top50: 110
gold_doc_rank_21_to_50: 24
gold_doc_rank_11_to_20: 14
top1_query_overlap_exceeds_gold: 103
gold_doc_query_overlap_ratio_lt_0_25: 20
gold_doc_low_query_overlap_lte_1: 1
top10_contains_source_candidate_doc: 148
```

The rank buckets show that most misses are not shallow top10 misses:

```text
not_found_top50: 110
rank_21_to_50: 24
rank_11_to_20: 14
```

Dev misses follow the same pattern:

```text
dev miss_count: 23
dev not_found_top50: 17
dev rank_21_to_50: 4
dev rank_11_to_20: 2
dev top1_query_overlap_exceeds_gold: 14
dev gold_doc_query_overlap_ratio_lt_0_25: 3
```

The largest route buckets for cross-split misses are:

```text
other: 57
error_or_log: 40
install_upgrade_config: 32
how_to_or_lookup: 12
security_bulletin_vulnerability_detail: 4
limitation_or_restriction: 3
```

## Interpretation

The main bottleneck is retrieval recall, not reranking. In 110 of 148
train/dev misses, the gold document is not found within the top50 BM25 search
depth. A reranker cannot repair those cases because the gold document never
enters the candidate window.

The near-miss group is smaller but still useful for the next stage: 38 misses
have the gold document at ranks 11-50. Those cases can test whether query
normalization, document field weighting, title/document-type features, or
hybrid lexical features improve recall before answer composition.

## Guard Checks

```text
analysis_splits_are_train_dev_only: passed
candidate_artifact_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
search_depth_covers_top_k: passed
stage75_report_is_public_safe_no_raw_text: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

Additional local check:

```text
Select-String over the Stage75 JSON for raw-text field names and known raw text
snippets returned no matches.
```

## Visualizations

```text
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_count_by_split.svg
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_rate_by_split.svg
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_reason_tags.svg
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_rank_buckets.svg
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_dev_miss_routes.svg
```

## Decision

```text
status: primeqa_hybrid_bm25_top10_miss_analysis_completed
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 76 should design train/dev-only retrieval-recall improvement candidates
from the Stage75 miss drivers. It must keep the frozen test split locked and
must not run final test metrics.
