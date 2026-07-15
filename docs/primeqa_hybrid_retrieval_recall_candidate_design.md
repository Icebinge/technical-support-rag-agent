# PrimeQA Hybrid Retrieval Recall Candidate Design

This document records Stage 76.

## Scope

Stage 76 designs train/dev-only retrieval-recall improvement candidates from
the public-safe Stage75 BM25 top10 miss analysis.

This stage does not load the frozen test split, does not run final test
metrics, does not tune on test, and does not change the default runtime policy.
It consumes only the Stage75 JSON report, which omits raw question text, raw
answer text, document titles, and document body text.

## Command

```text
python scripts\design_primeqa_hybrid_retrieval_recall_candidates.py ^
  --output artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76.json ^
  --visualization-dir artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals
```

## Input

```text
stage75 report:
  artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75.json
  sha256: 45272301a177a275c3489b176abb5f307cd3e809151e31760c36f6168e38a116
```

## Stage75 Baseline

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
  hit@10: 0.6682
  miss_count: 148
  miss_rate: 0.3318
```

## Candidate Designs

Recommended execution order:

```text
1. query_view_ablation_full_title_dedup
2. fielded_title_text_bm25_score_fusion
3. section_bm25_doc_rollup_train_dev_probe
4. dense_sparse_rrf_train_dev_probe
5. bm25_k1_b_grid_train_to_dev
```

Candidate target coverage:

| Candidate | Priority | Risk | Target Misses | Dev Targets | Status |
| --- | ---: | --- | ---: | ---: | --- |
| query_view_ablation_full_title_dedup | 196 | medium | 143 | 22 | recommended |
| fielded_title_text_bm25_score_fusion | 195 | medium | 143 | 23 | recommended |
| section_bm25_doc_rollup_train_dev_probe | 163 | medium | 119 | 17 | recommended |
| dense_sparse_rrf_train_dev_probe | 134 | high | 111 | 17 | recommended |
| bm25_k1_b_grid_train_to_dev | 69 | low | 38 | 6 | recommended |
| source_doc_ids_oracle_union_blocked | 0 | blocked | 148 | 23 | blocked |

## Blocked Diagnostic

`source_doc_ids_oracle_union_blocked` is intentionally blocked. Stage75 shows
that source `DOC_IDS` contain the gold document for the miss cases, but those
IDs are dataset source metadata, not a runtime user-query signal. Using them as
retrieval evidence would make the experiment non-deployable and misleading.

## Guard Checks

```text
source_report_is_stage75: passed
source_development_splits_are_train_dev_only: passed
source_forbidden_final_splits_include_test: passed
source_candidate_rows_have_no_test_split: passed
source_final_test_metrics_not_run: passed
source_default_runtime_policy_unchanged: passed
stage76_design_only_no_runtime_default_change: passed
stage76_uses_public_safe_stage75_report_only: passed
```

Additional local check:

```text
Select-String over the Stage76 JSON for raw-text field names and known raw text
snippets returned no matches.
```

## Visualizations

```text
artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_candidate_priority_scores.svg
artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_candidate_target_misses.svg
artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_candidate_dev_targets.svg
artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_allowed_vs_blocked_candidates.svg
```

## Decision

```text
status: primeqa_hybrid_retrieval_recall_candidate_design_completed
allowed_candidate_count: 5
blocked_candidate_count: 1
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 77 ran a train/dev-only retrieval-recall experiment for the
highest-priority allowed candidate, `query_view_ablation_full_title_dedup`, and
did not advance the route because the challenger query views underperformed the
full-question BM25 baseline.

Stage 78 ran the second allowed candidate,
`fielded_title_text_bm25_score_fusion`, and did not advance the route because
the train-selected challenger produced no dev hit@10 gain.

Stage 79 ran the third allowed candidate,
`section_bm25_doc_rollup_train_dev_probe`, and did not advance the route because
dev hit@10 regressed.

Stage 80 checked `dense_sparse_rrf_train_dev_probe` feasibility and found two
compatible local dense caches. It did not run train/dev metrics and did not
download models.

The current next step is Stage 81, but it requires confirming the dense
model/cache protocol first. The recommended option is
`compare_existing_cached_dense_models`.

Stage 81 must keep the frozen test split locked, must not run final test
metrics, must not use source `DOC_IDS` as runtime retrieval evidence, must not
download or choose dense retrieval dependencies silently, and must not change the
default runtime policy.
