# PrimeQA Hybrid Query View Ablation

This document records Stage 77.

## Scope

Stage 77 runs the highest-priority Stage76 candidate:
`query_view_ablation_full_title_dedup`.

This is a train/dev-only retrieval-recall experiment. It evaluates fixed query
views against the same BM25 document index. It does not load the frozen test
split, does not run final test metrics, does not use source `DOC_IDS` as
runtime retrieval evidence, and does not change the default runtime policy.

The report omits raw question text, raw answer text, document titles, and
document body text. It records metrics, query-view names, sample IDs for rank
transitions, ranks, and aggregate summaries.

## Command

```text
python scripts\run_primeqa_hybrid_query_view_ablation.py ^
  --output artifacts\primeqa_hybrid_query_view_ablation_stage77.json ^
  --visualization-dir artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals
```

The actual run completed in `218.083s`.

## Query Views

```text
full_question_baseline:
  current retrieval query: question title plus question text

title_only:
  question title only

full_question_dedup_terms:
  tokenized full question with duplicate lexical terms removed while preserving
  first occurrence order
```

## Results

Train:

```text
full_question_baseline:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR: 0.5023
  miss_count_at_10: 125

title_only:
  hit@1: 0.3865
  hit@5: 0.5486
  hit@10: 0.6054
  MRR: 0.4589
  hit@10 delta vs baseline: -0.0568

full_question_dedup_terms:
  hit@1: 0.4135
  hit@5: 0.5892
  hit@10: 0.6432
  MRR: 0.4908
  hit@10 delta vs baseline: -0.0190
```

Dev:

```text
full_question_baseline:
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  MRR: 0.5331
  miss_count_at_10: 23

title_only:
  hit@1: 0.3947
  hit@5: 0.5789
  hit@10: 0.6184
  MRR: 0.4638
  hit@10 delta vs baseline: -0.0790
  top10 improvements/regressions: 1 / 7

full_question_dedup_terms:
  hit@1: 0.4342
  hit@5: 0.6053
  hit@10: 0.6579
  MRR: 0.5116
  hit@10 delta vs baseline: -0.0395
  top10 improvements/regressions: 1 / 4
```

Train selected challenger:

```text
selected_view_id: full_question_dedup_terms
selected_dev_hit@10_delta: -0.0395
selected_dev_top10_improvements: 1
selected_dev_top10_regressions: 4
```

## Interpretation

The query-view candidate does not improve retrieval recall. Both challenger
views underperform the current full-question BM25 baseline on train and dev.
The best challenger by train metrics, `full_question_dedup_terms`, still loses
`0.0395` dev hit@10 and adds more top10 regressions than improvements.

This means the Stage77 query-view ablation should not move toward runtime
defaultization or final-test evaluation. The next train/dev-only route should
move to the next Stage76 candidate: `fielded_title_text_bm25_score_fusion`.

## Guard Checks

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
stage75_source_report_is_stage75: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
source_doc_ids_not_used_as_runtime_evidence: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

Additional local check:

```text
Select-String over the Stage77 JSON for raw-text field names and known raw text
snippets returned no matches.
```

## Visualizations

```text
artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_train_hit_at_10.svg
artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_dev_hit_at_10.svg
artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_dev_top10_changes.svg
```

## Decision

```text
status: primeqa_hybrid_query_view_ablation_completed
train_selected_view_id: full_question_dedup_terms
train_selected_dev_hit10_delta: -0.0395
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 78 should move to the next Stage76 candidate:
`fielded_title_text_bm25_score_fusion`.

Stage 78 must remain train/dev-only, keep the frozen test split locked, and not
run final test metrics.
