# PrimeQA Hybrid Section BM25 Doc Rollup

This document records Stage 79.

## Scope

Stage 79 runs the third allowed Stage76 retrieval-recall candidate:
`section_bm25_doc_rollup_train_dev_probe`.

This is a train/dev-only experiment. It evaluates the existing
`SectionBM25Retriever` behavior: score document sections with BM25, then roll up
the best section score to the parent document. It compares that fixed section
candidate against the current full-document BM25 baseline.

This stage does not load the frozen test split, does not run final test metrics,
does not use source `DOC_IDS` as runtime retrieval evidence, does not introduce
new model downloads or dense retrieval choices, and does not change the default
runtime policy.

The report omits raw question text, raw answer text, document titles, and
document body text. It records metrics, config IDs, sample IDs for rank
transitions, ranks, aggregate summaries, source fingerprints, and guard checks.

## Command

```text
python scripts\run_primeqa_hybrid_section_bm25_doc_rollup.py ^
  --output artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79.json ^
  --visualization-dir artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals ^
  --search-depth 50
```

The actual run completed in `458.995s`.

## Configuration

```text
baseline:
  full_document_bm25_baseline

candidate:
  section_bm25_max_section_rollup

section rollup:
  max_section_score_per_parent_document

search_depth:
  50

primary metric:
  dev hit@10
```

Loaded section index:

```text
document_count: 28482
section_count: 216648
documents_with_sections: 28482
documents_without_sections: 0
average_sections_per_document: 7.6065
```

## Results

Train:

```text
full_document_bm25_baseline:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  not_found@50: 93

section_bm25_max_section_rollup:
  hit@1: 0.4054
  hit@5: 0.5324
  hit@10: 0.5919
  MRR@10: 0.4631
  not_found@50: 94

delta:
  hit@1: -0.0189
  hit@5: -0.0730
  hit@10: -0.0703
  MRR@10: -0.0392
  not_found@50 delta: +1
  top10 improvements/regressions: 8 / 34
  search-depth improvements/regressions: 17 / 18
```

Dev:

```text
full_document_bm25_baseline:
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  MRR@10: 0.5331
  not_found@50: 17

section_bm25_max_section_rollup:
  hit@1: 0.4342
  hit@5: 0.6184
  hit@10: 0.6447
  MRR@10: 0.4974
  not_found@50: 18

delta:
  hit@1: +0.0000
  hit@5: -0.0395
  hit@10: -0.0527
  MRR@10: -0.0357
  not_found@50 delta: +1
  top10 improvements/regressions: 1 / 5
  search-depth improvements/regressions: 2 / 3
```

## Interpretation

The section BM25 max-section rollup does not improve retrieval recall. It
helps one dev sample enter the top10, but it loses five existing dev top10 hits.
It also increases dev not-found-within-top50 from `17` to `18`.

This suggests that simply scoring isolated sections and rolling up the maximum
section score is too brittle for the current long technical documents. It can
surface a sharply matching section, but it also removes useful full-document
context and hurts stable top10 recall.

Because the active blocker is gold document entry into the top10 candidate
window, a dev hit@10 delta of `-0.0527` blocks this route from runtime
defaultization or final-test evaluation.

## Guard Checks

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
stage75_source_report_is_stage75: passed
stage78_source_report_is_stage78: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
stage78_did_not_open_final_test_gate: passed
section_index_has_nonempty_sections: passed
source_doc_ids_not_used_as_runtime_evidence: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

Additional local check:

```text
Select-String over the Stage79 JSON for raw-text field names and known raw text
snippets returned no matches.
```

Stage79 JSON SHA256:

```text
32A6B0CF36E041E9C67AE430569AD4B4587EC07210C93D9654C8A4305406E1E7
```

## Visualizations

```text
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_train_hit_at_10.svg
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_hit_at_10.svg
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_not_found_at_50.svg
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_top10_changes.svg
```

Visualization SHA256:

```text
stage79_section_bm25_train_hit_at_10.svg: 3D6CC28F5B81F764A5338041FAB11B362E9373DF546AAB0ED4935A0F08BB3342
stage79_section_bm25_dev_hit_at_10.svg: 658A27A23C8A2895E5329EEA41B0D6A07595AF3557FD89FA7EFB189817895F92
stage79_section_bm25_dev_delta_hit_at_10.svg: E6C797882D2ABF3841C577F2AE3F99027D781D6C1BCE066A31F8DB776BEEDB6D
stage79_section_bm25_dev_not_found_at_50.svg: C122844ADCB94C288F7AAA8528DE769F7FB91F1E7B269FAC4BEBD46C35F1305C
stage79_section_bm25_dev_top10_changes.svg: D9C6CFA1A98A782D41DAEE67B4FC325A2C0D799D594D6CBB5F24FBE8EF438EA2
```

## Validation

```text
ruff check .: passed
pytest -q: 207 passed
git diff --check: passed
raw-text Select-String check over Stage79 JSON: no matches
artifact git check-ignore: artifacts ignored by .gitignore
```

## Decision

```text
status: primeqa_hybrid_section_bm25_doc_rollup_completed
candidate_config_id: section_bm25_max_section_rollup
candidate_dev_hit10_delta: -0.0527
candidate_dev_top10_improvements: 1
candidate_dev_top10_regressions: 5
candidate_dev_not_found_at_search_depth_delta: 1
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 80 should check `dense_sparse_rrf_train_dev_probe` feasibility before any
train/dev run. The check must record local model/cache identity and must not
download models or choose external dense retrieval dependencies silently.

Stage 80 must keep the frozen test split locked, avoid source `DOC_IDS` as
runtime retrieval evidence, and not run final test metrics.
