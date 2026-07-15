# PrimeQA Hybrid Fielded BM25 Fusion

This document records Stage 78.

## Scope

Stage 78 runs the second allowed Stage76 retrieval-recall candidate:
`fielded_title_text_bm25_score_fusion`.

This is a train/dev-only experiment. It evaluates a fixed title/text BM25 score
fusion grid against the current full-document BM25 baseline. It does not load
the frozen test split, does not run final test metrics, does not use source
`DOC_IDS` as runtime retrieval evidence, and does not change the default runtime
policy.

The report omits raw question text, raw answer text, document titles, and
document body text. It records metrics, config IDs, sample IDs for rank
transitions, ranks, aggregate summaries, source fingerprints, and guard checks.

## Command

```text
python scripts\run_primeqa_hybrid_fielded_bm25_fusion.py ^
  --output artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78.json ^
  --visualization-dir artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals ^
  --candidate-depth 100
```

The actual run completed in `243.716s`.

## Configuration

```text
baseline:
  full_document_bm25_baseline

candidate_depth:
  100 per title/text field index

fusion score:
  normalized_title_score * title_weight + normalized_text_score * text_weight

selection rule:
  select on train by hit@10, then hit@5, hit@1, MRR, then config_id;
  dev is used only for validation
```

Fixed challenger grid:

```text
fielded_title_0_25_text_1_00
fielded_title_0_50_text_1_00
fielded_title_1_00_text_1_00
fielded_title_1_50_text_1_00
fielded_title_2_00_text_1_00
```

## Results

Train:

```text
full_document_bm25_baseline:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR: 0.5023
  miss_count_at_10: 125

fielded_title_0_25_text_1_00:
  hit@1: 0.4054
  hit@5: 0.5892
  hit@10: 0.6378
  MRR: 0.4881
  hit@10 delta vs baseline: -0.0244
  top10 improvements/regressions: 4 / 13

fielded_title_0_50_text_1_00:
  hit@1: 0.4000
  hit@5: 0.5622
  hit@10: 0.6135
  MRR: 0.4700
  hit@10 delta vs baseline: -0.0487
  top10 improvements/regressions: 9 / 27

fielded_title_1_00_text_1_00:
  hit@1: 0.3811
  hit@5: 0.5216
  hit@10: 0.5730
  MRR: 0.4428
  hit@10 delta vs baseline: -0.0892

fielded_title_1_50_text_1_00:
  hit@1: 0.3703
  hit@5: 0.5108
  hit@10: 0.5432
  MRR: 0.4291
  hit@10 delta vs baseline: -0.1190

fielded_title_2_00_text_1_00:
  hit@1: 0.3703
  hit@5: 0.5000
  hit@10: 0.5405
  MRR: 0.4261
  hit@10 delta vs baseline: -0.1217
```

Dev:

```text
full_document_bm25_baseline:
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  MRR: 0.5331
  miss_count_at_10: 23

fielded_title_0_25_text_1_00:
  hit@1: 0.4737
  hit@5: 0.6579
  hit@10: 0.6974
  MRR: 0.5550
  hit@10 delta vs baseline: +0.0000
  top10 improvements/regressions: 1 / 1

fielded_title_0_50_text_1_00:
  hit@1: 0.4342
  hit@5: 0.6184
  hit@10: 0.6842
  MRR: 0.5200
  hit@10 delta vs baseline: -0.0132
  top10 improvements/regressions: 2 / 3

fielded_title_1_00_text_1_00:
  hit@1: 0.4342
  hit@5: 0.5526
  hit@10: 0.6316
  MRR: 0.4922
  hit@10 delta vs baseline: -0.0658
  top10 improvements/regressions: 2 / 7

fielded_title_1_50_text_1_00:
  hit@1: 0.4211
  hit@5: 0.5526
  hit@10: 0.6053
  MRR: 0.4713
  hit@10 delta vs baseline: -0.0921
  top10 improvements/regressions: 2 / 9

fielded_title_2_00_text_1_00:
  hit@1: 0.3947
  hit@5: 0.5395
  hit@10: 0.6053
  MRR: 0.4566
  hit@10 delta vs baseline: -0.0921
  top10 improvements/regressions: 3 / 10
```

Train selected challenger:

```text
selected_config_id: fielded_title_0_25_text_1_00
selected_dev_hit@10_delta: +0.0000
selected_dev_top10_improvements: 1
selected_dev_top10_regressions: 1
```

## Interpretation

The fielded title/text score fusion candidate does not improve retrieval
recall. The train-selected challenger is the lightest title boost, but even that
configuration trails the baseline on train hit@10 by `0.0244`. On dev it raises
hit@1 by `0.0395` and MRR by `0.0219`, but hit@10 stays unchanged and top10
changed cases are balanced at one improvement and one regression.

Because the active blocker is top10 retrieval recall, a dev hit@10 delta of
`+0.0000` is not enough to move toward runtime defaultization or final-test
evaluation. The next train/dev-only route should move to the next Stage76
candidate: `section_bm25_doc_rollup_train_dev_probe`.

## Guard Checks

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
candidate_depth_covers_primary_top10: passed
stage75_source_report_is_stage75: passed
stage77_source_report_is_stage77: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
stage77_did_not_open_final_test_gate: passed
source_doc_ids_not_used_as_runtime_evidence: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

Additional local check:

```text
Select-String over the Stage78 JSON for raw-text field names and known raw text
snippets returned no matches.
```

Stage78 JSON SHA256:

```text
6EE3DAE47AA3927AA4CF613A0DDC1C0CD5CDA6D4C62FE21C615AC38883CF6FAD
```

## Visualizations

```text
artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_train_hit_at_10.svg
artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_dev_hit_at_10.svg
artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_dev_top10_changes.svg
```

Visualization SHA256:

```text
stage78_fielded_bm25_train_hit_at_10.svg: 10841CF5C09A1FF1644029F6473536F44CA357D6732FCDD5BA6B47460D8F38E4
stage78_fielded_bm25_dev_hit_at_10.svg: 6D47D3FF6B21F1C0238FD045DD5170A8D086E36D59D04CA0E764235B84912161
stage78_fielded_bm25_dev_delta_hit_at_10.svg: 416D7A934B35A9BED2425CECAAD18F6BC5E63511B026366618F90C49376FB326
stage78_fielded_bm25_dev_top10_changes.svg: 9B50C9AAD56E42620AB23D2F0FE401BAD3905F4D3351F062E283F477AD3F6750
```

## Validation

```text
ruff check .: passed
pytest -q: 205 passed
git diff --check: passed
raw-text Select-String check over Stage78 JSON: no matches
artifact git check-ignore: artifacts ignored by .gitignore
```

## Decision

```text
status: primeqa_hybrid_fielded_bm25_fusion_completed
train_selected_config_id: fielded_title_0_25_text_1_00
train_selected_dev_hit10_delta: 0.0
train_selected_dev_top10_improvements: 1
train_selected_dev_top10_regressions: 1
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 79 should move to the next Stage76 candidate:
`section_bm25_doc_rollup_train_dev_probe`.

Stage 79 must remain train/dev-only, keep the frozen test split locked, avoid
source `DOC_IDS` as runtime retrieval evidence, and not run final test metrics.
