# PrimeQA Hybrid BM25 k1/b Grid

This document records Stage 82.

## Scope

Stage 82 runs the user-confirmed small-grid option for the final Stage76
retrieval-recall candidate: `bm25_k1_b_grid_train_to_dev`.

The confirmed grid is:

```text
k1: 1.2, 1.5, 1.8
b: 0.55, 0.75, 0.95
```

This is a train/dev-only experiment. It selects across the fixed grid on train,
validates the selected setting on dev, keeps the frozen test split locked, does
not run final metrics, does not use source `DOC_IDS` as runtime retrieval
evidence, and does not change runtime defaults.

The report stores metrics, public-safe changed-case sample IDs and ranks, guard
checks, and visualization paths. It does not output raw question text, raw
answer text, document titles, or document body text.

## Command

```text
python scripts\run_primeqa_hybrid_bm25_k1_b_grid.py ^
  --output artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82.json ^
  --visualization-dir artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals
```

The first Stage82 attempt timed out after about `604s` because it rebuilt BM25
indexes repeatedly. That timed-out run did not produce a report and was not
treated as completed. The implementation was then changed to use one shared
BM25 grid index per split.

The second attempt produced a blocked report because one guard required the
internal candidate ID in Stage81's next-step text, while the Stage81 report used
the human-readable phrase `BM25 k1/b grid candidate`. That report was
overwritten after the guard was corrected to accept both forms.

The final successful run completed in `93.445s`.

## Grid

| Config | k1 | b | Baseline |
| --- | ---: | ---: | --- |
| bm25_grid__k1_1_20__b_0_55 | 1.2 | 0.55 | no |
| bm25_grid__k1_1_20__b_0_75 | 1.2 | 0.75 | no |
| bm25_grid__k1_1_20__b_0_95 | 1.2 | 0.95 | no |
| bm25_grid__k1_1_50__b_0_55 | 1.5 | 0.55 | no |
| full_document_bm25_baseline | 1.5 | 0.75 | yes |
| bm25_grid__k1_1_50__b_0_95 | 1.5 | 0.95 | no |
| bm25_grid__k1_1_80__b_0_55 | 1.8 | 0.55 | no |
| bm25_grid__k1_1_80__b_0_75 | 1.8 | 0.75 | no |
| bm25_grid__k1_1_80__b_0_95 | 1.8 | 0.95 | no |

Selection rule:

```text
Select across the fixed user-confirmed BM25 small grid on train only by hit@10,
then hit@5, then hit@1, then MRR@10, then fewer not-found@50, then fewer rank
11-50 near misses, then config_id. Dev is validation only.
```

## Train Metrics

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | not found @50 | rank 11-50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_document_bm25_baseline | 0.4243 | 0.6054 | 0.6622 | 0.5023 | 93 | 32 |
| bm25_grid__k1_1_20__b_0_75 | 0.4270 | 0.6027 | 0.6622 | 0.5057 | 88 | 37 |
| bm25_grid__k1_1_20__b_0_95 | 0.4216 | 0.6108 | 0.6595 | 0.5009 | 92 | 34 |
| bm25_grid__k1_1_80__b_0_75 | 0.4216 | 0.6081 | 0.6595 | 0.4991 | 91 | 35 |
| bm25_grid__k1_1_20__b_0_55 | 0.4243 | 0.5946 | 0.6568 | 0.5044 | 88 | 39 |
| bm25_grid__k1_1_50__b_0_95 | 0.4216 | 0.5973 | 0.6514 | 0.4953 | 92 | 37 |
| bm25_grid__k1_1_80__b_0_55 | 0.4216 | 0.6027 | 0.6486 | 0.5012 | 83 | 47 |
| bm25_grid__k1_1_50__b_0_55 | 0.4243 | 0.6000 | 0.6432 | 0.5045 | 88 | 44 |
| bm25_grid__k1_1_80__b_0_95 | 0.4189 | 0.5892 | 0.6432 | 0.4945 | 93 | 39 |

Train-selected config:

```text
full_document_bm25_baseline
```

`bm25_grid__k1_1_20__b_0_75` tied baseline on train hit@10, but it lost on the
next train tie-breaker, hit@5. Therefore the train-only selection remains the
baseline.

## Dev Metrics

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | not found @50 | rank 11-50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_grid__k1_1_20__b_0_95 | 0.3947 | 0.6579 | 0.7105 | 0.5008 | 18 | 4 |
| bm25_grid__k1_1_50__b_0_95 | 0.3816 | 0.6579 | 0.7105 | 0.4977 | 18 | 4 |
| bm25_grid__k1_1_80__b_0_55 | 0.4868 | 0.6711 | 0.6974 | 0.5644 | 17 | 6 |
| bm25_grid__k1_1_50__b_0_55 | 0.4737 | 0.6711 | 0.6974 | 0.5534 | 17 | 6 |
| bm25_grid__k1_1_20__b_0_55 | 0.4605 | 0.6711 | 0.6974 | 0.5429 | 17 | 6 |
| bm25_grid__k1_1_80__b_0_75 | 0.4474 | 0.6579 | 0.6974 | 0.5419 | 17 | 6 |
| bm25_grid__k1_1_20__b_0_75 | 0.4474 | 0.6579 | 0.6974 | 0.5373 | 17 | 6 |
| full_document_bm25_baseline | 0.4342 | 0.6579 | 0.6974 | 0.5331 | 17 | 6 |
| bm25_grid__k1_1_80__b_0_95 | 0.3816 | 0.6316 | 0.6974 | 0.4933 | 18 | 5 |

Important boundary:

```text
The two b=0.95 configs have dev hit@10 = 0.7105, but they were not selected on
train. Using them would be dev-set selection and is not allowed.
```

## Guard Checks

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
stage75_source_report_is_stage75: passed
stage76_source_report_is_stage76: passed
stage76_bm25_grid_candidate_is_allowed: passed
stage76_requires_fixed_grid_values: passed
stage81_source_report_is_stage81: passed
stage81_did_not_open_final_test_gate: passed
stage81_recommends_bm25_grid_next: passed
user_confirmed_small_grid_protocol: passed
grid_values_fixed_before_run: passed
grid_includes_stage75_baseline: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
source_doc_ids_not_used_as_runtime_evidence: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

Stage82 JSON SHA256:

```text
EB9E0D5EC66401418E6254381A9638316EDE0A040156B0EF49C95CF6BDD786CA
```

## Visualizations

```text
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_train_hit_at_10.svg
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_dev_hit_at_10.svg
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_dev_near_miss_11_to_50.svg
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_dev_top10_changes.svg
```

Visualization SHA256:

```text
stage82_bm25_grid_dev_delta_hit_at_10.svg: 89A35B0B6D16EA6DD7231CDE8EB4EAFA40EBED4453EA8C253CBDEEAC15782E5C
stage82_bm25_grid_dev_hit_at_10.svg: 69B1229C2C4CB299781FE23C5CC2B7572EA9FF38E31C89E45C8D0B2683CAEF60
stage82_bm25_grid_dev_near_miss_11_to_50.svg: 4872DAE7FC8CBB5C91BEE3E6B67E20152BE65C47E0B77E8FD808DC427363C106
stage82_bm25_grid_dev_top10_changes.svg: 60EF49BEA97356A7FAFE7C4AE2A18449D538139AC374B1F8FF1C1AFEE4D1FE32
stage82_bm25_grid_train_hit_at_10.svg: C6240640E914101E5670EE4CA1133AA36726C0B34E55478CE4766743D0EC7C93
```

## Decision

```text
status: primeqa_hybrid_bm25_k1_b_grid_completed
selected_config_id: full_document_bm25_baseline
selected_dev_hit10_delta: 0.0
selected_dev_top10_improvements: 0
selected_dev_top10_regressions: 0
selected_dev_not_found_at_search_depth_delta: 0
selected_dev_rank_11_to_50_count_delta: 0
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage83 should summarize the exhausted Stage76 retrieval-recall candidates and
decide the next train/dev-only improvement route. The frozen test split remains
locked, final metrics must not be run, and runtime defaults must remain
unchanged unless a later train/dev gate justifies a guarded runtime experiment.
