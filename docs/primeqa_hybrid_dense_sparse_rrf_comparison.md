# PrimeQA Hybrid Dense Sparse RRF Comparison

This document records Stage 81.

## Scope

Stage 81 runs the user-confirmed Stage80 option:
`compare_existing_cached_dense_models`.

This is a train/dev-only retrieval-recall experiment for
`dense_sparse_rrf_train_dev_probe`. It compares the two compatible local dense
caches found in Stage80 against the full-document BM25 baseline. It selects the
dense+sparse RRF challenger on train only, validates the selected challenger on
dev, keeps the frozen test split locked, does not run final metrics, does not
use source `DOC_IDS` as runtime retrieval evidence, does not download models,
and does not change runtime defaults.

The report stores metrics, public-safe changed-case sample IDs and ranks,
cache/model identity, guard checks, and visualization paths. It does not output
raw question text, raw answer text, document titles, or document body text.

## Command

```text
$env:HF_HUB_OFFLINE='1'
python scripts\run_primeqa_hybrid_dense_sparse_rrf_comparison.py ^
  --output artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81.json ^
  --visualization-dir artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals
```

The actual run completed in `385.456s`.

## Compared Configurations

```text
baseline:
  full_document_bm25_baseline

dense+sparse RRF challengers:
  dense_sparse_rrf__intfloat_e5_small_v2__512_passage
  dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
```

Both dense challengers used existing local document-embedding caches and local
Hugging Face snapshots:

```text
intfloat/e5-small-v2
  cache: data\indexes\dense\intfloat__e5-small-v2_512_passage.npz
  document_text_max_chars: 512
  document_prefix: "passage: "
  query_prefix: "query: "
  snapshot: ffb93f3bd4047442299a41ebb6fa998a38507c52

sentence-transformers/all-MiniLM-L6-v2
  cache: data\indexes\dense\sentence-transformers__all-MiniLM-L6-v2_1600.npz
  document_text_max_chars: 1600
  document_prefix: ""
  query_prefix: ""
  snapshot: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

The query-prefix protocol came from Stage80 legacy dense metric records. If that
protocol had been missing, Stage81 would have blocked rather than silently
choosing a prefix.

## Train Metrics

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | not found @50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_document_bm25_baseline | 0.4243 | 0.6054 | 0.6622 | 0.5023 | 93 |
| dense_sparse_rrf__intfloat_e5_small_v2__512_passage | 0.4595 | 0.6270 | 0.6703 | 0.5294 | 63 |
| dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix | 0.4378 | 0.6514 | 0.6973 | 0.5249 | 55 |

Train-selected challenger:

```text
dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
```

Selection rule:

```text
Select among confirmed dense+sparse RRF challengers on train only by hit@10,
then hit@5, then hit@1, then MRR@10, then config_id. Dev is validation only.
```

## Dev Metrics

| Config | hit@1 | hit@5 | hit@10 | MRR@10 | not found @50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_document_bm25_baseline | 0.4342 | 0.6579 | 0.6974 | 0.5331 | 17 |
| dense_sparse_rrf__intfloat_e5_small_v2__512_passage | 0.5132 | 0.6316 | 0.6711 | 0.5659 | 10 |
| dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix | 0.3947 | 0.6316 | 0.6842 | 0.4998 | 11 |

Dev deltas versus baseline:

| Challenger | hit@10 delta | top10 improvements | top10 regressions | not found @50 delta |
| --- | ---: | ---: | ---: | ---: |
| dense_sparse_rrf__intfloat_e5_small_v2__512_passage | -0.0263 | 5 | 7 | -7 |
| dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix | -0.0132 | 3 | 4 | -6 |

The train-selected challenger improved train hit@10 by `+0.0351`, but dev
hit@10 regressed by `-0.0132`. The E5 challenger improved dev hit@1 and reduced
top50 not-found cases, but it also regressed dev hit@10 by `-0.0263`.

## Guard Checks

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
stage75_source_report_is_stage75: passed
stage80_source_report_is_stage80: passed
stage80_can_run_dense_sparse_rrf_without_download: passed
stage80_requires_user_confirmation_before_train_dev_run: passed
user_confirmed_protocol_matches_stage80_option: passed
selected_cache_count_matches_compare_protocol: passed
dense_cache_files_exist: passed
dense_cache_metadata_matches_stage80: passed
dense_cache_document_ids_match_current_corpus: passed
dense_cache_embedding_rows_match_current_corpus: passed
query_prefix_protocol_resolved_from_stage80: passed
local_model_snapshots_exist: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
no_model_download_attempted: passed
source_doc_ids_not_used_as_runtime_evidence: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

Stage81 JSON SHA256:

```text
56AFBE70545C05780631750EE39B9DBC2B41B26BFD310D5997F60448FB3B4C03
```

## Visualizations

```text
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_train_hit_at_10.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_dev_hit_at_10.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_dev_not_found_at_50.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_dev_top10_changes.svg
```

Visualization SHA256:

```text
stage81_dense_sparse_rrf_dev_delta_hit_at_10.svg: 9872FA6141CCDA90C5C813B59E76B8207070BE1D50CEBC2FB9461BB68A7D17CA
stage81_dense_sparse_rrf_dev_hit_at_10.svg: 1B5D61AD529DAFA37D3DB1043BFA10CD563C45140E619DCF5E255F4F769ED0A0
stage81_dense_sparse_rrf_dev_not_found_at_50.svg: 3B22099BC6040229F27DAB82BC788B6DF201FAF412AB2CF0AC5A98DA9A3D63AB
stage81_dense_sparse_rrf_dev_top10_changes.svg: F4A45D60332F1357B47EB95699C756E11B253D1CDEB96E549DF43F0DB1CFB326
stage81_dense_sparse_rrf_train_hit_at_10.svg: F252E610DB63E19EA41BA22DB6D7C644D052DEF94A74F99C138FF59039A177C1
```

## Decision

```text
status: primeqa_hybrid_dense_sparse_rrf_comparison_completed
selected_config_id: dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
selected_dev_hit10_delta: -0.0132
selected_dev_top10_improvements: 3
selected_dev_top10_regressions: 4
selected_dev_not_found_at_search_depth_delta: -6
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 82 should move to the remaining Stage76 candidate:
`bm25_k1_b_grid_train_to_dev`.

This next step must still use only train/dev, keep test locked, select by train,
validate on dev, avoid source `DOC_IDS` as runtime retrieval evidence, not run
final test metrics, and not change runtime defaults.
