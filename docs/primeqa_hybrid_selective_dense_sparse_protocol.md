# PrimeQA Hybrid Selective Dense+Sparse Protocol

This document records Stage 97.

## Scope

Stage 97 confirms and freezes the train/dev-only protocol for
`selective_dense_sparse_low_overlap_gate_design` after Stage96 stopped the
score-margin BM25 normalization route.

This is a protocol-freeze checkpoint. It reads only public-safe Stage84,
Stage96, Stage80, and Stage81 reports. It does not load train/dev/test split
files, does not run retrieval metrics, does not run final metrics, does not
download models, does not use source `DOC_IDS` as runtime retrieval evidence,
and does not change runtime defaults.

The candidate was explicitly confirmed for this run:

```text
confirmed_candidate_id: selective_dense_sparse_low_overlap_gate_design
confirmed: true
confirmation_note: user confirmed Stage97 selective dense sparse protocol freeze in current turn
```

## Command

```text
python scripts\freeze_primeqa_hybrid_selective_dense_sparse_protocol.py ^
  --user-confirmed-candidate ^
  --confirmed-candidate-id selective_dense_sparse_low_overlap_gate_design ^
  --confirmation-note "user confirmed Stage97 selective dense sparse protocol freeze in current turn" ^
  --output artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97.json ^
  --visualization-dir artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals
```

Console output was also captured in:

```text
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97.console.txt
```

## Frozen Protocol

```text
protocol_id: selective_dense_sparse_low_overlap_gate_train_dev_v1
candidate_id: selective_dense_sparse_low_overlap_gate_design
protocol_status: frozen_requires_user_confirmation_before_metric_run
source_stages: Stage 84, Stage 96, Stage 80, Stage 81
development_splits: train, dev
forbidden_final_splits: test
```

Baseline retriever:

```text
config_id: full_document_bm25_baseline
bm25_k1: 1.5
bm25_b: 0.75
candidate_depth: 50
primary_top_k: 10
```

Dense cache contract:

```text
allowed_cache_source: stage80_compatible_local_dense_caches_only
download_required: false
document_reencoding_allowed: false
query_encoding_mode: local_snapshot_path_with_local_files_only
model_selection_mode: predeclared_grid_then_train_selection_only
```

Allowed local dense configs:

```text
dense_sparse_rrf__intfloat_e5_small_v2__512_passage
  model: intfloat/e5-small-v2
  cache: data\indexes\dense\intfloat__e5-small-v2_512_passage.npz
  query_prefix: "query: "
  document_prefix: "passage: "

dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
  model: sentence-transformers/all-MiniLM-L6-v2
  cache: data\indexes\dense\sentence-transformers__all-MiniLM-L6-v2_1600.npz
  query_prefix: ""
  document_prefix: ""
```

## Candidate Policy Grid

```text
sdsl_e5_low_overlap_balanced_v1
  dense_config_id: dense_sparse_rrf__intfloat_e5_small_v2__512_passage
  gate_mode: low_bm25_lexical_overlap
  minimum_query_token_count: 8
  max BM25 top1 query-overlap ratio: 0.25
  max BM25 top10 mean query-overlap ratio: 0.22
  dense_candidate_rank_max: 10
  sparse_weight: 1.00
  dense_weight: 1.00
  max dense top10 promotions per query: 2
  protected BM25 top ranks: 5

sdsl_minilm_low_overlap_balanced_v1
  dense_config_id: dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
  gate_mode: low_bm25_lexical_overlap
  minimum_query_token_count: 8
  max BM25 top1 query-overlap ratio: 0.25
  max BM25 top10 mean query-overlap ratio: 0.22
  dense_candidate_rank_max: 10
  sparse_weight: 1.00
  dense_weight: 1.00
  max dense top10 promotions per query: 2
  protected BM25 top ranks: 5

sdsl_e5_low_overlap_dense_bias_v1
  dense_config_id: dense_sparse_rrf__intfloat_e5_small_v2__512_passage
  gate_mode: strict_low_overlap_dense_bias
  minimum_query_token_count: 10
  max BM25 top1 query-overlap ratio: 0.20
  max BM25 top10 mean query-overlap ratio: 0.18
  dense_candidate_rank_max: 8
  sparse_weight: 1.00
  dense_weight: 1.25
  max dense top10 promotions per query: 2
  protected BM25 top ranks: 5

sdsl_minilm_low_overlap_conservative_v1
  dense_config_id: dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
  gate_mode: conservative_low_overlap
  minimum_query_token_count: 6
  max BM25 top1 query-overlap ratio: 0.30
  max BM25 top10 mean query-overlap ratio: 0.25
  dense_candidate_rank_max: 8
  sparse_weight: 1.00
  dense_weight: 0.85
  max dense top10 promotions per query: 1
  protected BM25 top ranks: 7
```

Every policy requires the dense promotion candidate to be outside the BM25
top10. The grid is frozen before Stage98; Stage98 may select among these
policies on train only.

## Feature Contract

Runtime-allowed feature groups:

```text
query_aggregate_features:
  query_token_count
  query_unique_token_count
  query_length_bucket

bm25_lexical_features:
  bm25_top1_query_overlap_count
  bm25_top1_query_overlap_ratio
  bm25_top10_mean_query_overlap_ratio
  candidate_query_overlap_count
  candidate_query_overlap_ratio
  candidate_title_query_overlap_count
  candidate_title_query_overlap_ratio

sparse_rank_score_features:
  bm25_rank
  bm25_score
  bm25_rank_bucket
  bm25_score_margin_to_rank10

dense_rank_score_features:
  dense_config_id
  dense_rank
  dense_score
  dense_rank_bucket

rrf_gate_features:
  rrf_rank
  rrf_score
  sparse_rrf_contribution
  dense_rrf_contribution
  dense_sparse_contribution_ratio

action_budget_features:
  dense_top10_promotion_budget_remaining
  protected_bm25_top_rank_count
  gate_activation_reason_code
```

Explicitly prohibited runtime features:

```text
source_DOC_IDS
answer document IDs
gold_document_rank
gold_label
dev_selected_gate_threshold
dev_selected_dense_model
stage81_dev_selected_config
frozen_test_split_membership
raw_question_text
raw_answer_text
raw_document_text
raw_document_title
query_terms
matched_token_strings
```

## Train Selection Rule

```text
Select the gated dense+sparse policy on train only by hit@10, then larger
not-found@50 reduction, fewer top10 regressions, hit@1 non-collapse, MRR@10,
lower dense promotion budget, then policy_id. Dev is validation only.
```

Forbidden:

```text
dev threshold selection
test selection
Stage81 dev result selection
source DOC_IDS or answer document ID features
model downloads
dense cache refresh
runtime default changes
```

## Decision

```text
status: primeqa_hybrid_selective_dense_sparse_protocol_frozen
protocol_id: selective_dense_sparse_low_overlap_gate_train_dev_v1
candidate_id: selective_dense_sparse_low_overlap_gate_design
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_run_train_dev_metrics_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Guard Checks

All `46 / 46` guard checks passed.

Important checks:

```text
source_stage84_report_is_stage84: passed
source_stage96_report_is_stage96: passed
source_stage80_report_is_stage80: passed
source_stage81_report_is_stage81: passed
user_confirmed_selective_dense_sparse_protocol: passed
stage96_stopped_score_margin_bm25_route: passed
confirmed_candidate_matches_stage96_next_candidate: passed
stage80_can_run_without_download: passed
stage80_compatible_cache_count_at_least_two: passed
stage81_dense_sparse_comparison_completed: passed
stage84_candidate_contract_requires_train_selected_dev_hit10_gain: passed
stage84_candidate_contract_requires_not_found_decrease: passed
stage84_candidate_guard_blocks_downloads_and_dev_thresholds: passed
dense_cache_contract_uses_stage80_and_stage81_only: passed
stage80_stage81_dense_cache_identities_match: passed
candidate_policy_grid_is_predeclared: passed
candidate_policy_grid_reuses_existing_dense_configs: passed
candidate_policy_grid_has_low_overlap_gates: passed
train_selection_rule_forbids_dev_and_test_selection: passed
source_doc_ids_forbidden_in_runtime_features: passed
answer_doc_ids_forbidden_in_runtime_features: passed
downloads_and_cache_refresh_forbidden: passed
stage97_freezes_protocol_without_metrics: passed
stage97_final_test_metrics_not_run: passed
stage97_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_cache_readiness.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_gate_thresholds.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_rrf_weights.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_feature_group_counts.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_protocol_decision_flags.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_guard_check_status.svg
```

Stage97 JSON SHA256:

```text
61041D5C3CC2E862F71041D40106845BC3F468E9C2BC0C6E8EF6B7BB659640E4
```

Visualization SHA256:

```text
stage97_selective_dense_sparse_cache_readiness.svg: 189E1C73FB9C8F8DAC88EE9A2650D15E849B0F815B6C56AE4E39A2290D0417E9
stage97_selective_dense_sparse_feature_group_counts.svg: A760D77B693B4C6970A07FA1C157143EBC144D7FC5DF12FBB323B5D4EB4A5A54
stage97_selective_dense_sparse_gate_thresholds.svg: 501DECAA5DF00D9C341ECF6D0C0E316EEA8F8E1F5F9D610F0128906C46055AB2
stage97_selective_dense_sparse_guard_check_status.svg: 7F9F286DDC7B3EB9E1A19D850932EF3BBCE929E7B8199CA5AD14B6A15D1E920F
stage97_selective_dense_sparse_protocol_decision_flags.svg: 1E71BB23C2F8BB2BC167793EDFFD691CB2A2922617E5A33839A98A7BB8B6A253
stage97_selective_dense_sparse_rrf_weights.svg: 2AB4F308DF793121490FE969D1368EB30CFFDB4464E636626E717A2F0E9F1D78
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_selective_dense_sparse_protocol.py scripts\freeze_primeqa_hybrid_selective_dense_sparse_protocol.py tests\test_primeqa_hybrid_selective_dense_sparse_protocol.py: passed
pytest -q tests\test_primeqa_hybrid_selective_dense_sparse_protocol.py: 4 passed
Select-String actual private snippets over Stage97 JSON: no matches
Select-String raw text value fields over Stage97 JSON: no matches
git check-ignore Stage97 JSON, console, and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 257 passed
git diff --check: passed
```

## Next Step

Stage98 should run the frozen train/dev-only selective dense+sparse low-overlap
gate comparison after user confirmation. The frozen test split remains locked,
final metrics must not be run, source `DOC_IDS` must not be used as runtime
retrieval evidence, no model downloads or cache refreshes are allowed, and
runtime defaults remain unchanged.
