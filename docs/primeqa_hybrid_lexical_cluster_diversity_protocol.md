# PrimeQA Hybrid Lexical Cluster Diversity Protocol

This document records Stage 85.

## Scope

Stage 85 freezes the train/dev-only protocol for the Stage84 recommended
candidate:

```text
lexical_cluster_diversity_rerank_design
```

This stage reads the public-safe Stage84 design report. It does not run
retrieval metrics, does not load the frozen test split, does not run final
metrics, does not use source `DOC_IDS` as runtime retrieval evidence, and does
not change runtime defaults.

The candidate was explicitly confirmed for this run:

```text
confirmed: true
confirmed_candidate_id: lexical_cluster_diversity_rerank_design
confirmation_note: user confirmed recommended Stage85 candidate in current turn
```

## Command

```text
python scripts\freeze_primeqa_hybrid_lexical_cluster_diversity_protocol.py ^
  --user-confirmed-candidate ^
  --confirmed-candidate-id lexical_cluster_diversity_rerank_design ^
  --confirmation-note "user confirmed recommended Stage85 candidate in current turn" ^
  --output artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85.json ^
  --visualization-dir artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals
```

The run completed in `0.000s`.

## Frozen Protocol

```text
protocol_id: lexical_cluster_diversity_rerank_train_dev_v1
candidate_id: lexical_cluster_diversity_rerank_design
protocol_status: frozen_requires_user_confirmation_before_metric_run
source_stage: Stage 84
```

Baseline retriever:

```text
config_id: full_document_bm25_baseline
bm25_k1: 1.5
bm25_b: 0.75
candidate_depth: 50
primary_top_k: 10
```

Candidate config grid:

| Config | Duplicate penalty | Cluster key | Minimum title overlap terms | Minimum cluster size |
| --- | ---: | --- | ---: | ---: |
| lcdr_penalty_0_03_title_query_cluster | 0.03 | title_query_overlap_hash | 3 | 2 |
| lcdr_penalty_0_06_title_query_cluster | 0.06 | title_query_overlap_hash | 3 | 2 |
| lcdr_penalty_0_09_title_query_cluster | 0.09 | title_query_overlap_hash | 3 | 2 |
| lcdr_penalty_0_12_title_query_cluster | 0.12 | title_query_overlap_hash | 3 | 2 |

Rerank formula:

```text
cluster_duplicate_index:
  Count prior candidates in the same lexical_cluster_hash when traversing
  baseline BM25 order.

adjusted_score:
  baseline_bm25_score
  - duplicate_penalty_weight * top1_bm25_score * cluster_duplicate_index

tie_breakers:
  higher baseline_bm25_score
  lower baseline_bm25_rank
  lower stable document id sort key
```

No alternate retriever or replacement behavior is included outside the frozen
config grid.

## Feature Contract

Runtime-allowed feature groups:

```text
query_features:
  query_token_count
  query_unique_token_count

candidate_rank_score_features:
  baseline_bm25_rank
  baseline_bm25_score
  score_margin_to_top1
  score_margin_to_previous

candidate_overlap_features:
  query_overlap_count
  title_query_overlap_count
  document_token_count

cluster_features:
  title_query_overlap_hash
  lexical_cluster_hash
  cluster_duplicate_index
  cluster_size_in_candidate_depth
```

Cluster hash contract:

```text
hash_input: sorted normalized title tokens that also appear in the normalized runtime query
hash_algorithm: sha256
hash_length: 16
raw_tokens_written_to_report: false
```

Prohibited runtime features:

```text
source_DOC_IDS
answer_doc_id
gold_document_rank
gold_label
frozen_test_split_membership
```

Public-safe changed-case fields:

```text
sample_id
split
baseline_rank
challenger_rank
baseline_cluster_duplicate_index
challenger_cluster_duplicate_index
config_id
```

## Selection Rule

```text
selection_split: train
validation_split: dev
rule: Select the candidate config on train by hit@10, then hit@5, hit@1,
      MRR@10, fewer top10 regressions, fewer rank-down within top10, then
      config_id. Dev is validation only.
dev_selection_forbidden: true
test_selection_forbidden: true
```

## Guard Checks

All `15 / 15` guard checks passed:

```text
source_report_is_stage84: passed
user_confirmed_recommended_candidate: passed
confirmed_candidate_matches_stage84_recommendation: passed
candidate_is_recommended_for_protocol_design: passed
stage84_requires_confirmation_before_train_dev_run: passed
stage84_final_test_metrics_locked: passed
stage84_forbids_test_tuning: passed
stage84_runtime_default_unchanged: passed
protocol_id_is_fixed: passed
candidate_config_grid_is_predeclared: passed
source_doc_ids_forbidden_in_runtime_features: passed
report_fields_are_public_safe: passed
stage85_freezes_protocol_without_metrics: passed
stage85_final_test_metrics_not_run: passed
stage85_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals\stage85_lcdr_candidate_config_penalties.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals\stage85_lcdr_feature_group_counts.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals\stage85_lcdr_protocol_decision_flags.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals\stage85_lcdr_guard_check_status.svg
```

Stage85 JSON SHA256:

```text
A8A1AF1F38937EDDD87F146DE320C2DF45E798FF96ACBDE45A2E0F386303D814
```

Visualization SHA256:

```text
stage85_lcdr_candidate_config_penalties.svg: B83480EF8CC585DAB21D9B87B5343221DA959289539D86206AF69A46193B113F
stage85_lcdr_feature_group_counts.svg: 778FA393FFA1DFFF7B666B7A13CEE44C71B762CB149ED0B8AD280650A53B0A75
stage85_lcdr_guard_check_status.svg: 1F4FF15112C7C69DD7AD769BEFC83C8AD9C54DB62F073AAB2489F7112BF8B41E
stage85_lcdr_protocol_decision_flags.svg: 80187FC905E89456229381D869F64B8E46CA48C6EA7596B0E952C42534B0DFD6
```

## Validation

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_lexical_cluster_diversity_protocol.py scripts\freeze_primeqa_hybrid_lexical_cluster_diversity_protocol.py tests\test_primeqa_hybrid_lexical_cluster_diversity_protocol.py: passed
pytest -q tests\test_primeqa_hybrid_lexical_cluster_diversity_protocol.py: 3 passed
Select-String raw question / answer / document / snippet field patterns over Stage85 JSON: no matches after prohibited-field wording fix
git check-ignore Stage85 JSON and SVG artifacts: ignored by .gitignore
ruff check .: passed
pytest -q: 220 passed
git diff --check: passed
```

## Decision

```text
status: primeqa_hybrid_lexical_cluster_diversity_protocol_frozen
protocol_id: lexical_cluster_diversity_rerank_train_dev_v1
candidate_id: lexical_cluster_diversity_rerank_design
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_run_train_dev_metrics_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage86 has run the frozen train/dev-only lexical cluster diversity rerank
comparison after user confirmation. It selected
`lcdr_penalty_0_06_title_query_cluster` on train, but the selected config had
`dev hit@10 delta = 0.0000`, with zero dev top10 improvements and zero dev
top10 regressions.

The current next step is Stage87: stop lexical cluster diversity as a
retrieval-recall route unless a new train/dev-only protocol is explicitly
confirmed, then move to the next confirmed second-wave candidate. The frozen
test split remains locked, final metrics must not be run, source `DOC_IDS` must
not be used as runtime retrieval evidence, and runtime defaults remain
unchanged.
