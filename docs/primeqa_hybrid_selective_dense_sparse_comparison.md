# PrimeQA Hybrid Selective Dense+Sparse Comparison

This document records Stage 98.

## Scope

Stage 98 runs the frozen train/dev-only comparison for
`selective_dense_sparse_low_overlap_gate_train_dev_v1` after the user confirmed
the Stage97 protocol.

This stage loads only the frozen Stage68 train/dev splits, the PrimeQA
training/dev document file, the Stage75 BM25 baseline report, the Stage97
frozen protocol, and the existing local dense caches. It does not load the
frozen test split, does not run final metrics, does not use source `DOC_IDS` as
runtime retrieval evidence, does not download models or refresh dense caches,
does not write raw question, answer, document, query-term, or matched-token
text, and does not change runtime defaults.

## Command

The first run wrote a complete report but returned a PowerShell
`NativeCommandError` wrapper because model-loading progress output went to
stderr while all streams were redirected. To get a clean process status, the run
was repeated with progress bars disabled and stdout/stderr captured separately:

```text
$env:HF_HUB_DISABLE_PROGRESS_BARS='1'
$env:TQDM_DISABLE='1'
python scripts\run_primeqa_hybrid_selective_dense_sparse_comparison.py ^
  --user-confirmed-protocol ^
  --confirmed-protocol-id selective_dense_sparse_low_overlap_gate_train_dev_v1 ^
  --confirmation-note "user confirmed Stage98 selective dense sparse comparison in current turn" ^
  --output artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98.json ^
  --visualization-dir artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals ^
  1> artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98.console.txt ^
  2> artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98.stderr.txt
```

The repeated command completed with exit code `0`. The stderr file was empty.

## Loaded Data

```text
document_count: 28482
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
test_split_loaded: false
final_metrics_run: false
```

## Frozen Policies

```text
sdsl_e5_low_overlap_balanced_v1
sdsl_minilm_low_overlap_balanced_v1
sdsl_e5_low_overlap_dense_bias_v1
sdsl_minilm_low_overlap_conservative_v1
```

Train selection rule:

```text
Select the gated dense+sparse policy on train only by hit@10, then larger
not-found@50 reduction, fewer top10 regressions, hit@1, MRR@10, lower dense
promotion budget, then policy_id. Dev is validation only.
```

## Train Metrics

Baseline:

```text
full_document_bm25_baseline:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  MRR@50: 0.5061
  not_found@50: 93
```

All frozen policies had identical train hit@10 to the baseline:

```text
sdsl_e5_low_overlap_balanced_v1: hit@10 0.6622, gate activations 1, promotions 2
sdsl_minilm_low_overlap_balanced_v1: hit@10 0.6622, gate activations 1, promotions 2
sdsl_e5_low_overlap_dense_bias_v1: hit@10 0.6622, gate activations 0, promotions 0
sdsl_minilm_low_overlap_conservative_v1: hit@10 0.6622, gate activations 2, promotions 2
```

Train-selected policy:

```text
sdsl_minilm_low_overlap_conservative_v1:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  MRR@50: 0.5061
  not_found@50: 93
  gate_activation_count: 2
  promotion_count: 2
```

Train-selected comparison:

```text
hit@1_delta: +0.0000
hit@10_delta: +0.0000
top10_improvement_count: 0
top10_regression_count: 0
not_found_count_at_50_delta: 0
```

## Dev Metrics

Baseline:

```text
full_document_bm25_baseline:
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  MRR@10: 0.5331
  MRR@50: 0.5365
  not_found@50: 17
```

All frozen policies had identical dev metrics to the baseline and no dev gate
activation:

```text
sdsl_e5_low_overlap_balanced_v1: hit@10 0.6974, not_found@50 17, gate activations 0, promotions 0
sdsl_minilm_low_overlap_balanced_v1: hit@10 0.6974, not_found@50 17, gate activations 0, promotions 0
sdsl_e5_low_overlap_dense_bias_v1: hit@10 0.6974, not_found@50 17, gate activations 0, promotions 0
sdsl_minilm_low_overlap_conservative_v1: hit@10 0.6974, not_found@50 17, gate activations 0, promotions 0
```

Train-selected dev comparison:

```text
selected_policy_id: sdsl_minilm_low_overlap_conservative_v1
hit@1_delta: +0.0000
hit@10_delta: +0.0000
top10_improvement_count: 0
top10_regression_count: 0
not_found_count_at_50_delta: 0
gate_activation_count: 0
promotion_count: 0
```

## Decision

```text
status: primeqa_hybrid_selective_dense_sparse_comparison_completed
selected_policy_id: sdsl_minilm_low_overlap_conservative_v1
selected_dev_hit10_delta: 0.0000
selected_dev_hit1_delta: 0.0000
selected_dev_top10_improvements: 0
selected_dev_top10_regressions: 0
selected_dev_not_found_at_search_depth_delta: 0
selected_dev_gate_activation_count: 0
selected_dev_promotion_count: 0
primary_contract_passed: false
secondary_contract_passed: false
guard_contract_passed: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

The route does not advance. The selected policy produced no dev gate actions,
no dev hit@10 gain, and no dev not-found@50 reduction. This fails the frozen
Stage97 primary and secondary metric contracts, so the next stage should stop
and summarize the selective dense+sparse route while keeping the test split
locked.

## Guard Checks

All `23 / 23` guard checks passed:

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_matches_stage97_baseline_candidate_depth: passed
stage75_source_report_is_stage75: passed
stage97_source_report_is_stage97: passed
stage97_protocol_is_frozen: passed
user_confirmed_stage98_train_dev_run: passed
confirmed_protocol_id_matches_stage97: passed
stage97_requires_user_confirmation: passed
dense_cache_count_matches_stage97_protocol: passed
policy_grid_count_matches_stage97_protocol: passed
dense_configs_present_for_all_policies: passed
dense_caches_preflight_passed: passed
no_model_download_attempted: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
train_selection_uses_train_only: passed
selected_policy_has_dev_validation: passed
source_doc_ids_not_used_as_runtime_evidence: passed
answer_doc_ids_used_only_for_metric_scoring: passed
final_test_metrics_not_run: passed
test_split_not_loaded: passed
default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_train_hit_at_10.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_dev_hit_at_10.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_dev_hit10_delta.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_dev_not_found_delta.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_dev_promotions.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_guard_check_status.svg
```

Stage98 JSON SHA256:

```text
217BEB28AF66AAB2EE3C5C85E788277ED5DE54982A05F1065CA5D7BF1110030E
```

Visualization SHA256:

```text
stage98_selective_dense_sparse_dev_hit10_delta.svg: B6E448601DDE0AFDD472B2AF7BC868C156B036519547F59697019F3DC02D06E7
stage98_selective_dense_sparse_dev_hit_at_10.svg: 2B75EDC18AB57C1002C4472C1E86D7F53A282CDAA154DCA1BC43750241521878
stage98_selective_dense_sparse_dev_not_found_delta.svg: 0F4190FEDF6A4CF1B6C871DE3DC3EDAC3D8479EC368EE941C7F735DDEE29816C
stage98_selective_dense_sparse_dev_promotions.svg: 17DCF744D669612B87A7135CD9883F5ACEC8CD073C440E5C9D0464CDB1B07314
stage98_selective_dense_sparse_guard_check_status.svg: 885F59AF8E86414DF5014AFC9DA645C7A654688B97751B2919ADCE97503E40BF
stage98_selective_dense_sparse_train_hit_at_10.svg: ECD74BC3A9A94E9EA4EE18A5D1F492D661AF9A81F0A8B8B7AB690E6506F06A77
```

## Verification

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_selective_dense_sparse_comparison.py scripts\run_primeqa_hybrid_selective_dense_sparse_comparison.py tests\test_primeqa_hybrid_selective_dense_sparse_comparison.py
pytest -q tests\test_primeqa_hybrid_selective_dense_sparse_comparison.py
python scripts\run_primeqa_hybrid_selective_dense_sparse_comparison.py ...: exit code 0
```

Result:

```text
ruff: passed
pytest: 3 passed
Stage98 run: passed
stderr: empty
```

Full validation:

```text
ruff check .: passed
pytest -q: 260 passed
git diff --check: passed
```

Artifact safety scan:

```text
Select-String Stage98 JSON for question_text/question_title/answer_doc_id/DOC_IDS/query_terms/matched_token_strings: no matches
artifact_safety.raw_question_text_written: false
artifact_safety.raw_answer_text_written: false
artifact_safety.raw_document_text_written: false
artifact_safety.source_doc_ids_used_as_runtime_evidence: false
```

## Conclusion

- Stage98 completed the frozen train/dev-only comparison.
- Stage98 did not load test and did not run final metrics.
- Stage98 did not download models or refresh dense caches.
- Stage98 did not use source `DOC_IDS` as runtime retrieval evidence.
- Stage98 did not change runtime defaults.
- The selected policy did not improve dev hit@10 and did not reduce dev
  not-found@50.
- The next stage should stop the selective dense+sparse route and keep the test
  split locked.
