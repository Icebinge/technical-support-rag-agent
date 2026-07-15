# PrimeQA Hybrid Answer-Pipeline Error Decomposition

This document records Stage 102.

## Scope

Stage 102 runs the Stage101 frozen train/dev-only answer-pipeline error
decomposition protocol.

This stage loads the Stage68 train/dev split rows and local PrimeQA corpus
documents to compute bucketed diagnostics. It does not load the test split, does
not run final metrics, does not write raw question, answer, document, token, or
document-identifier fields, does not add fallback strategies, and does not
change runtime defaults.

The analysis was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: user confirmed Stage102 train-dev answer-pipeline error decomposition in current turn
```

## Command

```text
python scripts\run_primeqa_hybrid_answer_pipeline_error_decomposition.py ^
  --user-confirmed-analysis ^
  --confirmation-note "user confirmed Stage102 train-dev answer-pipeline error decomposition in current turn" ^
  --output artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json ^
  --visualization-dir artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals
```

The first invocation exceeded the initial 120-second tool timeout while the
Python process kept running. It wrote a JSON report, but the console capture was
empty, so it was not used as the final recorded command result. The same command
was rerun with a longer timeout and returned:

```text
stage102_exit_code=0
```

Console output was written to:

```text
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102.console.txt
```

## Configuration

```text
diagnostic_profile: stage102_bm25_top10_bm25_sentence_mcpd3_topk_verified_evidence7
retriever: BM25
bm25_k1: 1.5
bm25_b: 0.75
retrieval_top_k: 10
answer_generator: extractive_sentence_baseline
evidence_selector: bm25_sentence
max_candidates_per_document: 3
composition_policy: top_k
max_sentences: 3
min_sentence_score: 2.0
answer_verifier: citation_and_evidence_gate
min_evidence_score: 7.0
max_citation_rank: 3
min_citations: 1
max_gold_window_sentences: 3
gold_span_gap_margin: 0.05
low_answer_f1_threshold: 0.2
sample_limit_per_bucket: 5
```

## Data

```text
documents: 28482
train rows: 562
train answerable: 370
train unanswerable: 192
dev rows: 121
dev answerable: 76
dev unanswerable: 45
```

Stage102 did not load the Stage68 test split.

## Bucket Counts

| Bucket | Train count | Train rate | Dev count | Dev rate |
| --- | ---: | ---: | ---: | ---: |
| answerability_false_answer | 180 | 0.3203 | 41 | 0.3388 |
| retrieval_context_miss | 125 | 0.2224 | 23 | 0.1901 |
| evidence_selection_miss | 67 | 0.1192 | 12 | 0.0992 |
| verification_over_refusal | 3 | 0.0053 | 0 | 0.0000 |
| gold_span_beats_selected_answer | 174 | 0.3096 | 41 | 0.3388 |
| low_overlap_gold_cited_answer | 0 | 0.0000 | 0 | 0.0000 |
| answer_supported_and_cited | 13 | 0.0231 | 4 | 0.0331 |

Top priority buckets:

```text
train:
1. answerability_false_answer: count 180, priority_score 279.00
2. gold_span_beats_selected_answer: count 174, priority_score 252.30
3. retrieval_context_miss: count 125, priority_score 168.75
4. evidence_selection_miss: count 67, priority_score 113.90

dev:
1. answerability_false_answer: count 41, priority_score 63.55
2. gold_span_beats_selected_answer: count 41, priority_score 59.45
3. retrieval_context_miss: count 23, priority_score 31.05
4. evidence_selection_miss: count 12, priority_score 20.40
```

## Metrics

Original answer metrics:

| Split | Gold citation rate | Answerable refusal | Unanswerable refusal | Average token F1 |
| --- | ---: | ---: | ---: | ---: |
| train | 0.4811 | 0.0000 | 0.0052 | 0.1996 |
| dev | 0.5395 | 0.0000 | 0.0000 | 0.1978 |

Verified answer metrics:

| Split | Gold citation rate | Answerable refusal | Unanswerable refusal | Average token F1 |
| --- | ---: | ---: | ---: | ---: |
| train | 0.4958 | 0.0459 | 0.0625 | 0.2017 |
| dev | 0.6029 | 0.1053 | 0.0889 | 0.2040 |

Answerable gold context:

```text
train: 245 / 370 = 0.6622
dev: 53 / 76 = 0.6974
```

Verification decisions:

```text
train: answered 533, refused 29
dev: answered 109, refused 12
```

Retrieval-rank buckets:

```text
train:
not_applicable: 192
not_found_top_k: 125
rank_1: 157
rank_2_to_3: 47
rank_4_to_5: 20
rank_6_to_10: 21

dev:
not_applicable: 45
not_found_top_k: 23
rank_1: 33
rank_2_to_3: 16
rank_4_to_5: 1
rank_6_to_10: 3
```

## Decision

```text
status: primeqa_hybrid_answer_pipeline_error_decomposition_completed
analysis_id: answer_pipeline_error_decomposition_train_dev_analysis_v1
train_top_bucket: answerability_false_answer
dev_top_bucket: answerability_false_answer
train_evidence_selection_miss: 67
dev_evidence_selection_miss: 12
train_answerability_false_answer: 180
dev_answerability_false_answer: 41
train_verified_average_token_f1: 0.2017
dev_verified_average_token_f1: 0.2040
recommended_next_direction: evidence_selection_and_answerability_candidate_design
requires_user_confirmation_before_next_protocol: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_stage: Stage103: design a train/dev-only candidate intervention from the Stage102 bucket evidence, prioritizing shared train/dev bottlenecks without using test or changing runtime defaults.
```

## Guard Checks

All `20 / 20` guard checks passed:

```text
stage101_source_is_expected_stage: passed
stage101_protocol_is_frozen: passed
stage101_protocol_id_matches: passed
user_confirmed_stage102_analysis: passed
only_train_dev_splits_loaded: passed
loaded_samples_are_train_dev_only: passed
test_split_not_loaded: passed
stage101_allows_train_dev_error_decomposition: passed
stage101_final_metrics_locked: passed
stage101_forbids_test_tuning: passed
stage101_runtime_default_unchanged: passed
stage101_fallback_strategies_disabled: passed
expected_bucket_order_present: passed
public_case_samples_use_stage101_fields: passed
public_case_samples_do_not_exceed_limit: passed
public_case_samples_exclude_forbidden_keys: passed
stage102_runs_train_dev_analysis_only: passed
stage102_final_test_metrics_not_run: passed
stage102_default_runtime_policy_unchanged: passed
stage102_fallback_strategies_not_added: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_bucket_counts_by_split.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_pipeline_stage_counts.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_answerability_bucket_counts.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_verified_metric_rates.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_public_case_sample_counts.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_guard_check_status.svg
```

Stage102 JSON SHA256:

```text
08F2CA0F31F1B79AD78619AA33F8C53B3BFA7C6CB68DAEC3013987D43549B7C6
```

Visualization SHA256:

```text
stage102_answerability_bucket_counts.svg: 7D9651F9F91C528A4F18545968DC706529EFC76774412AE70FA20BC7FAAE4297
stage102_bucket_counts_by_split.svg: 5507B0FF0612656D517550C7B7E0E168E0191B63E66A06304DDDBEDF3A0FCAFB
stage102_guard_check_status.svg: 26F1B6DFBD4B648FFAFBF3DF316D4DF5E85C0816BBC950B8CF6EE5D1F7176AEF
stage102_pipeline_stage_counts.svg: E444FB7F5E95191C3611E60FC9947028FFB44B75F2228D18ACD28795E025EB40
stage102_public_case_sample_counts.svg: B17BA9B59D125D96310AE9AF9D46D88EFD4F559F0CF407B9E029EF1C7A03135E
stage102_verified_metric_rates.svg: 0CE34354E426705B12166226EA619894769FBC1070B6CFD38C506C816DE7A3F6
```

## Validation

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_answer_pipeline_error_decomposition_analysis.py scripts\run_primeqa_hybrid_answer_pipeline_error_decomposition.py tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_analysis.py
pytest -q tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_analysis.py
python scripts\run_primeqa_hybrid_answer_pipeline_error_decomposition.py ...: stage102_exit_code=0
```

Result:

```text
ruff: passed
pytest: 4 passed
Stage102 run: passed
guard checks: 20 / 20 passed
```

Full validation:

```text
ruff check .: passed
pytest -q: 274 passed
git diff --check: passed
```

Artifact safety scan:

```text
Select-String Stage102 JSON for raw question / answer / document id / token field patterns: no matches
```

## Conclusion

- Stage102 completed the train/dev-only answer-pipeline error decomposition.
- Stage102 did not load the test split and did not run final metrics.
- Stage102 did not add fallback strategies and did not change runtime defaults.
- The largest shared train/dev bottleneck is `answerability_false_answer`.
- The next largest shared bottleneck is `gold_span_beats_selected_answer`.
- Stage103 should design a train/dev-only intervention candidate from these
  shared bottlenecks before any further metric run.
