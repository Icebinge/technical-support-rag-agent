# PrimeQA Hybrid Answer-Pipeline Error Decomposition Protocol

This document records Stage 101.

## Scope

Stage 101 freezes a train/dev-only protocol for answer-pipeline error
decomposition after Stage100 concluded that the first-wave and second-wave
retrieval route families are exhausted.

This stage reads only the public-safe Stage100 route-exhaustion summary. It does
not load train/dev/test split files, does not run retrieval metrics, does not run
answer metrics, does not run final metrics, does not use document identifiers as
runtime evidence, does not add fallback strategies, and does not change runtime
defaults.

The protocol was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: user confirmed Stage101 answer-pipeline error decomposition protocol in current turn
```

## Command

```text
python scripts\freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py ^
  --user-confirmed-protocol ^
  --confirmation-note "user confirmed Stage101 answer-pipeline error decomposition protocol in current turn" ^
  --output artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101.json ^
  --visualization-dir artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals
```

The run completed successfully and wrote console output to:

```text
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101.console.txt
```

## Stage100 Premise

```text
decision_status: primeqa_hybrid_second_wave_route_exhaustion_summary_completed
recommended_next_direction: answer_pipeline_error_decomposition
first_wave_retrieval_candidates_exhausted: true
second_wave_retrieval_route_family_exhausted: true
second_wave_expected_candidate_count: 5
second_wave_stopped_candidate_count: 5
runtime_advancing_second_wave_candidate_count: 0
remaining_actionable_candidate_count: 0
best_second_wave_dev_hit10_delta: 0.0000
best_second_wave_top10_net: 0
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Frozen Protocol

```text
protocol_id: answer_pipeline_error_decomposition_train_dev_v1
protocol_status: frozen_requires_user_confirmation_before_analysis_run
analysis_mode: protocol_freeze_only
fallback_strategies_enabled: false
```

Stage101 defines the next analysis contract. It does not run the analysis.

### Bucket Precedence

| Bucket | Pipeline stage | Precedence | Priority weight | Candidate intervention family |
| --- | --- | ---: | ---: | --- |
| answerability_false_answer | answerability | 1 | 1.55 | answerability_or_evidence_sufficiency_gate |
| retrieval_context_miss | retrieval | 2 | 1.35 | retrieval_only_if_new_signal_exists |
| evidence_selection_miss | evidence_selection | 3 | 1.70 | evidence_selector_or_reranker |
| verification_over_refusal | verification | 4 | 1.20 | verification_threshold_or_calibration |
| gold_span_beats_selected_answer | answer_composition | 5 | 1.45 | answer_composition_or_span_selection |
| low_overlap_gold_cited_answer | answer_composition | 6 | 1.10 | answer_synthesis_or_sentence_windowing |
| answer_supported_and_cited | non_error_reference | 7 | 0.25 | no_fix_reference_slice |

The assignment rule is single-label and deterministic: first matching bucket by
the frozen precedence order.

### Public-Safe Case Fields

Stage102 may write sanitized case samples only with these fields:

```text
sample_id
split
answerability_label
pipeline_bucket_id
pipeline_stage
retrieval_rank_bucket
retrieval_context_status
citation_status
evidence_selection_status
answer_token_f1_bucket
best_gold_span_f1_bucket
answer_gold_span_gap_bucket
verifier_decision
refusal_reason_code
question_route
evidence_selector_name
composition_policy_id
bucket_confidence_band
```

The case field count is `18`.

Private label inputs may be used only after prediction to compute buckets, and
must not be written to the public report:

```text
gold answer text -> token-F1 buckets only
gold answer document identifier -> retrieval/citation status only
retrieved/cited document identifiers -> status buckets only
```

## Decision

```text
status: primeqa_hybrid_answer_pipeline_error_decomposition_protocol_frozen
protocol_id: answer_pipeline_error_decomposition_train_dev_v1
recommended_direction: answer_pipeline_error_decomposition
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_analysis: true
can_run_train_dev_error_decomposition_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_stage: Stage102: after user confirmation, run the frozen train/dev-only answer-pipeline error decomposition analysis with public-safe aggregate and sanitized case outputs; keep test locked and runtime defaults unchanged.
```

## Guard Checks

All `26 / 26` guard checks passed:

```text
stage100_source_is_expected_stage: passed
user_confirmed_stage101_protocol: passed
stage100_completed: passed
stage100_recommends_answer_pipeline_decomposition: passed
stage100_first_wave_exhausted: passed
stage100_second_wave_exhausted: passed
stage100_has_no_runtime_advancing_retrieval_candidate: passed
stage100_has_no_remaining_actionable_retrieval_candidate: passed
stage100_final_test_gate_closed: passed
stage100_final_metrics_locked: passed
stage100_forbids_test_tuning: passed
stage100_runtime_default_unchanged: passed
protocol_id_is_fixed: passed
protocol_requires_confirmation_before_analysis_run: passed
split_contract_is_train_dev_only: passed
test_split_is_forbidden: passed
expected_decomposition_buckets_are_frozen: passed
bucket_precedence_is_deterministic_and_unique: passed
public_case_fields_are_whitelisted: passed
public_case_fields_exclude_private_text_and_document_ids: passed
metric_labels_allowed_only_after_prediction: passed
source_document_identifiers_forbidden_as_runtime_evidence: passed
fallback_strategies_are_disabled: passed
stage101_freezes_protocol_without_analysis_metrics: passed
stage101_final_test_metrics_not_run: passed
stage101_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_error_bucket_priority_weights.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_pipeline_stage_order.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_public_case_field_counts.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_output_artifact_contract.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_protocol_decision_flags.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_guard_check_status.svg
```

Stage101 JSON SHA256:

```text
952821C48CF882A01ADC1EFA5DD8C6F041B05ED7B009B1DEA2C29CE406538CFA
```

Visualization SHA256:

```text
stage101_error_bucket_priority_weights.svg: C7FAF98716C938263B3E4214BBCD3B8AC5E00E0080675A6A99CAE562773B4E57
stage101_guard_check_status.svg: 009900216392943CD2EB7C319FC6BBCBE4B3C75B33E9DEFBDE22966348675FA4
stage101_output_artifact_contract.svg: F0EA26EBDA00B560F63FC7E41904F476803A3FCE0D92318A7199AB871A217043
stage101_pipeline_stage_order.svg: 007E551FD1B9F92C0688ACC181848D6A7F484741CB316147E6C2FEE057586723
stage101_protocol_decision_flags.svg: 30114C5F6C2B168C4DB738B8CC7037A595DE842E97E1A23513BC5D58AAFBE462
stage101_public_case_field_counts.svg: 1C56955DDABA9226711C7B65A2A8B65339452CF65E97E53D53C5A759B2D43E73
```

## Validation

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py scripts\freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py
pytest -q tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py
python scripts\freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py ...: exit code 0
```

Result:

```text
ruff: passed
pytest: 4 passed
Stage101 run: passed
guard checks: 26 / 26 passed
```

Full validation:

```text
ruff check .: passed
pytest -q: 270 passed
git diff --check: passed
```

Artifact safety scan:

```text
Select-String Stage101 JSON for private fixture snippets and raw case field patterns: no matches
```

## Conclusion

- Stage101 froze the answer-pipeline error decomposition protocol.
- Stage101 did not load split files and did not run retrieval or answer metrics.
- Stage101 did not run final metrics.
- Stage101 did not add fallback strategies.
- Stage101 did not change runtime defaults.
- The next stage is Stage102, which may run the frozen train/dev-only
  decomposition after user confirmation, with public-safe aggregate and
  sanitized case outputs only.
