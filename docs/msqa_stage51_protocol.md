# MSQA Stage 51 Protocol Design

This document records the Stage 60 MSQA source/citation adapter and Stage 51
comparison protocol design.

Stage 60 does not run Stage 51, does not tune policies, does not fetch external
pages, does not add fallback fields, and does not change the default runtime.

## Inputs

Stage 56 schema probe:

```text
artifacts/msqa_schema_probe_stage56.json
```

Stage 57 evaluation split:

```text
artifacts/msqa_evaluation_split_stage57.json
```

Stage 59 compatibility review:

```text
artifacts/msqa_stage51_compatibility_stage59.json
```

Current constraints:

```text
frozen_split: msqa_stage57_project_eval_v1
adapter_contract_version: msqa_eval_adapter_v1
selected_question_count: 3301
approved_answer_field: ProcessedAnswerText
approved_source_url_field: Url
no_answer_field_fallback: true
question_text_index_rejected: true
stage59_blocker_count: 5
default_runtime_policy: unchanged
```

## Source Identity Options

The scores below are generated protocol-fit rubric scores for Stage 60 design
only. They are not model-quality metrics.

| Option | Status | Coverage | Score | Decision |
| --- | --- | ---: | ---: | --- |
| `msqa_row_source_url` | recommended for user confirmation | 100.0 | 9 | Use `QuestionId + AnswerId + Url` as row-source citation identity |
| `processed_answer_links` | blocked | 61.807 | 8 | Do not use as required identity because coverage is incomplete |
| `processed_answer_learn_links` | blocked | 33.903 | 7 | Do not use as required identity because Learn-link coverage is incomplete |
| `processed_answer_azure_docish_links` | blocked | 13.473 | 7 | Do not use as required identity because coverage is too sparse |
| `question_answer_page_text` | rejected | 100.0 | 4 | Reject because indexing question text trivializes retrieval |

## Candidate Construction Options

| Option | Status | Coverage | Score | Decision |
| --- | --- | ---: | ---: | --- |
| `processed_answer_sentence_candidates` | recommended for user confirmation | 100.0 | 9 | Split `ProcessedAnswerText` into answer sentences attached to row-source identity |
| `processed_answer_chunk_candidates` | secondary option | 100.0 | 8 | Keep as backup design if sentence splitting proves too noisy |
| `source_row_single_candidate` | secondary option | 100.0 | 8 | Keep as smoke-test adapter only |
| `linked_learn_document_candidates` | blocked | 33.903 | 6 | Do not implement next because link coverage is incomplete |
| `question_answer_text_candidates` | rejected | 100.0 | 5 | Reject because question text must not enter the comparison index |

## Recommended Protocol

Status:

```text
draft_requires_user_confirmation
```

Recommended source/citation identity:

```text
msqa_row_source_url
```

Recommended candidate construction:

```text
processed_answer_sentence_candidates
```

Protocol:

```text
retrieval_corpus_scope: frozen_split_only
retrieval_index_text: ProcessedAnswerText only
excluded_index_text: QuestionText
gold_source_identity: QuestionId + AnswerId + Url
candidate_identity: QuestionId::processed_answer_sentence::<one_based_sentence_index>
```

Required fields for the next adapter:

```text
question_id
answer_id
source_url
candidate_id
candidate_sentence
retrieval_rank
retrieval_score
candidate_score
source_row_id
```

Allowed metrics after confirmation:

```text
source_row_hit@k
source_row_mrr
top1_answer_token_f1
oracle_answer_token_f1@k
source_citation_preservation_delta
```

Explicit exclusions:

- Do not use `AnswerText` fallback.
- Do not use `DoubleProcessedAnswerText` fallback.
- Do not index `QuestionText` for candidate comparison.
- Do not require processed-answer links as citation ground truth.
- Do not fetch external pages in this protocol.
- Do not change the default runtime.

## Decision

Current status:

```text
msqa_stage51_protocol_ready_for_user_confirmation
requires_user_confirmation: true
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
```

No implementation or Stage 51 MSQA candidate run should happen until the
recommended protocol is confirmed.

## Artifacts

```text
artifacts/msqa_stage51_protocol_stage60.json
artifacts/msqa_stage51_protocol_stage60_visuals/stage60_source_identity_scores.svg
artifacts/msqa_stage51_protocol_stage60_visuals/stage60_candidate_construction_scores.svg
artifacts/msqa_stage51_protocol_stage60_visuals/stage60_source_coverage.svg
artifacts/msqa_stage51_protocol_stage60_visuals/stage60_decision_flags.svg
```

Stage 60 report checksum:

```text
2267f1ac14e0866eb4c4835f40a06124d00833503a0c756bc06f0e891983db25
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Ask the user to confirm whether to proceed with:

```text
msqa_row_source_url + processed_answer_sentence_candidates
```

If confirmed, Stage 61 should implement the MSQA row-source answer-sentence
candidate adapter and dry-run contract tests. It still should not run a final
Stage 51 comparison until the adapter contract passes.
