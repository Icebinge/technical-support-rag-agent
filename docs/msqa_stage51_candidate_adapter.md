# MSQA Stage 51 Candidate Adapter Dry Run

This document records the Stage 61 dry run for the user-confirmed Stage 60
protocol:

```text
msqa_row_source_url + processed_answer_sentence_candidates
```

Stage 61 implements the adapter contract and dry-run checks only. It does not
run Stage 51, does not tune policies, does not fetch external pages, and does
not change the default runtime.

## Confirmation

The user confirmed Stage 60 option A in the current Codex conversation on
2026-07-14.

Confirmed protocol:

```text
source_citation_identity: msqa_row_source_url
candidate_construction: processed_answer_sentence_candidates
```

The dry-run CLI requires:

```text
--confirmed-protocol
```

Without that flag, Stage 61 refuses to run.

## Adapter Contract

Inputs:

```text
artifacts/msqa_evaluation_split_stage57.jsonl
artifacts/msqa_stage51_protocol_stage60.json
```

Contract:

```text
split_name: msqa_stage57_project_eval_v1
adapter_contract_version: msqa_eval_adapter_v1
retrieval_index_text: ProcessedAnswerText only
excluded_index_text: QuestionText
source_citation_identity: QuestionId + AnswerId + Url
candidate_construction: ProcessedAnswerText answer sentences
top_k: 10
min_sentence_chars: 1
no_answer_field_fallback: true
external_fetch_used: false
```

Required candidate fields:

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

`candidate_score` is a dry-run adapter score only. It combines answer-only BM25
source retrieval score, retrieval-rank prior, and query-sentence token overlap.
It is not a tuned Stage 51 model score.

## Dry-Run Result

```text
evaluation_samples: 3301
candidate_rows: 266647
samples_with_candidates: 3301
samples_without_candidates: 0
samples_with_gold_source_candidate: 2023
average_candidates_per_sample: 80.7776
median_candidates_per_sample: 79.0
unique_source_rows_in_candidates: 2879
```

Source retrieval summary:

```text
hit@1: 0.4147
hit@10: 0.6128
MRR: 0.4762
gold_source_missing_at_10: 1278
```

The source retrieval summary matches the Stage 58 answer-only baseline boundary.
This confirms that Stage 61 is using the same answer-only source-row retrieval
task and is not using the rejected question-text diagnostic index.

## Contract Checks

All Stage 61 checks passed:

| Check | Result |
| --- | --- |
| `user_confirmed_stage60_protocol` | pass |
| `protocol_matches_stage60_recommendation` | pass |
| `no_question_text_indexed_or_written_to_candidates` | pass |
| `no_answer_field_fallback_used` | pass |
| `no_external_fetch_used` | pass |
| `all_candidates_have_required_fields` | pass |
| `all_samples_have_candidate_rows` | pass |

Candidate JSONL validation:

```text
jsonl rows: 266647
rows_with_question_key: 0
```

## Decision

Current status:

```text
msqa_stage51_candidate_adapter_dry_run_passed
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
stage51_candidate_run_performed: false
```

Stage 61 proves the adapter contract can generate complete local candidate
rows. It still does not prove that a Stage 51 comparison is fair or useful.

## Artifacts

```text
artifacts/msqa_stage51_candidate_adapter_stage61.json
artifacts/msqa_stage51_candidate_adapter_stage61_candidates.jsonl
artifacts/msqa_stage51_candidate_adapter_stage61_visuals/stage61_adapter_candidate_counts.svg
artifacts/msqa_stage51_candidate_adapter_stage61_visuals/stage61_adapter_source_hit_rates.svg
artifacts/msqa_stage51_candidate_adapter_stage61_visuals/stage61_adapter_contract_checks.svg
```

Stage 61 report checksum:

```text
c43d8dd6a38b1539bde8d1681c23878b60f663916f01c877f93ff5d38400d783
```

Stage 61 candidate JSONL checksum:

```text
e505895730f1bf4451dc3c7e0130798692d0c2f298c1b284f19083cb99b96980
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Stage 62 should review the MSQA adapter candidate distribution and decide
whether one single Stage 51 adapter comparison is fair. It should still not
defaultize anything.
