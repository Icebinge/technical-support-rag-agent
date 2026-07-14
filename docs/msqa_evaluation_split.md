# MSQA Evaluation Split

This document records the Stage 57 MSQA adapter contract, PrimeQA leakage audit,
and project-owned evaluation split freeze.

Stage 57 did not run RAG answer-quality metrics, did not compare top-k against
Stage 51, did not tune any policy, and did not change the default runtime.

## Adapter Contract

Contract version:

```text
msqa_eval_adapter_v1
```

Field mapping:

| Local evaluation field | MSQA CSV field |
| --- | --- |
| sample ID | `QuestionId` |
| answer ID | `AnswerId` |
| question | `QuestionText` |
| gold answer | `ProcessedAnswerText` |
| source URL | `Url` |
| source split | `Split` |
| metadata | `Tags`, `IsAzure`, `IsM365`, `IsOther`, `isShort`, `isLong` |

No fallback policy:

```text
Do not fall back to AnswerText or DoubleProcessedAnswerText.
```

`ProcessedAnswerText` is the only approved answer field in this contract because
Stage 56 found it present for every local MSQA row. The row-level `Url` is the
approved source URL. This does not claim complete documentation-span citation
coverage.

Unsupported evaluation modes:

- native unanswerable/refusal evaluation;
- document-span exact citation evaluation.

## Leakage Audit

Method:

```text
token_jaccard_with_development_token_inverted_index
```

Normalization:

```text
lowercase_ascii_alnum_whitespace
```

Near-duplicate threshold:

```text
0.9
```

Results:

```text
MSQA questions: 32236
PrimeQA development questions: 910
PrimeQA train questions: 600
PrimeQA dev questions: 310
exact_overlap_count: 0
near_duplicate_overlap_count: 0
unhandled_overlap_count: 0
MSQA questions without detected overlap: 32236
```

The first naive full comparison attempt timed out before producing a valid
artifact. The committed Stage 57 implementation uses the same normalized
Jaccard definition with an inverted index and length-bound pruning, then wrote
the saved report below.

## Frozen Split

Split name:

```text
msqa_stage57_project_eval_v1
```

Protocol version:

```text
msqa_project_eval_split_v1
```

Selection rule:

```text
Use MSQA rows whose CSV Split is 'test', then exclude rows with invalid
row-level Microsoft Learn Q&A URLs, internal normalized-question duplicates, or
detected PrimeQA exact/near-duplicate leakage.
```

Filter results:

```text
loaded_contract_rows: 32236
source_split_candidates: 3301
excluded_invalid_source_url: 0
excluded_internal_normalized_duplicates: 0
excluded_primeqa_leakage: 0
selected_question_count: 3301
```

Selected-domain counts:

```text
IsAzure=True: 3301
IsM365=True: 490
IsOther=True: 0
isShort=True: 285
isLong=True: 0
```

Selected question ID checksum:

```text
26cab0b636845cd321a48c12e8bcbeb5b563e5eb234e63383bbc9d0a9d8cb93b
```

Frozen split JSONL checksum:

```text
b2beb8f20351999ee38c8679e37619da2a005d635d116ff0dedabf14f9600e54
```

First selected question IDs:

```text
3619, 2053, 3572, 2343, 282, 1102, 2096, 245, 252, 2221
```

Last selected question IDs:

```text
1192352, 1166696, 1185438, 1253836, 1195429, 1190151, 411097, 602066, 1218776, 1282090
```

## Decision

Stage 57 freezes MSQA for the next baseline step only.

Current status:

```text
msqa_split_frozen_for_baseline_evaluation
can_run_msqa_topk_baseline_next: true
can_run_stage51_comparison_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
```

The next stage may run a top-k baseline on the frozen split. Stage 51 comparison
must wait until the baseline result and failure modes are recorded.

## Artifacts

```text
artifacts/msqa_evaluation_split_stage57.json
artifacts/msqa_evaluation_split_stage57.jsonl
artifacts/msqa_evaluation_split_stage57_visuals/stage57_msqa_leakage_counts.svg
artifacts/msqa_evaluation_split_stage57_visuals/stage57_msqa_split_filter_counts.svg
artifacts/msqa_evaluation_split_stage57_visuals/stage57_msqa_selected_domain_flags.svg
artifacts/msqa_evaluation_split_stage57_visuals/stage57_msqa_adapter_field_coverage.svg
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Stage 58 should run the top-k baseline only on
`msqa_stage57_project_eval_v1`, then record baseline quality and failure modes
before any Stage 51 candidate comparison.
