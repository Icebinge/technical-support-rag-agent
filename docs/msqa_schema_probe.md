# MSQA Schema Probe

This document records the Stage 56 local probe of Microsoft Q&A (MSQA).

Stage 56 downloaded the public MSQA GitHub repository into local ignored storage
and inspected the CSV/test-id files. It did not run RAG answer-quality metrics,
did not freeze a final evaluation split, did not run near-duplicate leakage
search, and did not change the default runtime policy.

## Source

Repository:

```text
https://github.com/microsoft/Microsoft-Q-A-MSQA-
```

Download command used:

```powershell
git clone --depth 1 --filter=blob:none https://github.com/microsoft/Microsoft-Q-A-MSQA-.git data\raw\msqa_repo
```

Repository HEAD at probe time:

```text
4be7e0376f3fa2ee8cbaa90644bd0eeb291c43f4
```

Local raw data path:

```text
data/raw/msqa_repo/
```

This path is ignored by git and is not committed.

## File Fingerprints

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `data/raw/msqa_repo/data/msqa-32k.csv` | 106,085,656 | `38839bb6234a2195216762ee7827e770d9d6de2eb49f913a365cc04dbdc20d93` |
| `data/raw/msqa_repo/data/test_id.txt` | 4,343 | `644fa017e8abcabef91eacf3ef10f58b9118e959b9c6e4497779270183bc4ba1` |
| `data/raw/msqa_repo/README.md` | 16,664 | `2dad24449874c91d29089a97e1381af3eb95f4f4c4fa07dfd5f5fe8acf92b0ac` |

## Schema Result

Local parse result:

```text
row_count: 32236
README row count claim: 32252
delta: -16
field_count: 29
unique_question_ids: 32236
duplicate_question_id_rows: 0
duplicate_normalized_question_rows: 11
malformed_row_count: 0
```

Required fields checked:

```text
QuestionId: 0 missing
AnswerId: 0 missing
QuestionText: 0 missing
AnswerText: 0 missing
ProcessedAnswerText: 0 missing
Url: 0 missing
Split: 0 missing
```

Answer field availability:

```text
AnswerText: 32236 / 32236 available
ProcessedAnswerText: 32236 / 32236 available
DoubleProcessedAnswerText: 32160 / 32236 available
```

The missing `DoubleProcessedAnswerText` rows mean the future adapter must choose
its answer field explicitly. This stage does not choose or implement that
adapter contract.

## Distributions

Split counts:

```text
NNN: 21225
train: 7710
test: 3301
```

Domain flags:

```text
IsAzure=True: 11024
IsM365=True: 6912
IsOther=True: 15904
```

Length flags:

```text
isShort=True: 3345
isLong=True: 44
```

## Source-Link Coverage

```text
rows_with_row_url: 32236 / 32236 (100.0%)
rows_with_learn_answers_url: 32236 / 32236 (100.0%)
rows_with_question_text_link: 5307 / 32236 (16.463%)
rows_with_answer_text_link: 22024 / 32236 (68.321%)
rows_with_processed_answer_link: 19924 / 32236 (61.807%)
rows_with_double_processed_answer_link: 17048 / 32236 (52.885%)
rows_with_processed_answer_learn_link: 10929 / 32236 (33.903%)
rows_with_processed_answer_azure_docish_link: 4343 / 32236 (13.473%)
```

Every row has a Microsoft Learn Q&A page URL, which is useful for row-level
source attribution. Documentation-link coverage inside processed answers is
lower and must not be treated as complete evidence-document coverage.

## Test ID File

```text
test_id_count: 588
unique_test_id_count: 588
duplicate_test_id_count: 0
test_ids_found_in_csv: 587
test_ids_missing_from_csv_count: 1
test_ids_missing_from_csv: 699708
found_test_id_split_counts: test=587
```

The repository-provided `test_id.txt` is not directly sufficient as this
project's final held-out split because one ID is missing from the local CSV and
the repository split process was designed around additional Azure/length
filtering.

## PrimeQA Exact-Overlap Precheck

Compared MSQA question text against local PrimeQA train/dev question text using
the same normalized exact-match method used in earlier held-out leakage work.

```text
PrimeQA train questions: 600
PrimeQA dev questions: 310
exact_overlap_pair_count: 0
exact_overlap_msqa_question_count: 0
```

This passes only the exact-overlap precheck. Stage 56 did not run:

```text
near_duplicate_token_jaccard_search
semantic_duplicate_search
answer_or_document_overlap_search
```

## Decision

MSQA remains the recommended external evaluation candidate, but it is still not
an approved held-out test set.

Current status:

```text
schema_probe_passed_but_metrics_blocked
can_run_final_metrics_now: false
default_runtime_policy: unchanged
```

Blocking issues before metrics:

- near-duplicate leakage audit not run;
- project-owned MSQA evaluation split not frozen;
- MSQA adapter contract not implemented;
- no native unanswerable rows for refusal evaluation;
- local row count differs from README claim;
- `test_id.txt` contains one ID missing from local CSV;
- `DoubleProcessedAnswerText` has missing rows.

## Artifacts

```text
artifacts/msqa_schema_probe_stage56.json
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_split_distribution.svg
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_source_link_coverage.svg
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_domain_flags.svg
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_test_id_coverage.svg
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_primeqa_exact_overlap.svg
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Stage 57 should implement the MSQA adapter contract, run near-duplicate leakage
audit, and freeze a project-owned evaluation split before any top-k or Stage 51
comparison is attempted.
