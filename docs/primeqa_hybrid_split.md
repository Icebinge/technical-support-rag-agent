# PrimeQA Hybrid Split Plan

This document records Stage 67: a dry-run plan for rebuilding a project-owned
PrimeQA/TechQA train/dev/test split.

Stage 67 is a generated split plan based on real local PrimeQA/TechQA files. It
does not freeze final split files, does not rebuild retrieval indexes, does not
run model metrics, and does not change the default runtime.

## Route

After reviewing external dataset limits, the user chose to keep the final target
as document-style RAG and rebuild from the existing PrimeQA/TechQA data.

The route used for this dry run was:

```text
1A: fully isolate 10% of answer documents into a document-disjoint test subtype
2A: split the remaining grouped rows into 70% train, 15% dev, 15% random test
3A: include PrimeQA validation_reference rows in the planning pool
```

The PrimeQA validation rows are included for deduplication and distribution
planning only. They are not treated as independent held-out evidence.

## Inputs

```text
data/raw/primeqa_techqa/TechQA/training_and_dev/training_Q_A.json
bytes: 1149612
sha256: 0e9b77bc1ad7f7e95d05b095a34b1ece75c679575747949288a1d0f7470f4f46

data/raw/primeqa_techqa/TechQA/training_and_dev/dev_Q_A.json
bytes: 560864
sha256: df62a05619b07b386159bcbaaf3e0e50be93eaf983aea8a5f8ed1513a3c2ec6f

data/raw/primeqa_techqa/TechQA/validation/validation_reference.json
bytes: 40697
sha256: 83f2487dbfc3ae48538f84cc2f5d9427b160241381ed5ab5e13383b4c006a5d8
```

## Command

```powershell
python scripts\plan_primeqa_hybrid_split.py `
  --train-questions data\raw\primeqa_techqa\TechQA\training_and_dev\training_Q_A.json `
  --dev-questions data\raw\primeqa_techqa\TechQA\training_and_dev\dev_Q_A.json `
  --validation-reference data\raw\primeqa_techqa\TechQA\validation\validation_reference.json `
  --output artifacts\primeqa_hybrid_split_stage67.json `
  --assignments-output artifacts\primeqa_hybrid_split_stage67_assignments.jsonl `
  --visualization-dir artifacts\primeqa_hybrid_split_stage67_visuals `
  --document-disjoint-answer-doc-ratio 0.10 `
  --remainder-train-ratio 0.70 `
  --remainder-dev-ratio 0.15 `
  --remainder-test-ratio 0.15 `
  --seed 20260714
```

## Protocol

Rows are grouped by:

```text
normalized_question + answer_doc_id
```

For unanswerable rows, the group key uses:

```text
normalized_question + UNANSWERABLE
```

The document-disjoint subtype first samples 10% of unique answer document IDs
with deterministic seed `20260714`. Any group whose candidate `DOC_IDS` contain
one of those selected documents is assigned to `test/document_disjoint`.

The remaining groups are randomly assigned by group to train/dev/random-test at
70/15/15. Duplicate normalized-question groups stay in one split.

## Input Summary

| Metric | Value |
| --- | ---: |
| rows | 930 |
| groups | 889 |
| duplicate groups | 40 |
| duplicate rows | 81 |
| answerable rows | 621 |
| unanswerable rows | 309 |
| answerable rate | 0.6677 |
| unique answer documents | 496 |
| unique candidate documents | 28,461 |

Source rows:

| Source | Rows |
| --- | ---: |
| primeqa_train | 600 |
| primeqa_dev | 310 |
| primeqa_validation | 20 |

Duplicate group source patterns:

| Pattern | Groups |
| --- | ---: |
| primeqa_dev | 2 |
| primeqa_dev+primeqa_train | 14 |
| primeqa_dev+primeqa_train+primeqa_validation | 1 |
| primeqa_dev+primeqa_validation | 19 |
| primeqa_train | 4 |

## Document-Disjoint Test

| Metric | Value |
| --- | ---: |
| unique answer documents | 496 |
| selected answer documents | 50 |
| selected answer document checksum | `aedb4bef21d64aa58d6a99ccc2099ad4c0b03f07790d6d6c5194a386bd0dbdf9` |
| document-disjoint groups | 121 |
| document-disjoint rows | 126 |
| groups whose answer doc was selected | 67 |
| groups included only because candidate DOC_IDS intersected selected docs | 54 |
| answerable rows | 103 |
| unanswerable rows | 23 |

Document-disjoint source rows:

| Source | Rows |
| --- | ---: |
| primeqa_train | 88 |
| primeqa_dev | 37 |
| primeqa_validation | 1 |

## Split Summary

| Split | Rows | Groups | Answerable | Unanswerable | Answerable rate | Unique answer docs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 562 | 534 | 370 | 192 | 0.6584 | 309 |
| dev | 121 | 117 | 76 | 45 | 0.6281 | 71 |
| test | 247 | 238 | 175 | 72 | 0.7085 | 145 |

Test subtypes:

| Subtype | Rows |
| --- | ---: |
| document_disjoint | 126 |
| group_random_test | 121 |

## Leakage Checks

| Check | Observed | Expected | Passed |
| --- | ---: | ---: | --- |
| normalized_question_answer_doc_groups_do_not_cross_splits | 0 | 0 | true |
| selected_document_answer_docs_only_in_document_disjoint_test | 0 | 0 | true |
| selected_document_candidate_doc_ids_only_in_document_disjoint_test | 0 | 0 | true |

The third check verifies the strict document-disjoint rule at candidate-document
level: selected answer documents do not appear in non-document-disjoint
train/dev/random-test candidate `DOC_IDS`.

## Artifacts

The following outputs are local ignored artifacts under `artifacts/` and are not
committed:

```text
artifacts/primeqa_hybrid_split_stage67.json
sha256: c59ad5269861d866364468031f08fe577a05aeb84970fd89f48b0226a08718e2

artifacts/primeqa_hybrid_split_stage67_assignments.jsonl
sha256: 713bd017ab52e27a6524733499bf240a79a4eb46503ab5b8b82a526e0940599e

artifacts/primeqa_hybrid_split_stage67_visuals/stage67_primeqa_answerable_rows.svg
sha256: 816bceb4fddf5877d92a07b1e1a29af0c02420555c87e84aab0d399730fed270

artifacts/primeqa_hybrid_split_stage67_visuals/stage67_primeqa_source_rows.svg
sha256: 5256abaf6edaffcd9a7d248bb347f1778aa0dc2caf987ab5994d5f8656b5761c

artifacts/primeqa_hybrid_split_stage67_visuals/stage67_primeqa_split_rows.svg
sha256: 62251ab0d5b1ab42cfd2a688b20413ba3b61b15a68f9f8bc5672d5f55b6a64ed

artifacts/primeqa_hybrid_split_stage67_visuals/stage67_primeqa_test_subtypes.svg
sha256: e8c06e71108e39d407a68dac748251501dc1a1ef39e92a674a3707f9aef6a855
```

The assignments JSONL intentionally omits raw question text and raw answer text.
It contains row IDs, source split labels, assigned split/subtype, group hashes,
answerability flags, answer document IDs, candidate document counts, and
candidate document hashes.

## Decision

```text
status: primeqa_hybrid_split_dry_run_ready_for_review
split_files_finalized: false
can_run_final_metrics_now: false
default_runtime_policy: unchanged
```

Stage 67 is ready for review. It is not final evidence for quality or
defaultization.

## Next Step

Stage 68 should review the Stage 67 dry-run distribution and confirm whether to
freeze this hybrid split. If it is frozen, the next implementation stage should
rebuild train/dev/test artifacts from the new split boundary before rerunning
training, retrieval, or answer-composition metrics.
