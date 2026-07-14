# PrimeQA Hybrid Split Freeze

This document records Stage 68: the project-owned PrimeQA/TechQA hybrid split
freeze.

Stage 68 materializes the Stage 67 dry-run boundary into local train/dev/test
JSONL artifacts. It does not rebuild retrieval indexes, train or tune models,
run answer-quality metrics, use test rows for tuning, or change the default
runtime.

## Frozen Split

```text
split_name: primeqa_hybrid_stage68_v1
protocol_version: primeqa_hybrid_split_v1
status: primeqa_hybrid_split_frozen_for_rebuild
split_files_finalized: true
can_run_final_metrics_now: false
default_runtime_policy: unchanged
```

The split uses the Stage 67 `aaa` route:

```text
1A: fully isolate 10% of answer documents into a document-disjoint test subtype
2A: split the remaining grouped rows into 70% train, 15% dev, 15% random test
3A: include PrimeQA validation_reference rows in the planning pool
```

## Command

```powershell
python scripts\freeze_primeqa_hybrid_split.py `
  --train-questions data\raw\primeqa_techqa\TechQA\training_and_dev\training_Q_A.json `
  --dev-questions data\raw\primeqa_techqa\TechQA\training_and_dev\dev_Q_A.json `
  --validation-reference data\raw\primeqa_techqa\TechQA\validation\validation_reference.json `
  --output artifacts\primeqa_hybrid_split_stage68_freeze.json `
  --split-output-dir artifacts\primeqa_hybrid_split_stage68_splits `
  --visualization-dir artifacts\primeqa_hybrid_split_stage68_visuals `
  --document-disjoint-answer-doc-ratio 0.10 `
  --remainder-train-ratio 0.70 `
  --remainder-dev-ratio 0.15 `
  --remainder-test-ratio 0.15 `
  --seed 20260714
```

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

Source distribution:

| Split | primeqa_train | primeqa_dev | primeqa_validation |
| --- | ---: | ---: | ---: |
| train | 370 | 178 | 14 |
| dev | 70 | 49 | 2 |
| test | 160 | 83 | 4 |

Sample ID checksums:

```text
all samples: 4adca87e4c2537b38d6fed1b36bebdd77a7fa89ba9ff5577b7421acb6ec67bff
train: 8d632a9f7bb88240f6136259ce039155bd9ffbc42727d91a9e75d676707ff0e5
dev: 3bafd55fc4a604c1dc08c63f67da99a4d5f8872fa7683e9fdb3272c636329e63
test: 8eb92ba16e4066903bf6255850665ce24ac9a525abeed892179ff2fdc48a029f
```

## Checks

Stage 67 leakage checks carried into the freeze:

```text
normalized_question_answer_doc_groups_do_not_cross_splits: passed
selected_document_answer_docs_only_in_document_disjoint_test: passed
selected_document_candidate_doc_ids_only_in_document_disjoint_test: passed
```

Stage 68 freeze checks:

```text
all_stage67_leakage_checks_passed: passed
all_raw_rows_have_assignments: passed
frozen_row_count_matches_input: passed
train_dev_test_are_nonempty: passed
document_disjoint_test_rows_materialized: passed
```

## Artifacts

These are local ignored artifacts and are not committed by git policy.

Report:

```text
artifacts/primeqa_hybrid_split_stage68_freeze.json
sha256: 9d6873ccbf9ae5d23882c68b64d264b8e957725f5e1ad03c7a3534158df59525
```

Frozen split JSONL files:

```text
artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_train.jsonl
rows: 562
sha256: cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155

artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_dev.jsonl
rows: 121
sha256: 071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f

artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_test.jsonl
rows: 247
sha256: f2479cf636bd40f6d066e1c9be03431f9b37460a8b489a37222523f2d902b1c1
```

The split JSONL files contain raw question and answer text because future local
training and evaluation rebuilds need it. They remain ignored local artifacts.

Visualizations:

```text
artifacts/primeqa_hybrid_split_stage68_visuals/stage68_primeqa_frozen_answerable_rows.svg
sha256: 079fe41a37a3d9ca095bf2ec644745c5f0018d1fec8b8851af9d4a3c5e51b858

artifacts/primeqa_hybrid_split_stage68_visuals/stage68_primeqa_frozen_source_rows.svg
sha256: 530be40a1d49d431ac5e6d60b2af6cb5640746110dfb9033dbcb7cb2593f5675

artifacts/primeqa_hybrid_split_stage68_visuals/stage68_primeqa_frozen_split_rows.svg
sha256: 42d50d6ac933b07e8252b5a2c62fc4a7a252913f577166bd0d0bdc0186588759

artifacts/primeqa_hybrid_split_stage68_visuals/stage68_primeqa_frozen_test_subtypes.svg
sha256: b6c65f4baea688364b2505c1c4b20788a257dc016e9c29ee0d996bc827337b9b
```

## Decision

Stage 68 freezes `primeqa_hybrid_stage68_v1` for local artifact rebuilds.

This is not a model-quality result. No retrieval, reranker, answer-composition,
or final-test metric was run in this stage.

## Next Step

Stage 69 should rebuild PrimeQA train/dev/test data loaders and derived
candidate artifacts from `primeqa_hybrid_stage68_v1`, without using test rows
for tuning.
