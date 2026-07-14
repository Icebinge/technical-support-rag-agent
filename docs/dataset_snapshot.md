# Dataset Snapshot

This snapshot records the local dataset download and verification status from
2026-07-12.

The raw dataset files are intentionally ignored by git. This document records
the observed local facts so the repository remains public-safe while the setup
state is still auditable.

## Download Command

```powershell
python scripts\download_datasets.py --include-primeqa
```

## Local Files

```text
data/raw/nvidia_techqa_rag_eval/train.json      4,019,332 bytes
data/raw/nvidia_techqa_rag_eval/corpus.zip      45,814,227 bytes
data/raw/primeqa_techqa/TechQA.tar.gz           2,959,973,525 bytes
```

## Verification Command

```powershell
python scripts\verify_datasets.py
```

## NVIDIA TechQA-RAG-Eval

Observed local verification result:

```text
train_json_mb: 3.83
corpus_zip_mb: 43.69
total_rows: 910
answerable_rows: 610
impossible_rows: 300
unique_referenced_files: 496
missing_referenced_files: 0
corpus_files: 28481
min_contexts: 0
max_contexts: 1
avg_contexts: 0.67
```

Original intended use: evaluation.

Current boundary: Stage 53 later proved that all 910 rows in this file exactly
overlap PrimeQA train/dev after question normalization. Do not use this file as
an independent held-out evaluation source for the current project history.

## PrimeQA TechQA

Observed local verification result:

```text
archive_exists: true
archive_mb: 2822.85
```

The archive header was readable. The first listed entries included:

```text
TechQA/
TechQA/README.txt
TechQA/CDLA-Permissive-v1.0.pdf
TechQA/technote_corpus/
TechQA/technote_corpus/technotes.id_title_text
TechQA/technote_corpus/technotes.text
TechQA/technote_corpus/full_technote_collection.sections.json
TechQA/technote_corpus/full_technote_collection.txt.bz2
TechQA/validation/validation_questions.json
TechQA/validation/validation_reference.json
TechQA/training_and_dev/training_Q_A.json
TechQA/training_and_dev/dev_Q_A.json
TechQA/evaluation.py
```

Original intended use: training and development experiments.

Current boundary: Stage 67 used the extracted PrimeQA/TechQA train, dev, and
validation reference rows to plan a project-owned hybrid train/dev/test split.
Stage 68 froze that split as `primeqa_hybrid_stage68_v1`; final metrics still
must wait until derived loaders and artifacts are rebuilt from the frozen
boundary.

## Verification Boundary

The verification script validates NVIDIA sample parsing, corpus zip listing,
and missing referenced corpus files. For the large PrimeQA archive, the current
verification records local existence, size, and a readable tar header listing.
It does not yet perform full extraction, schema validation, or leakage checks.
