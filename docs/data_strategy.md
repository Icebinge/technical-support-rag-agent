# Data Strategy

## Main Sources

### PrimeQA/TechQA

Use case: training and development source.

The original TechQA package is large and contains the TechQA question-answer
data plus a large IBM Technotes corpus. This project uses it for local corpus
construction and for development experiments after leakage checks.

Local path:

```text
data/raw/primeqa_techqa/TechQA.tar.gz
```

### nvidia/TechQA-RAG-Eval

Use case: final evaluation set.

This dataset is a reduced TechQA-derived RAG evaluation dataset with question,
answer, answerability flag, and evidence contexts.

Local paths:

```text
data/raw/nvidia_techqa_rag_eval/train.json
data/raw/nvidia_techqa_rag_eval/corpus.zip
```

## Split Rule

The NVIDIA dataset is held out for evaluation. It must not be used for:

- prompt tuning
- retriever parameter tuning
- reranker training
- answerability classifier training
- model selection after many repeated attempts

The PrimeQA data can be used for development, but any overlap with NVIDIA
evaluation questions must be removed before training or tuning.

## Leakage Checks

Before reporting evaluation results:

1. Normalize questions from both sources.
2. Compare exact normalized strings.
3. Compare near-duplicates with token overlap or embedding similarity.
4. Exclude overlapping samples from training or development.
5. Save a leakage report under `artifacts/`.

## Git Policy

Commit:

- scripts
- docs
- schema definitions
- small synthetic fixtures for tests

Do not commit:

- downloaded datasets
- extracted corpus documents
- indexes
- model outputs from evaluation
- model weights
- private notes or credentials

## Honest Reporting

Do not claim model training, retrieval improvement, or answer quality before the
corresponding experiment has actually been run and saved.
