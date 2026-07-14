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

Original intended use case: final evaluation set.

This dataset is a reduced TechQA-derived RAG evaluation dataset with question,
answer, answerability flag, and evidence contexts.

Local paths:

```text
data/raw/nvidia_techqa_rag_eval/train.json
data/raw/nvidia_techqa_rag_eval/corpus.zip
```

Current boundary:

Stage 53 leakage audit found that `train.json` is not an independent held-out
set for the current PrimeQA train/dev development history. All 910 NVIDIA rows
have exact normalized question overlap with PrimeQA train/dev, producing 974
heldout-development overlap pairs because some development questions normalize
to duplicate text.

Therefore, do not use `nvidia/TechQA-RAG-Eval/train.json` as a held-out
defaultization test for the current Stage 51 candidate unless a future workflow
first redesigns the data split and removes or redoes all affected tuning.

### Microsoft Q&A (MSQA)

Current use case: external evaluation candidate only.

Stage 55 reviewed public dataset-owner sources and recommends Microsoft Q&A
(MSQA) as the next schema-probe candidate because it is an external
technical-support QA dataset collected from Microsoft Q&A, with human-generated
accepted answers and an explicitly listed dataset license.

Current public source:

```text
https://github.com/microsoft/Microsoft-Q-A-MSQA-
```

Stage 55 source-backed facts:

- public GitHub archive;
- 32,252 rows according to the dataset README;
- Microsoft product and IT technical-problem QA domain;
- data files include `msqa-32k.csv` and `test_id.txt`;
- dataset license is listed as CDLA-Permissive-2.0 in the README.

Current boundary:

- Stage 55 did not download or commit the MSQA CSV.
- Stage 55 did not parse MSQA rows locally.
- Stage 55 did not run a PrimeQA leakage audit against MSQA.
- Stage 55 did not run top-k or Stage 51 metrics on MSQA.
- MSQA is not yet a held-out test set for this project.

Before MSQA can become a real held-out evaluation source, the next stage must
probe its local schema, measure source-link/citation coverage, run leakage
checks against PrimeQA train/dev, and freeze an evaluation split.

## Split Rule

Any future held-out dataset must not be used for:

- prompt tuning
- retriever parameter tuning
- reranker training
- answerability classifier training
- model selection after many repeated attempts

The PrimeQA data can be used for development, but any overlap with NVIDIA
evaluation questions must be removed before training or tuning.

For the current repository state, NVIDIA `train.json` has already been shown to
overlap completely with PrimeQA train/dev, so it is blocked as a held-out
evaluation source.

The current evaluation path options are recorded in:

```text
docs/evaluation_strategy.md
```

The current external dataset discovery snapshot is recorded in:

```text
docs/external_eval_datasets.md
```

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
