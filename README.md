# Technical Support RAG Agent

Technical Support RAG Agent is a learning-first project for building a
technical-support knowledge-base assistant with retrieval, tool use, evaluation,
and traceable engineering decisions.

The project uses public TechQA-derived datasets instead of private company
documents:

- **PrimeQA/TechQA**: original IBM TechQA package. Use it as the training and
  development source after leakage checks.
- **nvidia/TechQA-RAG-Eval**: a reduced TechQA RAG evaluation dataset. Use it
  as the held-out evaluation set, not as training data.

No downloaded dataset files are committed to this repository.

## Current Status

Stage 0: project scaffold and dataset verification.

Implemented:

- public-safe repository structure
- dataset download script
- dataset verification script
- local download completed for NVIDIA TechQA-RAG-Eval and PrimeQA/TechQA
- architecture and data strategy docs
- minimal domain models and loader utilities

Not implemented yet:

- BM25 retriever
- vector retriever
- reranker
- RAG answer generator
- LangGraph workflow
- FastAPI service
- full evaluation runner

## Quickstart

```powershell
cd C:\d_desktop\profile\technical_support_rag_agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -e ".[dev,data]"
```

Download the lightweight RAG evaluation dataset:

```powershell
python scripts\download_datasets.py --eval-only
python scripts\verify_datasets.py
```

Download the original PrimeQA TechQA archive as well:

```powershell
python scripts\download_datasets.py --include-primeqa
python scripts\verify_datasets.py
```

The original PrimeQA archive is large. Keep it under `data/raw/primeqa_techqa/`
and out of git.

Local dataset verification from 2026-07-12 is recorded in
[docs/dataset_snapshot.md](docs/dataset_snapshot.md).

The project learning route and implementation notes are recorded in
[docs/learning_journal.md](docs/learning_journal.md).

## Dataset Policy

This project treats data splits conservatively:

- NVIDIA TechQA-RAG-Eval is reserved for final evaluation.
- PrimeQA/TechQA is used for corpus construction, development, and possible
  training of retrieval-related components.
- Before any training result is reported, the project must check overlap between
  development data and evaluation questions.

See [docs/data_strategy.md](docs/data_strategy.md).

## Project Goal

Build a RAG Agent that can answer technical support questions with cited
evidence, decide when a question is unanswerable, and expose measurable system
quality through retrieval and answer-level evaluation.

## Planned Architecture

```text
Question
  -> question normalization
  -> retriever tool
       -> BM25 retrieval
       -> vector retrieval
       -> optional rerank
  -> answerability check
  -> grounded answer generation
  -> citation validation
  -> evaluation and trace logging
```

See [docs/architecture.md](docs/architecture.md).

## License

Project code is released under Apache-2.0. Dataset files remain governed by
their upstream licenses and terms.
