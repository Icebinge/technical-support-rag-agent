# Learning Journal

This document records the learning route, implementation decisions, problems,
debugging process, and evidence produced while building this project.

It is intentionally written as an engineering learning log rather than a polished
final report. The goal is to make each project step explainable in interviews:
what was learned, what failed, how it was fixed, and why a design choice was
made.

## How To Use This Journal

Add one entry after each meaningful study or implementation session.

Each entry should separate:

- facts: commands, outputs, files, metrics, links, commit ids
- reasoning: why a method or architecture was chosen
- problems: errors, wrong assumptions, missing knowledge
- resolution: what changed and how it was verified
- next step: the smallest concrete action for the next session

Do not record an experiment as completed unless it was actually run. If a result
is estimated, inferred, or planned, mark it clearly.

## Current Learning Route

| Stage | Topic | Goal | Output |
| --- | --- | --- | --- |
| 0 | Project and data setup | Build a public-safe project with real public data | Repo, docs, download scripts, verification snapshot |
| 1 | Dataset parsing | Understand TechQA fields and corpus structure | PrimeQA loader, NVIDIA eval loader, schema notes |
| 2 | Sparse retrieval | Build BM25 before using any Agent framework | Top-k retriever, hit@k evaluation |
| 3 | Dense retrieval | Compare embeddings with BM25 | Vector index, latency and quality comparison |
| 4 | RAG answering | Generate grounded answers from retrieved documents | Citation-aware answer generator |
| 5 | Agent workflow | Add query rewrite, retrieval grading, and verification | LangGraph workflow and trace logs |
| 6 | Service and report | Package the project for resume/interview use | FastAPI demo, evaluation report, project summary |

## Entry Template

Copy this block when adding a new record.

````markdown
## YYYY-MM-DD - Session Title

### Goal

- TODO

### What I Studied

- TODO

### What I Built Or Changed

- TODO

### Commands And Evidence

```powershell

```

### Problems Encountered

- TODO

### Root Cause

- TODO

### Solution

- TODO

### Why This Choice

- TODO

### Verification

- TODO

### What I Still Do Not Understand

- TODO

### Next Step

- TODO
````

## 2026-07-12 - Stage 0: Project Setup And Dataset Verification

### Goal

- Start a complete Agent/RAG project that can be learned step by step.
- Use public official datasets instead of private enterprise data or generated
  toy data.
- Keep the GitHub repository public-safe while still downloading real data
  locally.

### What I Studied

- Why VLM training and testing is not the right first project direction on the
  current local environment.
- Why a technical-support RAG Agent is more suitable for local learning:
  retrieval, evaluation, data processing, and workflow orchestration can be
  built without training a large model.
- The difference between using a dataset for development/training and reserving
  another dataset for evaluation.

### What I Built Or Changed

- Created the project folder:
  `C:\d_desktop\profile\technical_support_rag_agent`
- Created a public GitHub repository:
  `https://github.com/Icebinge/technical-support-rag-agent`
- Added project structure:
  `src/`, `scripts/`, `tests/`, `docs/`, `data/`, `artifacts/`, `outputs/`
- Added dataset download and verification scripts.
- Added typed dataset models and a loader for NVIDIA TechQA-RAG-Eval.
- Added architecture, data strategy, roadmap, and dataset snapshot documents.

### Commands And Evidence

```powershell
python -m pip install -e ".[dev,data]"
python -m ruff check .
python -m pytest -q
python scripts\download_datasets.py --include-primeqa
python scripts\verify_datasets.py
```

Observed verification facts:

```text
NVIDIA TechQA-RAG-Eval train.json: 3.83 MB
NVIDIA TechQA-RAG-Eval corpus.zip: 43.69 MB
NVIDIA rows: 910
answerable rows: 610
impossible rows: 300
missing referenced files: 0

PrimeQA/TechQA archive exists: true
PrimeQA/TechQA archive size: 2822.85 MB
```

Git commit:

```text
1c574d8 Initialize technical support RAG agent
```

### Problems Encountered

- Original VLM route was not suitable because local training and realistic
  testing were not feasible in the current environment.
- Public enterprise-style support data is not easy to obtain.
- Downloaded datasets and generated artifacts must not be committed to a public
  GitHub repository.
- Editable Python installation generated `egg-info` metadata that should not be
  part of the repository.

### Root Cause

- VLM projects often require GPU memory, model weights, and large-scale
  multimodal data that do not match the current local setup.
- Real company documents are private, so the project needs public technical
  support data with clear provenance.
- GitHub repositories should contain reproducible code and documentation, not
  large raw datasets or local runtime artifacts.
- `pip install -e .` can create local packaging metadata under `src/`.

### Solution

- Changed the project direction to a technical-support RAG Agent.
- Used `PrimeQA/TechQA` as the development/training source and
  `nvidia/TechQA-RAG-Eval` as the evaluation source.
- Added `.gitignore` rules for raw data, processed data, indexes, artifacts,
  outputs, caches, model files, and packaging metadata.
- Verified ignored data paths with `git check-ignore`.

### Why This Choice

- Agent/RAG work maps better to current mainstream engineering and algorithm
  roles: data processing, retrieval, evaluation, model calling, workflow
  orchestration, and service packaging.
- BM25 and retrieval baselines can be learned locally before adding LangChain or
  LangGraph.
- Separating development data from evaluation data makes the project easier to
  defend in interviews because it reduces evaluation leakage.
- Keeping raw datasets out of git protects the repository from large-file and
  licensing problems.

### Verification

- `ruff` passed.
- `pytest` passed.
- Dataset verification script completed.
- GitHub repository was created and pushed.
- Raw datasets were present locally but ignored by git.

### What I Still Do Not Understand

- The exact schema and field meanings inside the original PrimeQA archive.
- How much overlap exists between PrimeQA/TechQA and NVIDIA TechQA-RAG-Eval.
- What BM25 baseline score should be considered acceptable for this dataset.

### Next Step

- Parse the original PrimeQA archive.
- Inspect training/dev question formats and technote corpus formats.
- Build the first BM25 retrieval baseline.
- Evaluate hit@1, hit@5, and hit@10 before adding dense retrieval or Agent
  orchestration.
