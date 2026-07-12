# Architecture

The project is intentionally split into layers so that each learning stage can
be implemented and tested independently.

## Layers

```text
src/ts_rag_agent/
  domain/          pure data models and evaluation concepts
  ports/           interfaces for retrieval, generation, and tracing
  infrastructure/  dataset loaders, indexes, model adapters
  application/     workflows and use cases
```

## Stage 1: Data Ingestion

Goal: load TechQA data into stable internal models.

Deliverables:

- dataset download script
- dataset verification script
- loader for NVIDIA TechQA-RAG-Eval
- dataset statistics report

## Stage 2: Retrieval Baseline

Goal: build a first retrieval baseline before using any Agent framework.

Deliverables:

- BM25 index
- top-k retrieval API
- hit@1, hit@5, hit@10 evaluation
- error examples

## Stage 3: Vector Retrieval

Goal: compare dense retrieval with BM25.

Deliverables:

- local embedding model
- vector index
- retrieval comparison table
- latency and memory notes

## Stage 4: RAG Answering

Goal: answer only from retrieved contexts.

Deliverables:

- grounded answer prompt
- citation format
- unanswerable handling
- answer-level evaluation notes

## Stage 5: Agent Workflow

Goal: use LangGraph after the basic pipeline is understood.

Planned nodes:

```text
classify_question
retrieve_documents
grade_retrieval
rewrite_query
generate_answer
verify_citations
finalize_response
```

## Stage 6: Service And Report

Goal: package the system as a small service and a defensible project.

Deliverables:

- FastAPI endpoint
- trace log
- evaluation report
- README project summary
