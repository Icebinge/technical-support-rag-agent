# Project Scope

## Problem

Technical support teams answer questions by searching historical support notes,
product documents, and troubleshooting articles. A RAG system should retrieve
the right support document, produce a grounded answer, and refuse questions that
cannot be answered from available evidence.

## Why This Dataset Direction

The project uses public technical support data instead of synthetic company
documents. This makes the project more realistic and easier to explain in a
resume or interview.

## In Scope

- public dataset ingestion
- corpus indexing
- retrieval evaluation
- answerability handling
- grounded answer generation
- citation checking
- LangGraph workflow orchestration
- local API service
- experiment reports

## Out of Scope For The First Version

- training a large language model
- private enterprise data
- paid cloud APIs as the core project dependency
- customer-service UI polish
- production security certification

## Success Criteria

The first complete version should be able to:

1. Build a local index from public technical documents.
2. Answer held-out technical questions with cited sources.
3. Reject unanswerable questions.
4. Report retrieval hit rate, answer exact/semantic quality, citation correctness,
   refusal accuracy, latency, and failure categories.
