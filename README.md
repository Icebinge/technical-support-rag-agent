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

Stage 155: strict Agent runtime activation and node observability implemented and validated.

Implemented:

- public-safe repository structure
- dataset download script
- dataset verification script
- local download completed for NVIDIA TechQA-RAG-Eval and PrimeQA/TechQA
- architecture and data strategy docs
- minimal domain models and loader utilities
- BM25, section BM25, dense-cache, and fused high-recall retrieval
- extractive answer generation, verification, and optional sidecar Agent orchestration
- disabled-by-default process-scoped runtime bootstrap with public-safe traces
- train grouped-CV and dev report-only evaluation through Stage 145
- executable concurrent-runtime validation policy and workload contract
- process-shared, request-isolated concurrency-four research runtime
- disabled-by-default concurrent application bootstrap with evidence recomputation
- executable private-call/public-telemetry facade protocol with lifecycle and error guards
- label-free runtime query and transport-neutral Agent request facade
- typed capacity/lifecycle errors and natural no-timeout shutdown
- executable local-loopback HTTP transport protocol with strict size and error guards
- disabled-by-default loopback-only FastAPI adapter with exact ASGI schemas
- real HTTP/1.1 socket, overload, disconnect, readiness, and shutdown validation
- executable local service-entrypoint composition and fail-closed startup protocol
- exact `--port` local service CLI with stable startup exit codes
- built-in label-free warmup, one-shot resource composition, and prebound listener ownership
- real resource, HTTP/1.1, main-thread Uvicorn, and natural shutdown lifecycle validation
- executable nine-state Agent tool-workflow protocol with strict transition guards
- frozen three-tool context authority, private state, and public trace contracts
- LangGraph 1.2.9 `StateGraph` adapter compiled once per workflow instance
- framework-neutral and LangGraph engine equivalence for complete/refuse paths
- request-isolated four-way graph execution with exact error propagation
- active facade, FastAPI, and local service request paths using the graph workflow
- strict Stage154 formal-evidence and current-source fingerprint activation gate
- synchronous 22-field public-safe workflow and node observation events
- request-isolated node timing across complete, refuse, failure, and four-way concurrency

Not implemented yet:

- autonomous LLM tool selection, memory, and repeated retrieval
- remote network serving
- runtime defaultization and final locked-test evaluation

The optional single-request runtime remains disabled unless
`TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT=true` is set explicitly and the frozen
activation evidence passes. It is not registered as the default runtime.
The Stage146 concurrent runtime is available only through the explicit
`TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT=true` application bootstrap with a
compliant Stage145 aggregate. It remains off by default and is mutually
exclusive with the single-request runtime flag. Neither runtime is registered
as the default. Stage148 adds a transport-neutral facade over the eligible
concurrent runtime, but it is still not a network service and is not registered
as the default. Stage149 freezes the exact local HTTP surface around that
facade. Stage150 implements the FastAPI adapter and validates it with
in-process ASGI calls plus a temporary real loopback socket. Stage151 freezes
the process composition protocol. Stage152 implements its strict local service
entrypoint and validates one real resource and HTTP lifecycle. Stage153 freezes
the request-local tool workflow. Stage154 installs LangGraph 1.2.9, implements
the framework-neutral and `StateGraph` engines, and routes the concurrent
runtime through the compiled graph. Stage155 requires current Stage154 evidence
before resource construction or socket binding and emits content-free timing
events for every graph node. The entrypoint remains disabled by default,
binds loopback only after both explicit flags are true, and no network service
remains running after validation.

The Stage150 implementation and formal evidence are recorded in
[docs/primeqa_hybrid_agent_http_transport_validation.md](docs/primeqa_hybrid_agent_http_transport_validation.md).
Stage151 freezes the process entrypoint, explicit port, canonical
source order, label-free synthetic warmup, single-worker socket ownership,
Uvicorn signal behavior, shutdown ordering, startup exit codes, and public
startup-event boundary. Stage152 implements and validates that contract. The
Stage151 protocol is recorded in
[docs/primeqa_hybrid_agent_service_entrypoint_protocol.md](docs/primeqa_hybrid_agent_service_entrypoint_protocol.md).
The implementation evidence is recorded in
[docs/primeqa_hybrid_agent_service_entrypoint_validation.md](docs/primeqa_hybrid_agent_service_entrypoint_validation.md).
The Stage154 graph implementation and current-code lifecycle evidence are
recorded in
[docs/primeqa_hybrid_agent_tool_workflow_validation.md](docs/primeqa_hybrid_agent_tool_workflow_validation.md).
The strict activation and operational observation evidence is recorded in
[docs/primeqa_hybrid_agent_runtime_observability_validation.md](docs/primeqa_hybrid_agent_runtime_observability_validation.md).

Run the non-default local service only with both explicit activation flags and
an explicit non-privileged port:

```powershell
$env:TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT = "true"
$env:TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT = "true"
python -m ts_rag_agent.local_agent_service --port 18152
```

## Quickstart

```powershell
cd C:\d_desktop\profile\technical_support_rag_agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -e ".[agent,app,dev,data]"
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
