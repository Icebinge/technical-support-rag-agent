# PrimeQA Hybrid Agent Tool-Orchestration Protocol

## Scope

Stage153 freezes the request-local orchestration contract that may be
implemented in Stage154. It reads only the saved public Stage152 and Stage139
aggregate reports. It does not load train, dev, or test questions; documents;
models; indexes; or candidate pools. It does not install LangGraph, implement a
workflow, bind a port, change defaults, or enable queues, retries, or fallback.

The frozen design is deliberately described as a deterministic tool workflow,
not an autonomous Agent. The path is predetermined; an LLM does not select
tools, create loops, rewrite the query, or trigger a second retrieval.

## Workflow Contract

The graph has nine states:

```text
received -> validated -> retrieved -> context_prepared -> answered
         -> verified -> observed -> complete | refuse
```

It contains seven nodes and eight allowed transitions. Each successful request
uses exactly seven transitions. The only conditional edge starts at
`observed` and chooses the terminal `complete` or `refuse` state. The graph is
acyclic, and an invalid transition raises before state or trace mutation.

The three sequential tools are:

1. `retrieve_candidate_pool`
2. `compose_grounded_answer`
3. `verify_grounded_answer`

Each tool may be called at most once on a successful path. Tool errors propagate
unchanged to the private execution boundary. Exception messages do not enter
the public trace. There is no in-graph timeout, retry edge, alternate tool, or
fallback route.

## Context Authority

The protocol preserves the validated Stage139 answer path:

```text
candidate pool: Top400
generation context: Top10
verification prefix: rank <= 200
sidecar observation slots: 3
final response source: verified answer only
```

The sidecar may observe diagnostics but may not generate the answer, verify it,
replace the primary context, or replace the final response. Query rewrite and a
second retrieval are closed.

## State And Telemetry

Each invocation receives an independent private state. The graph is compiled
once per process, but request state is never shared between invocations. The
private contract has 13 fields for query, contexts, answer, verification,
terminal response, transition state, tool counts, and failure stage.

The public trace has exactly 20 allowlisted fields. It exposes only protocol and
terminal status, aggregate transition/tool/context counts, verified refusal and
citation counts, diagnostic status, failure stage, and zero queue/retry/fallback
counts. It excludes the request handle, question, answer text, document content,
document identifiers, and exception messages.

The existing outer runtime remains nonblocking at concurrency four. Stage153
does not add an in-graph queue or alter service capacity semantics.

## Framework Decision

Official LangGraph documentation was checked on 2026-07-18. The facts used are:

- LangGraph is a low-level orchestration framework and can be used without
  LangChain.
- `StateGraph` is built from shared state, nodes, and edges and must be compiled
  before invocation.
- Official documentation distinguishes predetermined workflows from agents that
  dynamically choose tools.
- `ToolNode` includes tool execution and error-handling behavior, so it is not
  selected while hidden recovery is prohibited.

The latest PyPI release observed during research was LangGraph 1.2.9. Neither
`langgraph` nor `langchain` was installed locally, and Stage153 did not install
either package. The next adapter candidate is `langgraph.graph.StateGraph`;
Stage154 must install and record exact local versions and prove that the adapter
preserves this framework-neutral contract.

Sources:

- <https://docs.langchain.com/oss/python/langgraph/overview>
- <https://docs.langchain.com/oss/python/langgraph/graph-api>
- <https://docs.langchain.com/oss/python/langgraph/workflows-agents>
- <https://docs.langchain.com/oss/python/langchain/tools>
- <https://pypi.org/project/langgraph/>

## Validation

The unconfirmed preflight passed `45/46` guards; only
`stage153_user_confirmed` failed. After the user's explicit confirmation, the
formal run passed `46/46` guards in `0.002492s`. One exact deterministic case
was eligible and six unsafe variants were rejected. Formal and preflight each
generated ten parseable SVG files.

```text
formal status: primeqa_hybrid_local_agent_tool_orchestration_protocol_frozen
states / nodes / transitions: 9 / 7 / 8
tools / successful tool calls: 3 / 3
private / public fields: 13 / 20
formal artifact SHA-256:
eb984f9f1e023048ba564faa85870d188f0d6cc3447181ac8171ddc616cbbebf
```

The first targeted test run was truthfully `28 passed, 1 failed`: the test
expected 47 guards while the implementation defined 46. No artificial guard
was added. The expected count was corrected to the real contract, after which
the targeted suite passed `29/29`.

Final repository validation passed targeted formatting for all three Stage153
Python files, full Ruff lint, and `687` pytest tests. One existing FastAPI
`TestClient` dependency deprecation warning remains visible; it is unrelated to
the Stage153 protocol and was not suppressed.

Three later independent audit snippets initially failed because the first two
assumed the JSON policy/check collections had the wrong container types and the
third assumed a source fingerprint contained a path field. They did not modify
either artifact. After reading the actual schema and mapping the two named
sources explicitly, the final audit confirmed formal `46/46`, preflight
`45/46`, one eligible and six rejected cases, source fingerprints, both report
SHA-256 fingerprints, and 20 unique parseable SVG files across the two runs.

## Decision

Stage153 authorizes Stage154 to implement the deterministic workflow and a
LangGraph `StateGraph` adapter. It does not authorize autonomous tool choice,
memory, persistence, streaming, human interruption, query rewrite, repeated
retrieval, remote serving, runtime defaultization, test evaluation, queues,
retries, or fallback.
