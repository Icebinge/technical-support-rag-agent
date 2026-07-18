# PrimeQA Hybrid Agent Tool Workflow Validation

## Scope

Stage154 implements the deterministic tool workflow frozen by Stage153 and
integrates it into the existing concurrent runtime, facade, FastAPI adapter,
and local service composition. The implementation remains disabled by default
through the existing runtime and transport activation flags.

This stage does not add autonomous LLM tool selection, query rewrite, repeated
retrieval, memory, a checkpointer, persistence, cache, streaming, human
interrupts, queues, retries, fallback, remote exposure, runtime defaultization,
or test evaluation.

## Dependencies

The `agent` extra now has one exact direct dependency:

```toml
agent = [
  "langgraph==1.2.9",
]
```

Full `langchain` and `langchain-community` are not direct dependencies.
LangGraph installs `langchain-core` transitively. The installation completed
once without retry and changed the pre-existing `websockets` version from 16.1
to 15.0.1 to satisfy `langgraph-sdk<0.5.0`. `pip check` reports no broken
requirements.

The exact recorded package set is:

```text
langgraph==1.2.9
langchain-core==1.4.9
langchain-protocol==0.0.18
langgraph-checkpoint==4.1.1
langgraph-prebuilt==1.1.0
langgraph-sdk==0.4.2
langsmith==0.10.6
orjson==3.11.9
ormsgpack==1.12.2
requests-toolbelt==1.0.0
sniffio==1.3.1
tenacity==9.1.4
uuid-utils==0.17.0
websockets==15.0.1
xxhash==3.8.1
```

`tomli` is declared only in the development extra for structured TOML parsing
on Python 3.10. Quickstart now includes the `agent` extra because the active
concurrent request path imports and executes the graph adapter.

## Architecture

The implementation has three layers:

1. `PrimeQAHybridAgentToolset` owns the three authorized tools and deterministic
   context preparation/diagnostic assembly.
2. `AgentToolWorkflowNodeExecutor` owns the exact node semantics and frozen
   state transitions.
3. `DeterministicAgentToolWorkflowEngine` and
   `LangGraphAgentToolWorkflowEngine` execute the same node methods.

The graph contains the seven Stage153 nodes:

```text
validate_request
retrieve_candidate_pool
prepare_context
compose_grounded_answer
verify_grounded_answer
observe_diagnostics
finalize_response
```

`StateGraph` is compiled once per workflow instance. Every invocation creates a
new 13-field private state. The only conditional graph edge follows
`observe_diagnostics` and chooses complete or refuse before the common final
node. No checkpointer or cache is attached.

The successful contract remains one retrieval, one answer composition, and one
verification. Context authority remains Top400 candidate pool, Top10 generation
context, rank-200 verification prefix, and three diagnostic sidecar slots. The
20-field public workflow trace contains no request, answer, document, or
exception text.

## Runtime Integration

`create_primeqa_hybrid_concurrent_sidecar_agent_runtime` now constructs the
LangGraph workflow instead of the older fixed entrypoint object. Existing
runtime, facade, HTTP, and service response types remain compatible through
execution ports. The sidecar result assembler is shared by the old entrypoint
and the new workflow, avoiding two implementations of answer/verification
telemetry.

LangGraph executes nodes in copied contexts. The previous concurrent retriever
profiling mechanism wrote a `ContextVar` inside the entrypoint and consumed it
outside. That update was not visible after graph invocation. It was replaced
with a caller-created private token inherited by graph nodes and a locked
token-to-profile map consumed by the caller. The token and profile never enter
public telemetry.

The same copied-context boundary affected failure snapshots. Tool exceptions
now remain unwrapped and are indexed temporarily by the inherited invocation
token plus the original exception object. The outer workflow consumes and
deletes the snapshot when the exact same exception propagates. No exception is
swallowed, converted, retried, or used to select another path.

Every node validates its transition before tool execution. Directly invoking
an out-of-order node therefore produces zero tool calls and no state mutation.

## Synthetic Validation

The formal validator executes:

- complete and refuse equality between the framework-neutral and LangGraph
  engines;
- exact seven transitions and three tool calls;
- exact Top400, Top10, rank-200, and three-sidecar context counts;
- invalid-transition rejection before retrieval;
- identity-preserving retrieval exception propagation with no public error
  message;
- four simultaneous graph invocations with request-isolated candidates and
  one graph compilation;
- an in-process FastAPI live, ready, and answer request through the actual
  concurrent runtime and graph factory.

The synthetic HTTP result is `200/200/200` with a Top400 runtime trace. No
train, dev, or test row is loaded. Synthetic documents are generated only in
memory and their content or identifiers are not written to the report.

## Real Lifecycle

After explicit user confirmation, one current-code Stage152 support lifecycle
ran on fixed port `18154`. It was not retried and had no alternate port,
application timeout, forced cancellation, or process restart.

```text
wall-clock process time: 52.3s
reported real lifecycle: 49.813499s
Stage152 support guards: 46 / 46
HTTP/1.1 live / ready / answer: 200 / 200 / 200
answer refused: false
citations: 3
exit code: 0
listener released: true
transport closed: true
test metrics run: false
```

The support artifact does not store the real request's candidate-pool depth.
Stage154 therefore does not claim or backfill that value from another source;
Top400 is directly observed only in the separate synthetic HTTP evidence. The
support lifecycle does prove that the current graph-integrated code loads the
real technote resources, models, and indexes and completes the local HTTP and
shutdown path.

## Formal Result

The corrected unconfirmed preflight passes `46/54` guards. Its eight failures
are the user-confirmation guard and seven groups of unavailable real-support
evidence. The final confirmed formal report passes `54/54` in `0.204824s` after
reading the separately completed real support artifact.

```text
formal artifact SHA-256:
ed7ce5b99be2c31e045d1c20d0f0965f2cbf99c0f5c89414158c2bab8c773a5d

preflight artifact SHA-256:
4020845a899421d61b06c7673d918e93f3707d39447bf9b624aa35e7012f63f1

current-code real support artifact SHA-256:
639a71ac921ebec2c06378fb06e9f48eb4798bb38612a6682f391901e34e8e40
```

Formal and preflight each contain ten parseable SVG files. Full repository
validation finishes with Ruff passing and `700 passed`; one existing FastAPI
`TestClient` deprecation warning remains visible.

The final independent audit matches all four Stage154 formal source
fingerprints and all six real-resource support fingerprints, parses 30 SVGs
across formal/preflight/support, and successfully rebinds port 18154 after the
service exits.

An earlier formal artifact passed `54/54` in `0.165285s`, but a later
maintainability change replaced duplicated protocol/graph ID strings in the
concurrent runtime with public Stage153 constants. Because that file is a
formal fingerprint source, formal and preflight were regenerated. The earlier
SHA values are process history, not final evidence; the values above match the
committed source.

## Decision

The deterministic LangGraph Agent workflow is implemented and is now the
execution path used by the explicitly activated concurrent runtime. It is not
the default runtime and is not remotely exposed. Stage155 may freeze runtime
activation evidence and operational observability for the graph workflow.
Test, remote serving, defaultization, queues, retries, and fallback remain
closed.

## Stage155 Compatibility Note

The SHA values above remain the historical Stage154 completion evidence for
commit `9815a79`. Stage155 later added workflow observation and a stricter
service activation chain, changing two Stage154 fingerprint sources. Stage155
therefore regenerated a current-source compatibility formal (`54/54`) and
preflight (`46/54`) by reading the same previously completed real support
artifact; it did not rerun the Stage154 real lifecycle. The compatibility
hashes and that process distinction are recorded in
`docs/primeqa_hybrid_agent_runtime_observability_validation.md`.
