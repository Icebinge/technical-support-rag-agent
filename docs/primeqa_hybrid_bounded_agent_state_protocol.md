# Stage 156 Bounded Agent Tool Selection And Volatile State Protocol

## Objective

Stage 156 freezes the first genuinely dynamic local Agent boundary without
granting the model an autonomous tool loop. It reads only the saved Stage 155
public aggregate, executes synthetic policy and state-isolation cases, and
keeps the evaluation test set closed.

The protocol is implemented in:

```text
src/ts_rag_agent/application/primeqa_hybrid_bounded_agent_state_protocol.py
```

The formal artifact is:

```text
artifacts/primeqa_hybrid_bounded_agent_state_protocol_stage156.json
```

## Dynamic Decision Boundary

Retrieval remains a system-required operation and runs exactly once before the
dynamic decision. After context preparation, a future structured model router
may make exactly one choice:

```text
compose_grounded_answer | refuse_insufficient_evidence
```

The model cannot call retrieval, rewrite the query, request a second retrieval,
create a tool, invoke tools in parallel, or enter a decision loop. On the
compose branch, answer composition, verification, and diagnostics each run
exactly once before verified finalization. On the refuse branch, none of those
three operations runs; a fixed system refusal constructor finalizes the result.
Diagnostics are a system-owned read-only step, not a model-selectable tool. A
composed response may reach the user only through the existing system verifier.

This is a bounded structured router, not a `create_agent` or `ToolNode` loop.
The distinction follows the current official LangGraph description of routing,
workflows, and agents.

## Multi-Turn State Boundary

`VolatileThreadStateLedger` provides an executable process-local state contract.
Each thread must be opened with an opaque handle and is accessed only by exact
handle. Only completed terminal turns may cross turn boundaries:

```text
opaque thread handle
completed turn sequence
user turn input
verified terminal response
terminal state
```

Candidate pools, generation and verification contexts, unverified generated
responses, verification details, diagnostics, exception details, and model
internal reasoning are turn-local and must be discarded.

The ledger requires callers to provide both a positive completed-turn limit and
a positive retained-byte limit. Stage 156 does not invent production values for
those limits. Overflow, invalid sequence, missing thread, and nonterminal turn
commits are rejected before mutation. There is no silent truncation, eviction,
implicit thread creation, state reconstruction, retry, or fallback. Explicit
close deletes the thread state. Process restart loses all state because neither
a LangGraph checkpointer nor a persistent store is selected.

## Validation

The unconfirmed preflight result is:

```text
guards: 42 / 43
failed: stage156_user_confirmed
```

After the user's explicit instruction to proceed with the next major stage, the
formal result is:

```text
Stage155 source guards: 57 / 57
Stage156 formal guards: 43 / 43
dynamic policy cases: 2 eligible / 6 rejected
thread-state cases: 5 / 5 eligible
targeted tests: 24 passed
current-source full repository tests: 736 passed, 1 existing warning in 7.31s
formal/preflight visualizations: 10/10 and 10/10 parseable SVG
```

The formal artifact SHA-256 is
`1057cd70ed0ce872529bdc04d1182b84327a50cf6f9bcce9fedb76a4f2952a97`.

The accepted dynamic cases are the bounded compose and bounded refuse paths.
The rejected cases cover an unauthorized tool, repeated retrieval, a decision
loop, model-owned final authority, hidden retry/fallback, and default/remote/test
activation. Thread cases prove retention of a completed turn, cross-thread
isolation, byte-overflow rejection without mutation, implicit-creation
rejection, and close-time clearing.

Synthetic strings and synthetic limits used by unit cases are test fixtures,
not observed runtime traffic or selected production values. They are not saved
in the public artifact.

The first attempt to start the full repository suite passed `timeout_ms=0` to
the shell tool under the mistaken assumption that zero meant unlimited. The
tool instead stopped the command after about 14 milliseconds with exit 124 and
produced no pytest output. No pytest process remained. This was recorded before
any replacement run. After explicit user confirmation, exactly one hidden
pytest process ran without a test or wait timeout and exited naturally with
`736 passed, 1 warning in 12.72s`. The warning is the existing FastAPI
`TestClient` Starlette deprecation warning.

A later pre-commit node audit found and corrected an unrealizable early-refusal
diagnostics contract. The `12.72s` full-suite result therefore remains a
historical result for the pre-correction source. The corrected current source
then ran in exactly one new hidden pytest process, with no test or wait timeout,
and exited naturally with `736 passed, 1 warning in 7.31s`.

## Closed Boundaries

Stage 156 did not change the current runtime, select or call a model, load an
index, corpus, candidate pool, or evaluation row, bind a port, enable remote
access, make the runtime default, or run a test metric. Query rewrite, repeated
retrieval, persistence, queues, retries, and fallback remain closed.

Stage 157 may implement the structured decision-router port against the existing
LangGraph workflow. Before runtime activation it must explicitly select the
local model/provider and exact thread-history limits, then validate structured
schema compliance, complete/refuse routing, state isolation, errors, and source
fingerprints on synthetic and non-test runtime probes.
