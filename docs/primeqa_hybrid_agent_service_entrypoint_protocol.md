# PrimeQA Hybrid Agent Service Entrypoint Protocol

## Scope

Stage151 freezes the process-level composition contract that a future local
Agent service entrypoint must implement. It uses only the saved Stage150 public
validation report and synthetic policy cases. It does not implement the
entrypoint, load any split or document, build resources, install signal
handlers, bind a socket, or start a service.

The protocol remains local-research-only. Remote deployment, runtime
defaultization, locked-test evaluation, queues, retries, fallback, reload,
multiple workers, forced cancellation, and an implicit shutdown timeout remain
closed.

## Invocation

The only allowed invocation surface is:

```text
python -m ts_rag_agent.local_agent_service --port <PORT>
```

`--port` is required and has no default. Stage151 defines the allowed range as
`1024..65535`; port zero is forbidden because a persistent local service must
have a caller-known address. This range is a conservative Stage151 engineering
contract, not a production traffic measurement.

There is no `--host`; binding remains exactly `127.0.0.1`. The protocol also
forbids reload, worker-count, UDS, inherited-fd, source-path, encoder-device,
and encoder-batch CLI overrides. Unknown options are rejected. Canonical source
locations come from `ProjectSettings`, so an invocation cannot silently swap
one validation report or document collection for another.

## Source Authorization

The future entrypoint must resolve these six canonical source roles:

```text
stage150_http_transport_validation
stage145_concurrent_runtime_validation
stage128_agent_retrieval_protocol
stage125_recall_expansion_protocol
stage80_dense_sparse_report
primeqa_technote_documents
```

Startup must validate Stage150 before any other source or resource work. It
must then require both explicit settings:

```text
TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT=true
TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT=true
```

Stage145 evidence must recompute as eligible before resource construction.
Failure at either authorization gate builds no resources and binds no socket.
All source files are read-only and fingerprinted.

## Startup Order

The exact process order is:

```text
1.  parse exact CLI
2.  load and validate Stage150 public report
3.  validate explicit runtime and transport flags
4.  load and validate Stage145 public report
5.  load frozen retrieval protocols
6.  construct process resource factory
7.  build shared resources once
8.  run built-in label-free synthetic warmup
9.  create FastAPI app and Uvicorn config
10. prebind exact loopback listener once
11. run Uvicorn Server on the main thread
12. Uvicorn stops accepting and waits for HTTP tasks
13. FastAPI lifespan drains the transport
14. entrypoint finally confirms the listener is closed
15. release process references
```

Resource ownership remains explicit. The entrypoint retains the resource
factory; the bootstrap result retains the active runtime; the transport does
not own models or indexes. The current resource graph has no explicit close
port, so Stage151 does not claim one exists. After server return, the process
releases references without inventing a resource-close operation.

## Label-Free Warmup

The Stage146 bootstrap still types its warmup parameter as the old
label-bearing `PrimeQAQuestion`. A service must not read a train/dev/test row
just to satisfy that signature. Stage151 therefore requires Stage152 to change
the bootstrap warmup input to `PrimeQARuntimeQuery` and use one built-in,
synthetic, label-free warmup query.

The warmup contains no answer, answerability, gold document, split membership,
or other evaluation fields. Its content is not written to the public startup
event. Warmup must complete successfully before a listener is created.

## Socket And Process

After authorization, resource build, and warmup, the entrypoint creates one
`127.0.0.1:<PORT>` listener and prebinds it exactly once before calling
`uvicorn.Server.run(sockets=[listener])`. This removes the bind-time race
between checking a port and starting the server.

A bind failure maps to its stable startup failure and stops. The entrypoint
does not retry, select an alternate port, or fall back to port zero. The
listener is entrypoint-owned and is closed idempotently in `finally`, even
though Uvicorn also closes passed sockets during its shutdown sequence.

The process model is exactly one process and one worker, without reload.
`Server.run` executes on the main thread. Uvicorn owns the platform signals it
supports; the entrypoint installs no competing signal handler. External
signal-derived process exit codes remain platform- and Uvicorn-defined rather
than being falsely normalized into one cross-platform number. A second-signal
force-exit behavior is not overridden or claimed as a natural shutdown.

## Shutdown Ordering

The final protocol follows the observed Uvicorn 0.51.0 implementation order:

```text
stop accepting and close listener
request existing connection shutdown
wait for HTTP tasks naturally
run FastAPI lifespan shutdown
drain and close transport
confirm listener closed
release process references
```

The Uvicorn graceful-shutdown timeout remains `None`, so no application time
limit is introduced. There is no force-cancel path. In-flight work completes or
raises naturally before the process releases its runtime references.

## Exit And Logging Contract

Stage151 assigns stable proposed entrypoint exit statuses for clean return and
startup failures:

```text
0 clean server return
1 unexpected composition failure
2 CLI contract invalid
3 Stage150 authorization rejected
4 activation configuration rejected
5 Stage145 or runtime activation rejected
6 resource or warmup failure
7 socket bind or listen failure
8 server or lifespan failure
```

These are frozen engineering semantics for Stage152 implementation, not
observed process exits in Stage151. Startup failures are not retried. External
signal exit codes are explicitly not normalized.

The public startup event has exactly 18 fields covering phase, outcome, exit
code, binding, source/activation state, resource/warmup/listener/server state,
shutdown trigger, transport state, default status, and zero queue/retry/fallback
counts. Request, response, warmup content, source paths, and exception messages
are absent from this event. Uvicorn's own framework-error logging is a separate
channel and is not falsely described as the public startup event.

## Formal Result

The unconfirmed preflight used the real Stage150 public aggregate. Of 33 guards,
only `stage151_user_confirmed` failed. It loaded no data or resources and did
not bind a port or install signal handlers.

The user-confirmed final result is:

```text
Stage150 source guards: 37 / 37
Stage151 guards: 33 / 33
canonical policy cases: 1 eligible / 5 rejected
source unchanged after freeze: true
required CLI options: 1
optional CLI options: 0
port range: 1024..65535
resource build / warmup / bind attempts: 1 / 1 / 1
bind retry / queue / retry / fallback: 0 / 0 / 0 / 0
worker / process count: 1 / 1
public startup fields: 18
formal time: 0.001723s
train / dev / test loaded: false / false / false
questions / documents / models / indexes / candidate pools: false
service implemented / port bound / signal handlers installed: false / false / false
```

The Stage150 source SHA-256 is
`0f380553bc8602c679b56568ed939b051badc84ee3cd0a468ba1be85611e1403`.
The final Stage151 report SHA-256 is
`ccea1acbcc7afb0ebb874f79db432a93f32fe84fd55b515567e9edd0faf931da`.

Ten formal and ten preflight SVG files were generated, parsed as XML, and
confirmed ignored by Git.

## Execution Correction

The policy, CLI, and tests first passed `28` targeted tests. The first formal
artifact then passed `33/33` guards, but a submission review compared the
protocol wording with installed Uvicorn 0.51.0 source and found that the frozen
shutdown order was reversed. Uvicorn closes its servers and passed sockets,
waits connection/tasks, and only then invokes lifespan shutdown. This was a
protocol-fact error even though all internal guards agreed with one another.

The protocol, tests, and shutdown visualization were corrected to the actual
Uvicorn-then-lifespan order. Public startup-event privacy wording was also
separated from Uvicorn framework-error logging. Targeted tests again passed
`28`, and both preflight and formal artifacts were regenerated. The initial
artifact is not final evidence; this section preserves the true correction
history.

## Repository Verification

```text
targeted Stage151 tests: 28 passed
Stage151 Python format check: 3 files already formatted
full repository Ruff: passed
full repository pytest: 641 passed, 1 dependency deprecation warning in 6.49s
git diff --check: passed (with an informational CRLF-to-LF worktree warning)
formal Stage151 guards: 33 / 33
formal / preflight SVG XML parse: 10 / 10 and 10 / 10
formal / preflight artifact ignore checks: passed
```

## References

The design uses the official [Uvicorn settings](https://www.uvicorn.org/settings/),
[Uvicorn server behavior](https://www.uvicorn.org/server-behavior/), and
[Python 3.10 signal documentation](https://docs.python.org/3.10/library/signal.html).
The exact installed Uvicorn 0.51.0 `Server.startup`, `Server.shutdown`, and
signal-capture implementations were also inspected locally before the final
freeze. Formal Stage151 validation itself did not access the network.

## Decision

The local service-entrypoint composition protocol is frozen and Stage152 may
implement it. That implementation must first refactor bootstrap warmup to the
label-free runtime query type, then validate all fail-closed startup outcomes
with synthetic resources before one real local-resource service lifecycle.
Remote serving, defaults, test, queues, retries, and fallback remain closed.
