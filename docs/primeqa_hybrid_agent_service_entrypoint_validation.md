# PrimeQA Hybrid Agent Service Entrypoint Validation

## Scope

Stage152 implements the Stage151 non-default local service entrypoint. It
composes the previously validated concurrent runtime and FastAPI transport into
one strict process root. It does not enable remote serving, register the runtime
as the default, open the locked test split, add a queue, retry a startup failure,
select an alternate port, or add a fallback strategy.

The production invocation is exactly:

```text
python -m ts_rag_agent.local_agent_service --port <PORT>
```

`--port` is required and accepts only decimal ASCII values in `1024..65535`.
Host, worker, reload, source-path, encoder, UDS, and inherited-fd overrides are
rejected. The host remains fixed at `127.0.0.1`.

## Composition

The implementation validates the saved Stage150 report before activation. Both
settings must then be explicitly true:

```text
TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT=true
TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT=true
```

After loading Stage145, the existing concurrent bootstrap recomputes its
eligibility before resource construction. The service fingerprints all six
canonical sources, constructs `PrimeQAHybridProcessRuntimeResourceFactory`,
builds the process graph once, and performs one warmup. The bootstrap warmup
signature now requires `PrimeQARuntimeQuery`; the built-in query contains only
`id`, `title`, and `text`, so startup does not load a train, dev, or test
question for warmup.

Only after warmup succeeds does the entrypoint create the FastAPI app and
Uvicorn config. It binds and listens on the exact requested loopback port once,
passes that socket to `uvicorn.Server.run` on the main thread, and closes the
listener idempotently after server return. Uvicorn retains ownership of its
supported platform signal handlers. There is no application shutdown timeout
or forced cancellation.

## Failure Contract

Nine synthetic cases exercised the frozen exit mapping:

```text
clean server return                     0
CLI contract invalid                    2
Stage150 authorization rejected         3
activation configuration rejected       4
Stage145/runtime activation rejected    5
resource or warmup failure              6
socket bind or listen failure           7
server or lifespan failure              8
```

The separate unexpected-composition value remains `1` and is covered by unit
tests. All synthetic terminal events used the exact 18-field allowlist. The
clean case observed one factory construction, one resource build, one warmup,
one bind, one server run, one listener close, and one terminal event. Queue,
retry, and fallback counts were all zero.

The exact production module command was also invoked once with both activation
flags explicitly false. Stage150 authorized first, then the process returned
exit code 4 with `resources_initialized=false` and `listener_bound=false`; it
did not build resources or start a service.

## Real Lifecycle

The formal run used one fixed port, `127.0.0.1:18152`, and the actual local
technote corpus, dense models, lexical indexes, Stage145 bootstrap, FastAPI app,
and Uvicorn server. A validation-only client thread connected to the already
prebound listener with no HTTP timeout. The real Uvicorn server itself ran on
the main thread. After live, ready, and answer completed, the client requested
normal Uvicorn shutdown through `should_exit`; the validator then waited for
the server and client to finish without a monitoring deadline.

```text
real lifecycle time: 51.098075s
HTTP versions: HTTP/1.1 / HTTP/1.1 / HTTP/1.1
live / ready / answer: 200 / 200 / 200
answer refused: false
answer citation count: 3
terminal event fields: 18
canonical source fingerprints: 6
listener released after shutdown: true
transport closed: true
process exit code: 0
```

The probe request handle, question text, answer text, and citation identities
were not persisted in the Stage152 report. The report records only status,
schema, aggregate citation count, lifecycle state, and source fingerprints.

No train, dev, or test question split was loaded. The technote document corpus
was loaded because this was a real serving lifecycle, not an evaluation run.
No test metric was computed.

## Formal Result

```text
Stage151 source guards: 33 / 33
Stage152 guards: 46 / 46
synthetic cases: 9 / 9
real resource lifecycle: passed
Ruff: passed
pytest: 658 passed, 1 dependency deprecation warning
formal SVG XML parse: 10 / 10
preflight SVG XML parse: 10 / 10
```

The formal artifact is
`artifacts/primeqa_hybrid_agent_service_entrypoint_validation_stage152.json`
with SHA-256
`7976dfd7d19251f013d2bb246da3e0302e8ecadd26a23baf3418d0d63498566b`.

## Execution Record

The first read-only preflight completed its source and synthetic logic, but its
SVG write failed because the new chart wrapper passed `data=` to the existing
helper, whose real parameter is `bars=`. That invocation produced no preflight
JSON and is not reported as successful. The wrapper was corrected and the
preflight was rerun. It then passed the Stage151 source gate and all 9 synthetic
cases; real-lifecycle guards remained false because confirmation was
deliberately disabled.

The formal real-resource lifecycle was executed once and completed naturally.
It was not retried. A later audit found that the preflight-only SHA guard used
vacuous `all(...)` truth when no real fingerprints existed. The guard was
tightened to require exactly six fingerprints and the read-only preflight was
regenerated. This did not change the already observed formal result because the
formal run had six valid SHA-256 fingerprints.

The first targeted formatting check after implementation reported two files
that needed formatting. Ruff lint and 74 targeted tests in the same command did
pass, but that combined command is not described as all green. After running
the formatter, format check and lint passed. Final full-repository validation
then produced 657 passing tests and the existing Starlette `TestClient`
deprecation warning. A final self-review added the missing direct unit case for
unexpected composition exit code `1`; targeted tests became 75 and the full
suite became 658.

A later repository-wide `ruff format --check .` reported 311 legacy files that
would be reformatted. Those unrelated files were not bulk rewritten. The
Stage152 file set passes its targeted format check; full-repository Ruff lint
still passes.

## Learned

- A serving warmup should be a serving-shape type, not an evaluation type whose
  gold fields happen to be emptied at runtime.
- Prebinding the exact listener lets the composition root own the only bind
  attempt and pass the same socket to Uvicorn without a check-then-bind race.
- Synthetic exit-path coverage and one real resource lifecycle prove different
  facts; both are needed before calling a process entrypoint implemented.
- A no-timeout shutdown claim requires observing natural HTTP completion,
  FastAPI lifespan closure, listener release, and process return rather than
  merely setting a shutdown flag.

Next: Stage153 should freeze the local Agent tool-orchestration protocol before
introducing LangGraph or another graph runner. The service remains non-default,
loopback-only, and outside the locked-test gate.
