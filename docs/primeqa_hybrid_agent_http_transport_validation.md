# PrimeQA Hybrid Agent HTTP Transport Validation

## Scope

Stage150 implements the Stage149 protocol as a disabled-by-default FastAPI
adapter around the Stage148 transport-neutral facade. The implementation is
strictly local and HTTP/1.1 only. It does not define a persistent service
entrypoint, register the runtime as the default, expose a remote interface, or
open the locked test split.

The formal validator reads the saved Stage149 public aggregate, runs synthetic
in-process ASGI cases, and opens one temporary operating-system-assigned
`127.0.0.1` socket for a real HTTP/1.1 smoke test. It does not load questions,
documents, models, indexes, or candidate pools.

## Runtime Activation

The new setting is:

```text
TS_RAG_ENABLE_LOCAL_AGENT_HTTP_TRANSPORT=false
```

Only strict boolean values are accepted. Enabling this flag without also
enabling the concurrent sidecar runtime is a configuration error. Even when
both flags are true, the transport creates its facade only from an eligible,
warm, active concurrent-runtime bootstrap result. The adapter never constructs
or owns model, index, or retrieval resources.

The lifecycle is monotonic:

```text
created -> accepting -> draining -> closed
disabled -> not_active
```

FastAPI lifespan starts the facade and shuts it down. Shutdown first rejects
new requests, then waits naturally for in-flight requests to complete. There
is no application shutdown timeout, forced cancellation, retry, queue, or
fallback path.

## HTTP Surface

Exactly three routes are registered:

```text
POST /v1/agent/answers
GET  /health/live
GET  /health/ready
```

OpenAPI and documentation routes are disabled. The Uvicorn configuration is
fixed to `127.0.0.1`, one worker, h11 HTTP/1.1, no WebSocket implementation, no
proxy-header trust, no server header, and no default access log.

The answer endpoint requires `Content-Type: application/json` with optional
UTF-8 charset. It rejects duplicate JSON keys, unknown fields, and type
coercion. Limits are enforced before facade dispatch:

```text
raw body:       32,768 bytes
request_handle:    128 characters
title:             512 characters
text:           24,576 characters
```

The body cap is checked from a declared content length and again while reading
the ASGI body stream. A body exactly at the cap is accepted; declared and
streamed overflow both return 413.

## Status And Privacy Contract

Observed status mapping:

```text
complete / verified refusal: 200
malformed JSON:               400
body too large:               413
unsupported media type:       415
invalid request:              422
capacity / lifecycle:         503
unknown downstream error:     500
unknown route:                404
wrong method:                 405
```

Unknown downstream exception text is not returned. Framework errors use the
same stable `error.code/error.message` envelope. Verified refusal remains a
successful domain response with zero citations.

Default Uvicorn access logging is disabled. Application transport events have
an exact 18-field public allowlist. Request handles, request/response content,
citations, document references, headers, cookies, client address, user agent,
and exception text are not present in public events or in the formal report.

## Concurrency And Disconnects

The transport owns four admission permits and a four-worker executor for the
synchronous facade call. The event loop is not used to execute retrieval or
answer generation. Four blocked synthetic calls are admitted; a fifth request
is rejected immediately with `503 capacity_exceeded`. Observed application
waiting, queue, retry, and fallback counts are all zero.

After body parsing and before synchronous dispatch, the adapter probes the
ASGI disconnect state. A known pre-dispatch disconnect produces no ASGI
response frame, records one cooperative pre-dispatch cancellation, and makes
zero runtime calls. This does not claim that the remaining check-to-dispatch
race has been eliminated or that in-flight synchronous work can be hard
cancelled.

## Real Socket Validation

The formal socket smoke test pre-bound an operating-system-assigned loopback
port, passed that listener to Uvicorn, and waited for server startup without an
application timeout. It then observed:

```text
HTTP version: HTTP/1.1
liveness / readiness / answer: 200 / 200 / 200
unsupported media: 415
runtime calls: 1
access log / server header / proxy headers: false / false / false
```

Shutdown was requested and the server thread was joined naturally. The
transport reached `closed`, the socket was closed, and a fresh socket could
bind the same port. The port number is intentionally absent from the public
artifact. No service remains running after validation.

## Formal Result

The unconfirmed preflight used the real Stage149 aggregate but did not execute
ASGI or socket validation:

```text
source checks: 6 / 7 passed
failed check: stage150_user_confirmed
in-process / socket executed: false / false
```

The user-confirmed formal run produced:

```text
Stage149 source guards: 39 / 39
Stage150 guards: 37 / 37
routes: 3 exact / 0 unexpected
overload: 4 completed / 1 immediate capacity rejection
application waiting / queue / retry / fallback: 0 / 0 / 0 / 0
disconnect frames / runtime calls: 0 / 0
public log fields: 18
real socket HTTP/1.1: true
server stopped / transport closed / port rebound: true / true / true
closed transport request: 503 facade_closed
formal time: 0.300549s
train / dev / test loaded: false / false / false
models / indexes / candidate pools: false / false / false
```

The saved Stage149 source SHA-256 is
`d25d5867c325f719a0a22c22d25e1a63a38b88d3af198cc9dab469584baa3197`.
The Stage150 report SHA-256 from the formal run is
`0f380553bc8602c679b56568ed939b051badc84ee3cd0a468ba1be85611e1403`.

Ten formal SVG files were generated and parsed as XML successfully. Artifacts
remain ignored by Git.

## Execution Corrections

- The first editable-install command was cut off by the shell runner's short
  default execution allowance before installation completed. It was rerun with
  enough execution allowance and allowed to finish naturally. Installed
  versions used by the formal result are FastAPI `0.139.2`, Starlette `1.3.1`,
  Uvicorn `0.51.0`, HTTPX `0.28.1`, and Pydantic `2.13.4`.
- The first concurrency test design used four operating-system threads against
  one synchronous `TestClient` portal. Eighteen ordinary tests and the direct
  ASGI disconnect case completed, but both test invocations then stopped making
  CPU progress at the same overload case. After the deadlock was established,
  only the two orphaned pytest processes were terminated. The overload and
  shutdown tests were redesigned with `httpx.AsyncClient` and `ASGITransport`;
  no production behavior or acceptance threshold was changed.
- The redesigned targeted suite first produced `21 passed` while Ruff also
  reported one unused test import. Removing that import and adding the formal
  validator tests produced `26 passed` with Ruff clean.
- A submission review then found a transport lifecycle race not exercised by
  the first formal artifact: shutdown could close the executor between permit
  acquisition and worker submission, turning a lifecycle rejection into a
  generic 500. Executor submission and transport state transition now share one
  synchronization boundary. A closed-transport regression test and a 37th
  formal guard prove the stable `503 facade_closed` result. The earlier 36-guard
  artifacts were regenerated and are not the final evidence.
- The current FastAPI `TestClient` import emits a
  `StarletteDeprecationWarning` recommending a future `httpx2` transition. The
  warning is retained in the record; it did not change the observed behavior or
  formal decision.
- The first combined repository-verification command built its changed-Python
  argument incorrectly in PowerShell. Ruff received several paths as one path
  and reported that it did not exist; later commands still ran and pytest
  passed, so that combined exit code could not be treated as proof of formatting.
  The six Stage150 Python files were then checked with an explicit argument list
  and all six were already formatted.

## Repository Verification

```text
targeted Stage150 tests: 27 passed, 1 dependency deprecation warning
Stage150 Python format check: 6 files already formatted
full repository Ruff: passed
full repository pytest: 613 passed, 1 dependency deprecation warning in 11.82s
git diff --check: passed (with an informational CRLF-to-LF worktree warning)
formal Stage150 guards: 37 / 37
formal / preflight SVG XML parse: 10 / 10 and 10 / 10
formal / preflight artifact ignore checks: passed
```

## Design References

The implementation follows the official
[FastAPI lifespan documentation](https://fastapi.tiangolo.com/advanced/events/),
[FastAPI lifespan testing documentation](https://fastapi.tiangolo.com/advanced/testing-events/),
[Starlette request documentation](https://www.starlette.io/requests/), and
[Uvicorn settings documentation](https://www.uvicorn.org/settings/). These
references informed implementation. The formal validation itself used local
code and the saved local Stage149 aggregate.

## Decision

The disabled local FastAPI transport is implemented and validated. It is not a
persistent service and is not authorized for remote exposure, runtime
defaultization, or locked-test evaluation.

Stage151 should freeze the local service-entrypoint composition protocol:
which public aggregate authorizes startup, how bootstrap and transport
lifetimes compose, which local port input is allowed, how startup failure is
reported, and how the process exits naturally. Implementation of that
entrypoint should wait until the protocol is frozen.
