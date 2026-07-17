# PrimeQA Hybrid Agent Network Transport Protocol

## Scope

Stage149 freezes the HTTP boundary around the Stage148 transport-neutral
facade. It reads only the saved Stage148 public aggregate and runs synthetic
policy cases. It does not import or instantiate FastAPI, bind a port, load a
split, build a candidate pool, load a model or index, change a runtime default,
or access test.

This is a protocol result, not a claim that network serving already exists.
The `32 KiB` body limit and field limits are conservative Stage149 engineering
choices for a local research service. They are not measurements of production
traffic and were not selected on train, dev, or test rows.

## HTTP Surface

The first implementation is limited to loopback `127.0.0.1`, HTTP/1.1, and
three exact routes:

```text
POST /v1/agent/answers
GET  /health/live
GET  /health/ready
```

Remote exposure, TLS termination, authentication, CORS, OpenAPI/interactive
docs, streaming responses, WebSockets, and unversioned Agent routes are not
authorized. This local-only boundary avoids pretending that an unauthenticated
research service is ready for LAN or Internet deployment.

## Request And Response

Only `application/json` with UTF-8 content is accepted. The request schema is
strict and coercion-free:

```text
request_handle: required, nonblank, maximum 128 characters
title: optional/null, maximum 512 characters
text: required, nonblank, maximum 24,576 characters
unknown fields: rejected
raw body: maximum 32,768 bytes
```

The body cap must be enforced twice: reject an oversized `Content-Length`
before parsing, and enforce the same hard cap while streaming the body so a
missing or false header cannot bypass it. Oversized input is rejected; it is
never truncated and processed as a different question.

A successful transport response maps the Stage148 private facade result
exactly: `request_handle`, `text`, `refused`, and `citations`. Citation fields
remain `document_reference`, `title`, `rank`, and `evidence_score`. A verified
domain refusal is a valid answer outcome and therefore remains HTTP `200`, not
an infrastructure error.

Error bodies have one stable envelope:

```json
{
  "error": {
    "code": "stable_machine_code",
    "message": "stable_nonprivate_message"
  }
}
```

## Status Mapping

```text
malformed JSON                       -> 400 malformed_json
body over 32 KiB                     -> 413 request_body_too_large
unsupported media type               -> 415 unsupported_media_type
schema/facade input invalid           -> 422 invalid_request
facade inactive                      -> 503 facade_not_active
runtime capacity full                -> 503 capacity_exceeded
facade draining                      -> 503 facade_draining
facade closed                        -> 503 facade_closed
unexpected downstream failure        -> 500 internal_error
known client disconnect pre-dispatch -> no fabricated HTTP response
```

Capacity is a transient service-availability failure, not a domain refusal and
not an answer payload. The transport does not queue or automatically retry it.
Unexpected exception content is not returned publicly; the facade may preserve
the original object internally while the HTTP boundary emits generic `500`.

## Disconnect And Lifespan

After reading and validating the body, the transport checks client connection
state immediately before dispatch. A known pre-dispatch disconnect sets the
existing cooperative signal and reaches the runtime zero times. It does not
attempt to send a response to a known-disconnected client.

Stage149 explicitly records that this check cannot eliminate the race between
the check and synchronous dispatch. Once dispatch starts, the current runtime
has no hard-cancellation port. A later disconnect does not make the work
magically cancelled: it completes or raises naturally, and its real terminal
outcome is retained.

FastAPI lifespan will create one facade from an active Stage146 bootstrap per
process. Shutdown calls facade shutdown, enters draining, rejects new work,
and waits naturally for in-flight calls to finish. There is no request timeout,
implicit graceful-shutdown timeout, or force-cancel in the application
contract. The process bootstrap, not the transport, retains ownership of model,
index, and runtime resources.

The design follows current primary documentation:
[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[Starlette request disconnect detection](https://www.starlette.io/requests/),
[FastAPI custom error handling](https://fastapi.tiangolo.com/tutorial/handling-errors/),
and [Uvicorn settings](https://www.uvicorn.org/settings/).

## Health And Logging

`/health/live` returns `200` based on process availability and never loads or
probes models. `/health/ready` returns `200` only while the facade is
`accepting`; disabled, rejected, draining, or closed states return `503`.
Neither endpoint loads questions or documents.

Default Uvicorn access logging is disabled because it includes data outside
the Stage149 allowlist. Structured transport logs may contain only route,
method, status, stable outcome, facade/runtime aggregate state, candidate depth,
latencies, and the existing zero queue/retry/fallback counters. Request and
response content, handles, citations, document identifiers, headers, cookies,
client addresses, user agents, and public exception messages are forbidden.

## Formal Result

The unconfirmed preflight produced the complete static protocol but correctly
rejected it on exactly one guard:

```text
failed check: stage149_user_confirmed
network service / port: false / false
train / dev / test loaded: false / false / false
```

The user-confirmed formal result is:

```text
Stage148 source guards: 37 / 37
Stage149 guards: 39 / 39
canonical policy cases: 1 eligible / 5 rejected
source unchanged after freeze: true
raw body limit: 32,768 bytes
field limits handle / title / text: 128 / 512 / 24,576 characters
routes: 3
public logging fields: 18
formal time: 0.001029s
network service / port: false / false
train / dev / test loaded: false / false / false
models / indexes / candidate pools: false / false / false
queue / retry / fallback: false / false / false
public forbidden keys: []
```

The Stage148 source SHA-256 saved by Stage149 is
`2ac205de34cbc2badf05843368f2a8c1e24552c2bbe46216d0f191c58c04a55b`,
matching the file read during the formal run.

## Verification

```text
targeted Stage149 tests: 22 passed
Stage149 Python format check: 3 files already formatted
full repository Ruff: passed
full repository pytest: 586 passed in 6.35s
formal / preflight SVG XML parse: 10 / 10 and 10 / 10
formal / preflight artifact ignore checks: passed
```

The repository-wide formatter check also reported 311 historical Python files
that the current Ruff formatter would rewrite; 131 files already matched. All
three Stage149 Python files pass format checking. The historical files were not
mass-formatted as unrelated Stage149 churn.

## Visualizations

```text
stage149_source_gate.svg
stage149_http_surface.svg
stage149_request_limits.svg
stage149_status_mapping.svg
stage149_disconnect_boundary.svg
stage149_lifespan_boundary.svg
stage149_health_semantics.svg
stage149_logging_boundary.svg
stage149_policy_cases.svg
stage149_guard_check_status.svg
```

## Next Step

Stage150 completed the disabled-by-default, loopback-only FastAPI adapter and
validated it with in-process ASGI integration plus a real local HTTP/1.1 socket
smoke test. The result is recorded in
`docs/primeqa_hybrid_agent_http_transport_validation.md`.

Stage151 may now freeze the local service-entrypoint composition protocol. The
adapter is not yet a persistent service; remote deployment, runtime
defaultization, test evaluation, retries, and fallback remain closed.
