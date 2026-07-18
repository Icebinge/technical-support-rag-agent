# Stage 159 Warm Multi-Turn Agent Service Validation

## Objective

Stage 159 validates the Stage 158 bounded dynamic Agent service as one warm,
nondefault, loopback-only process across the entire frozen development split.
It measures four-turn process-local state growth, HTTP and model latency, token
growth, branch behavior, and one real two-request capacity rejection.

This is an execution and operating-behavior validation. It is not answer-quality
training or evaluation. The test split remains locked, development labels are
not used for selection or metrics, and the runtime remains nondefault.

## Frozen Development Protocol

The exact Stage 68 development file is authorized by all of the following:

```text
file: primeqa_hybrid_split_stage68_dev.jsonl
SHA-256: 071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f
rows: 121
assigned split: dev
split name: primeqa_hybrid_stage68_v1
protocol version: primeqa_hybrid_split_v1
```

Rows are ordered by SHA-256 of the private `sample_id`, then divided into
consecutive groups of at most four. This creates 30 four-turn threads and one
one-turn thread. The grouping is deterministic and synthetic: adjacent turns
are not asserted to form a natural conversation. Its grouping SHA-256 is
`7aa271a775c2926b32226e0a4fccc96cff3a7bf98fc90246c8002d79561fd6d0`.

The standard JSON parser necessarily materializes each authorized dev object.
Only `question_title` and `question_text` are projected into runtime requests;
`sample_id` is hashed for private ordering. Label fields are not used for
selection, projected into runtime, or used for metrics. The public artifact
contains no individual row, private identity, question, answer, document,
citation identity, or raw model output.

## Runtime Protocol

The validator authorizes the exact Stage 158 artifact SHA-256
`12649c087c3140feeb4121837152b41ef4005922eb73931f3770a5fac83889b0`,
all `51/51` Stage 158 guards, its decision status, and eight Stage 158 source
fingerprints before resource construction. It also fingerprints the Stage 159
protocol, validator, CLI, and `pyproject.toml` before and after execution.

One service preparation performs exactly one CPU retrieval-resource build and
one local Qwen load. One temporary warmup turn runs before listening. The 121
development turns then use the same resources and model. A final admitted
capacity-probe request runs real retrieval and Qwen inference. Expected and
observed model generations are therefore:

```text
startup warmup: 1
development turns: 121
capacity first request: 1
total: 123
```

Each development thread is explicitly opened, receives one to four sequential
HTTP turns, and is explicitly closed. There is no implicit thread creation,
persistence, queue, retry, fallback, query rewrite, or second retrieval.

The real capacity probe uses a validation-only observation gate. The gate
pauses the first request after coordinator admission but before its real
runtime execution. A second real HTTP request is sent while the slot is held;
after its response, the gate releases the first request to complete normal
retrieval and Qwen inference. Event waits, HTTP connections, queues, thread
joins, and service shutdown use no timeout.

## Formal Result

The formal process used port `18159`, completed naturally, returned exit code
zero, joined every process, and released the listener. All `65/65` guards pass.

```text
development HTTP turns: 121 / 121 returned 200
threads opened / closed: 31 / 31
branch-protocol-valid turns: 121 / 121
monotonic state-growth threads: 31 / 31
retrieval / model decisions: 121 / 121
resource factory builds: 1
model generations: 123
maximum in-flight turns: 1
failed turns: 0
queue / retry / fallback actions: 0 / 0 / 0
opened threads after shutdown: 0
port 18159 listener after shutdown: 0
SVG: 10 / 10 XML-parseable
```

The development action distribution is:

```text
compose_grounded_answer: 34 (28.0992%)
refuse_insufficient_evidence: 87 (71.9008%)
complete / refuse terminals: 34 / 87
citations emitted: 102
composition / verification / diagnostics: 34 / 34 / 34
```

These are runtime branch counts, not quality metrics. No development gold label
is used, so the answer rate is not routing accuracy, the refusal rate is not a
false-refusal rate, and the citation count is not citation precision or recall.

Aggregate warm-turn measurements are:

```text
end-to-end latency median / p95 / max: 1,977.732 / 11,835.247 / 16,624.877 ms
end-to-end latency average: 3,675.600 ms
generation latency median / p95 / max: 1,742.688 / 11,501.323 / 16,454.567 ms
generation latency average: 3,418.816 ms
router input tokens median / p95 / max: 2,608 / 3,194 / 4,009
retained state bytes median / p95 / max: 1,300 / 2,591 / 3,045
```

Per-position averages expose the multi-turn trend:

| Position | Turns | E2E avg ms | E2E median ms | E2E p95 ms | Generation avg ms | Input tokens avg | State bytes avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31 | 1,971.820 | 1,916.105 | 2,747.249 | 1,747.229 | 2,444.613 | 652.677 |
| 2 | 30 | 2,285.573 | 1,870.004 | 2,818.578 | 2,075.795 | 2,574.733 | 1,105.533 |
| 3 | 30 | 4,232.412 | 2,269.792 | 11,835.247 | 3,888.975 | 2,754.967 | 1,636.200 |
| 4 | 30 | 6,269.388 | 2,455.159 | 16,481.358 | 6,018.985 | 2,935.167 | 2,119.367 |

Retained state and average input tokens grow monotonically by turn position.
Median latency grows moderately after turn two, while third- and fourth-turn
averages and p95 values show a substantial generation long tail. Because the
grouping is synthetic and there is no per-turn public row, this stage does not
attribute that tail to a semantic query class or individual example.

## Real Capacity Result

Both capacity-probe threads opened with HTTP `201`. While the first request
held the only whole-turn admission slot, the second real request returned:

```text
HTTP status: 503
error code: gpu_capacity_exceeded
rejection latency: 1.401 ms
downstream dispatched: false
GPU admitted: false
```

The first request then completed real retrieval and Qwen execution:

```text
HTTP status: 200
end-to-end latency: 1,962.984 ms
terminal: complete
completed turns: 1
```

Both threads closed with HTTP `200`. The final coordinator counters are 122
admitted and completed turns, one capacity rejection, zero failed turns, zero
opened threads, and zero queue, retry, or fallback actions. This replaces the
Stage 158 synthetic-overlap limitation with a deterministic two-request real
HTTP observation while keeping model execution serialized.

## Startup And Timing

The verified environment is PyTorch `2.11.0+cu128`, Transformers `5.13.1`,
FastAPI `0.139.2`, Uvicorn `0.51.0`, CUDA 12.8, and an NVIDIA GeForce RTX 5060
with capability `(12, 0)`. The dense retrieval encoders remain on CPU.

```text
source authorization: 3.579429 s
retrieval resource build: 53.747254 s
model load: 7.791960 s
warmup: 3.316405 s
app composition: 0.006007 s
total service prepare: 68.441055 s
server start: 0.058862 s
full development workload: 445.539670 s
capacity probe: 2.116151 s
shutdown: 0.169886 s
final audit: 0.023740 s
formal total: 516.365146 s
peak allocated GPU memory: 7,344,342,016 bytes
```

The formal stderr file is not empty. It contains a PowerShell
`NativeCommandError` presentation wrapper followed by three successful model
weight-loading progress blocks. The Python process completed with exit code
zero, all guards passed, and no runtime traceback was present. This is recorded
as wrapper formatting, not described as empty stderr or hidden as a failure.

## Execution Corrections

An initial read attempted a project-root `AGENTS.md` that does not exist. The
applicable parent `AGENTS.md` was then read successfully; no file changed. A
separate combined read-only `nvidia-smi` pipeline printed device information
but returned shell exit code one because of pipeline state. It did not affect
the formal process or provide a formal validation result.

The formal run itself was started once and was not restarted. It used a hidden
detached process, no runtime timeout, no monitor deadline, no sleep-based
monitor, no kill, and no retry. Read-only polling observed progress through all
31 thread groups. It naturally wrote the artifact, shut down, and released the
port.

After formal completion, two attempts to launch a nested PowerShell full-test
wrapper were rejected by the desktop execution policy before execution. They
started no pytest process and produced no test result. A direct Python test
driver was then launched, but `pytest.main(())` passed a tuple where pytest 9
requires a list. PID `9648` exited with `TypeError` before collection, wrote no
exit file, and cannot be counted as a test run. No automatic replacement was
started.

After the user explicitly selected option A, the driver argument was corrected
to `pytest.main([])`. Exactly one replacement process then ran naturally and
completed successfully. These launch failures and the approved replacement do
not alter the Stage 159 product-source fingerprints or formal artifact.

## Current-Source Verification

```text
Stage157-159 targeted pytest: 52 passed, 1 existing warning in 1.84 s
full repository pytest: 797 passed, 1 existing warning in 11.52 s
full pytest exit code / stderr: 0 / empty
full repository Ruff: passed
Stage159 new-file Ruff format check: 5 files already formatted
```

The warning is the existing FastAPI `TestClient` Starlette deprecation warning.

The formal public artifact is
`artifacts/primeqa_hybrid_bounded_dynamic_agent_warm_service_stage159.json`
with SHA-256
`93eb319aeb0c2212f55df0bbb2c2b1790eeba02aa4ec20439464bc72a7f3bfe6`.

## Next Boundary

Stage 160 should remain development-only and choose explicitly between two
directions: analyze the 87 development refusals and the turn-three/four latency
tail with private, non-published diagnostics, or freeze this Agent runtime
behavior and move to the next integration boundary. Test, default runtime
registration, remote exposure, persistence, queueing, retry, fallback, query
rewrite, and second retrieval remain closed until separately authorized.
