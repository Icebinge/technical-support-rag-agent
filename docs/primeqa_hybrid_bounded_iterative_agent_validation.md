# Stage 168 Bounded Iterative Agent Validation

## Objective

Stage 168 starts the Agent-capability branch after Stage 167 closed the unsafe
history-isolation gate. It implements and evaluates the user-authorized A+C
fallback contract:

- A inspects one alternate evidence view from the candidate pool already built
  for the request.
- C asks one system-owned clarification question for one of six fixed missing
  information categories.

This stage does not register the new runtime as default, integrate it into the
HTTP service, open development or test, rewrite the query, perform a second
retrieval, retry a model call, or enable any unapproved fallback.

## Runtime Contract

The initial router can select compose, inspect, clarify, or refuse. Inspection
exposes the original-RRF ranks 1-10 from the existing candidate pool and does
not call retrieval. After inspection, a second and final router decision can
select compose, clarify, or refuse. Inspection cannot be selected again.

The graph is acyclic and enforces these per-turn limits:

```text
retrieval calls:              exactly 1
alternate evidence checks:   at most 1
model decisions:              at most 2
query rewrites:               0
second retrievals:            0
retries:                      0
```

Clarification is a distinct `clarify` terminal state, not a grounded answer or
an insufficient-evidence refusal. The model selects only a category; system
code owns the final clarification text. The six categories are product or
component, version or build, error code or log, environment or platform,
requested outcome, and reproduction steps.

The existing v1 state ledger still defaults to `complete` and `refuse`. The v2
runtime explicitly authorizes `complete`, `clarify`, and `refuse`, preserving
backward compatibility.

## Train-Only Feasibility

The formal analysis replayed the exact frozen Stage 161 train retrieval
contract over 562 train rows and 112,400 Top200 candidate records. It authorized
all source files by SHA-256. Development and test were not accepted by the CLI
and were not loaded.

For the 370 answerable train rows, gold-document visibility was:

```text
candidate Top200:             345 / 370
initial query-overlap Top10:  175 / 370
alternate original-RRF Top10: 255 / 370
initial + alternate union:    267 / 370
```

The alternate view contributed 92 gold hits not present in the initial view;
the initial view contributed 12 not present in the alternate view. Both views
contained the gold document for 163 rows, while neither Top10 view contained it
for 103 rows. They shared a mean 2.715302 documents, producing a mean union
context size of 17.284698 and a maximum of 20.

Among the 138 Stage 165 post-first answerable cases where the gold document was
in the candidate pool but absent from generation context, the alternate view
made it visible in 68 cases, or 49.2754%. The other 70 remain unresolved by
this A strategy.

These are evidence-visibility measurements, not answer-quality metrics. No
real router model, answer generation, F1, citation scoring, or human
clarification-quality labeling ran in this analysis.

## Result And Boundary

All 16 process guards passed. The current-source report is
`artifacts/primeqa_hybrid_bounded_iterative_agent_stage168.json`, SHA-256:

```text
27dd3266414e9e2e766588095b0792be035b7e3e1610bc9355167b0243fcf80a
```

Four SVGs cover gold-document visibility, complementary view contribution,
known generation-miss rescue, and the bounded call budget. All parse as XML;
the coverage chart was also rendered through Edge headless and visually
checked for readable labels, values, and nonoverlapping bars.

The decision is `advance_to_stage169_real_gpu_router_calibration`. It means the
contract is train-feasible and implementation-safe enough for a real local
Qwen routing experiment. It does not authorize default activation, service
integration, development evaluation, or test evaluation.

## Process Corrections

Two exploratory reads failed before changing files because Windows does not
accept the passed wildcard form in `rg`; subsequent reads used `rg --files`
and directory-level searches. The first core lint found four import and line
format issues. The first validation lint found twelve format issues. Both sets
were corrected before formal analysis.

The first formal process exited at Typer argument parsing because
`Start-Process` split a confirmation note containing spaces. It did not load
data or models and produced no report. A replacement using an equivalent
space-free audit identifier completed successfully and produced report hash
`929b803ee9f7737bd9a41adfff1eb8981f00dabfaf78e9aea14228890ebc08de`.

During review, the validation report's runtime-budget fields were changed from
duplicated literals to the runtime's exported frozen contract. Because source
code changed, the earlier successful report was not presented as current. A
current-source replacement completed and produced the final hash above. Each
successful formal run was one PowerShell command tracking one process with
`Wait-Process` until natural exit, without status polling or a PowerShell wait
timeout.

## Repository Verification

```text
Stage 168 focused tests: 24 passed in 1.73s
Stage 168 eight-file Ruff format check: passed
full repository Ruff lint: passed
full repository pytest: 953 passed, 1 warning in 12.86s
full pytest exit code / stderr bytes: 0 / 0
```

The full pytest run was started once by one PowerShell command and tracked by
one `Wait-Process` call against the same PID until natural completion. The
warning is the existing FastAPI/Starlette `TestClient` deprecation warning.
