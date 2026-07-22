# Stage 171 Expanded Hierarchical Router Validation

## Objective

Stage 171 replaces the failed four-action prompt with two independent strict
classifiers. The request layer judges only request completeness and support
scope. The evidence layer assumes a complete technical request and judges only
whether the visible evidence directly supports an answer. Program code maps
the product of those classifications and the current phase to one authorized
Agent action.

The user selected the expanded option: 100 frozen train cases, 20 from each
existing evidence stratum. Per-call evidence and token limits remain at the
Stage 169/170 memory envelope. Development, test, answer generation, retries,
fallbacks, and default runtime activation remain closed.

## Frozen Architecture

The request schema allows exactly:

```text
complete_technical_request
missing_specific_fact + one authorized clarification kind
unsupported_request
```

The evidence schema allows exactly:

```text
sufficient_evidence
insufficient_evidence
```

Both layers run for every decision, even when the request disposition would
make the evidence output irrelevant. This fixes the model-call budget and
prevents branching behavior from changing the experiment:

```text
18 synthetic decisions x 2 layers =  36 calls
100 train cases x 2 phases x 2   = 400 calls
total                               436 calls
```

The deterministic action mapping is:

```text
unsupported request                     -> refuse
missing specific fact                   -> clarify
complete + sufficient evidence          -> compose
complete + insufficient, initial phase  -> inspect
complete + insufficient, final phase    -> refuse
```

Any invalid layer JSON invalidates the complete decision. There is no retry,
schema repair, action substitution, or fallback.

## Expanded Sampling

The same frozen 562-row train split and 112,400 candidate rows are replayed.
Within each of five evidence strata, the first 20 identities in ascending
SHA-256 order are selected. The available stratum populations are at least
175, 92, 78, 25, and 192, so no selected row is duplicated and the smallest
stratum retains five unselected rows.

The existing five fold IDs are retained for grouped stability reporting. This
stage has no trained parameters or selected threshold, so the fold table is a
grouped five-fold stability evaluation, not an OOF model-selection procedure.

## Quality Results

All 436 layer outputs passed their strict schemas. The structural design
improves the comparable Stage 169 quality gates from 3/8 to 5/8:

```text
metric                              observed    threshold    passed
synthetic phase action              72.2222%    >= 80.0000%  no
synthetic clarification kind         0.0000%    >= 83.3333%  no
initial-visible compose            100.0000%    >= 70.0000%  yes
alternate-only initial inspect      50.0000%    >= 50.0000%  yes
alternate-only final compose        90.0000%    >= 70.0000%  yes
alternate-only exact path           50.0000%    >= 40.0000%  yes
insufficient final compose          80.0000%    <= 20.0000%  no
layer schema valid                 100.0000%    = 100.0000%  yes
```

Four of five hierarchy-specific gates pass:

```text
synthetic request disposition       72.2222%    >= 90.0000%  no
synthetic evidence disposition     100.0000%    >= 90.0000%  yes
train request complete              98.0000%    >= 95.0000%  yes
request layer schema               100.0000%    = 100.0000%  yes
evidence layer schema              100.0000%    = 100.0000%  yes
```

The request layer is nearly stable on real technical train questions: it marks
196/200 phase decisions complete. However, it recognizes only one of six
synthetic missing-fact requests, then selects the wrong clarification kind.
The other five are classified as complete. Clarification-kind accuracy is
therefore 0/6.

The evidence layer is perfect on the ten synthetic phases with an evidence
expectation, but over-accepts topical real documents:

```text
stratum                          sufficient / 40 phase decisions
initial gold visible             40 / 40
alternate-only gold visible      29 / 40
union gold missing               26 / 40
candidate-pool gold missing      25 / 40
unanswerable                     29 / 40
```

This is an evidence-entailment failure. The classifier reliably recognizes
obvious synthetic evidence but treats product/topic overlap as direct answer
support on real technotes. The resulting insufficient-evidence final compose
rate is 80%, which is unsafe.

## Five-Fold Stability

Every fold passes both layer schemas, but every fold fails the safety rate:

```text
fold    alternate exact path    insufficient final compose
1        0.0000%                 76.9231%
2       60.0000%                 54.5455%
3       33.3333%                 85.7143%
4       50.0000%                 75.0000%
5       75.0000%                100.0000%
```

The failure is not isolated to one fold. The formal decision is:

```text
stage171_hierarchy_requires_redesign
```

The hierarchical router is not registered in the default runtime.

## Resource Consumption

PID `23392` completed all 436 generations without OOM and exited naturally:

```text
wall time:                         388.730042 s
train evidence build:              91.567470 s
model load:                         4.764013 s
synthetic calibration:             14.269941 s
expanded train validation:        274.773031 s
total generation time:            284.699760 s
generation throughput:              1.531438 calls/s
input/output tokens:           441914 / 5880
```

Per-generation input token counts range from 249 to 3,539, with p50 578 and
p95 2,412. Request-layer latency p50/p95/max is 438.320/523.339/987.808 ms.
Evidence-layer latency is 672.343/1,223.550/12,368.132 ms. The long evidence
outlier appeared only after expanding the sample coverage.

Memory peaks were:

```text
process working set:                6.833 GiB
process private usage:             15.275 GiB
minimum system available memory:    2.429 GiB
CUDA allocated:                     6.267 GiB
CUDA reserved:                      9.020 GiB
physical CUDA device memory:        7.960 GiB
```

On Windows WDDM, PyTorch's reserved counter can represent virtual allocator
reservations above physical VRAM; it is not live tensor memory. CUDA allocated
is the closer live-demand measure. Nevertheless, the 12.37-second latency
outlier and increased allocated peak show that the longer covered inputs are
nearer memory pressure. Future stages may increase sample count, but must not
increase context length, batch size, or concurrent generation on this GPU.

Resource capture is event-driven inside the formal process before and after
each generation. The process was launched once and the same PowerShell command
called `Wait-Process` once for PID `23392` until natural exit. There was no
polling, PowerShell wait timeout, kill, restart, or partial continuation.

## Visualizations

Six SVGs cover quality-gate progress, synthetic layer accuracy, expanded train
proxy rates, grouped five-fold stability, per-layer p95 latency, and process/GPU
resource peaks. All six parse as XML and rendered successfully with independent
Edge headless profiles. Titles, labels, values, and bar bounds were visually
checked. At conversation-image scale, two rendered titles appeared abbreviated;
direct SVG inspection confirmed the complete `Stage 171` title text. Edge
stderr contained only existing QQBrowser-profile notices.

## Process Corrections

The first read-only repository search used Windows wildcard paths directly in
`rg` and failed before reading files. A second filtered `rg --files` expression
also returned no matches because it assumed slash direction. Listing real paths
with PowerShell `Select-String` succeeded. No files were changed by either
failure.

The first static check found three 101-character prompt lines and one unused
import. They were corrected before tests or model execution. A combined patch
for that correction later failed because Ruff had already reformatted one
context block; the patch was atomic and changed no files. Smaller patches then
applied the same intended corrections.

The first focused pytest PID `4724` exited 2 during collection because a
parameter was named pytest's reserved `request`. No tests and no model calls
ran. Renaming it to `request_json` produced 50 passing focused tests on PID
`9324`.

The formal GPU run succeeded on its first attempt. During the final full suite,
the tool channel yielded while the same shell command and same direct
`Wait-Process` remained active; receiving the eventual result did not launch a
second shell command or poll the PID. PID `35152` exited naturally with 983
passing tests.

## Artifacts

The public report is
`artifacts/primeqa_hybrid_hierarchical_router_validation_stage171.json`,
SHA-256:

```text
cfb4dad9dd55587b058623c7d818f89ba5bcd8199bdf85f19c0d8df70d921e5d
```

All 15 process guards pass. The report contains no raw question, answer,
document text, model output, private sample ID, or gold document ID.

## Next Boundary

Stage 172 should replace the generative evidence layer with a dedicated binary
evidence-entailment candidate trained only on runtime-safe query/evidence
features. Candidate family and threshold selection must use grouped five-fold
OOF predictions over train, with fold-level safety non-regression. The frozen
Stage 171 request layer can remain an explicit experimental input, while the
clarification taxonomy is handled as a separate unresolved gate. Development,
test, answer E2E, and default activation remain closed.

## Repository Verification

```text
Stage 171 focused pytest: 50 passed in 1.65s
six-file Ruff format check: passed
full repository Ruff lint: passed
full repository pytest: 983 passed, 1 warning in 11.77s
full pytest PID / exit code: 35152 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation warning.
