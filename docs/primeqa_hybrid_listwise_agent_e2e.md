# Stage 178 Listwise Agent End-to-End Evaluation

## Objective

Stage 178 tests whether the Stage 177 listwise second-stage reranker improves
the actual Agent retrieval, context, composition, and verification path. The
work follows the user-selected route C:

```text
Stage 178A: deterministic LangGraph tool-Agent OOF E2E
Stage 178B: full sharded Qwen dynamic-Agent E2E, only if every 178A gate passes
```

Both phases remain train-only. Development and test data stay closed. The
failed sufficiency gate remains disabled, no fallback is introduced, and the
default runtime is unchanged.

## Frozen Stage 178A Protocol

Stage 178A reconstructs the exact 562-row train candidate replay and 9,714
complete Stage 177 pairs. It trains five grouped out-of-fold `listwise_none`
models. Only held-out OOF logits may contribute to answer-quality metrics.

A sixth model is trained on all train pairs and exported as an authenticated
optional runtime checkpoint. That checkpoint is used only for a 50-query CPU
latency probe and one label-free runtime smoke. It is never used to score
train answer quality.

The paired real LangGraph tool-Agent comparison runs 562 questions through
each arm:

```text
baseline:  query-overlap Top10 from the immutable first-200 prefix
candidate: query-overlap Top10 union original-RRF Top10, listwise rerank to Top10
```

The live candidate retriever must reproduce the frozen first-200 candidate
sequence exactly for every question. Arm order is balanced by a stable hash.
The workflow topology is retrieval, context preparation, deterministic answer
composition, and answer verification. Two thousand paired bootstrap samples
measure uncertainty. The full-train checkpoint is loaded on CPU because a
future Qwen runtime must retain GPU memory for generation.

The predeclared Stage 178A gates are:

```text
verified F1 delta:                         > 0
verified F1 paired CI lower:              >= 0
gold citation count delta:                >= 0
gold citation paired CI lower:            >= 0
context gold-hit count delta:              > 0
context gold-hit paired CI lower:          > 0
answerable refusal delta:                 <= 0
unanswerable false-answer delta:          <= 0
F1 fold non-regression:                   >= 4 / 5 folds
citation fold non-regression:             >= 4 / 5 folds
full checkpoint CPU rerank p95:           <= 1.0 second
```

Every quality gate and every process guard must pass before Stage 178B is
authorized. A failure stops route C at Stage 178A and is reported as measured;
there is no alternate policy or fallback path.

## Frozen Stage 178B Boundary

If authorized, Stage 178B will compare the same paired policies across all
562 train questions with the full Qwen dynamic Agent. It will use 12 fresh,
strictly sequential GPU processes, 96 model turns per shard except the final
shard, for 1,124 measured model turns in total. OOF listwise scores will be
precomputed by Stage 178A, so MiniLM and Qwen will not occupy the GPU together.

The parent process must launch each shard sequentially and wait for natural
completion. Formal long-running commands use one PowerShell `Wait-Process`
against the launched PID, with no polling loop, segmented monitor, or
user-imposed process timeout.

## Preflight Record

The initial implementation added a replaceable primary-context selection
interface while preserving the current query-overlap Top10 default. It also
added authenticated checkpoint export, OOF and CPU score providers, the
listwise union policy, a Stage 178A runner, CLI, metrics, gates, resource
accounting, and SVG outputs.

The first new-test run had one incorrect assertion: it required `doc-20` to
survive the final Top10 even though the synthetic score intentionally ranked
it below ten other members. The candidate had entered and been scored in the
20-document union. The assertion was corrected to verify that no document
outside the union can enter and that exactly 20 pairs were scored.

A related-regression command initially named a nonexistent sidecar test file,
so pytest stopped before collection. The real filenames were enumerated and
the regression was rerun. A later report audit found retry and fallback counts
were initialized as constants rather than aggregated from traces; they were
changed to measured values before formal execution.

Current preflight verification:

```text
Stage 178A focused tests:                     6 passed
related Agent/ranking/sidecar regression:    50 passed, 1 existing warning
Ruff format and lint:                        passed
Python syntax compilation:                   passed
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation.
Formal results, visual verification, resource measurements, final repository
verification, and the resulting Stage 178 decision will be appended only
after they have actually occurred.

## Formal Attempt 1 And Alignment Audit

Formal attempt 1 ran as PID `16368` and naturally ended after the parent
PowerShell command made one `Wait-Process` call. Five OOF fits and the sixth
full-train fit completed, and the authenticated checkpoint was written. The
run then failed before its first paired Agent trace because the live Stage 128
candidate prefix did not exactly match the Stage 161 replay. No Stage 178A
quality report or private OOF score artifact was written, so this attempt is
not a completed evaluation. It did not OOM.

The enclosing PowerShell command returned `0`, but the retained Windows
`Start-Process` object exposed a blank child `ExitCode` after `Wait-Process`.
The missing report and the traceback, rather than that enclosing status, are
the authoritative failure evidence.

A dedicated train-only aggregate alignment audit then ran as PID `35116`,
again with one `Wait-Process`. It performed no model fitting, loaded no dev or
test rows, and wrote no raw question or document identifiers. All seven audit
process guards passed. Results across 562 train questions were:

```text
full Top200 sequence exact:                 387 / 562
full Top200 set exact:                      466 / 562
original-RRF Top10 sequence exact:          561 / 562
query-overlap Top10 sequence exact:         560 / 562
two-view union sequence/set exact:          559 / 562
live union pairs absent from offline union:   3 pairs / 3 questions
prefix symmetric difference mean/p95/max:  0.608541 / 4 / 16
```

Gold-hit aggregates happen to be unchanged despite the identity drift:

```text
view                   offline  live  gains  losses
original RRF Top10          255   255      0       0
query-overlap Top10         175   175      0       0
two-view union              267   267      0       0
```

The root cause is a depth mismatch in the exact-token derived route. Stage 161
reconstructs the Stage 116 prefix with `component_depth=200`. The current
Stage 128 online runtime resolves channels at depth 400 before taking its
prefix, allowing specially boosted documents from base ranks 201-400 to enter
the first 200. Stage 140 proved the runtime matched its own Stage 128 legacy
reconstruction, but that reconstruction used the same 400-depth behavior and
therefore did not prove identity with the original Stage 116 prefix.

The audit decision is deliberately non-authorizing:

```text
full_prefix_contract_exact:                     false
selection_surface_exact:                        false
live_union_fully_covered_by_stage177_pairs:     false
stage178_protocol_change_authorized:            false
```

Stage 178A remains incomplete, Stage 178B is not authorized, and dev/test
remain closed pending an explicit correction-route decision.

## Selected Correction Route A

The user selected route A: restore the live runtime to the frozen Stage 116
Top200 contract instead of redefining training around the drifting runtime.
The online candidate retriever now resolves independent routes once at depth
400 and takes their stable Top200 prefixes. A derived route is evaluated twice
from already-resolved data: once against its source Top200 for the immutable
prefix and once against its source Top400 for append generation. This adds no
second BM25 or dense index search.

The pre-fix checkpoint, logs, and audit were retained in explicitly named
attempt-1 artifact directories. After the correction, a second train-only
alignment audit ran as PID `14428` with one `Wait-Process`. All seven process
guards passed, and every identity boundary is now exact:

```text
full Top200 sequence exact:                 562 / 562
full Top200 set exact:                      562 / 562
original-RRF Top10 sequence exact:          562 / 562
query-overlap Top10 sequence exact:         562 / 562
two-view union sequence/set exact:          562 / 562
live union pairs absent from offline union:   0
prefix symmetric difference mean/p95/max:    0 / 0 / 0
```

Live retrieval mean/p95/max was `0.066837/0.104169/0.677110` seconds. The
corrected audit report SHA-256 is
`e2398024edf128ad0628900d25eb1ccc9c83c437fb474921fe136e2603e47272`.
Stage 178A now authenticates this report as a mandatory source before fitting
or evaluating any model. Development and test remained closed.

## Stage 178A Formal Result

Formal attempt 2 ran as PID `10028` and naturally completed after one
`Wait-Process` call. It produced all public/private reports, the authenticated
checkpoint, and eight SVG charts. All 20 process guards passed. The Windows
`Start-Process` child `ExitCode` field was again blank after waiting, so no
child exit-code value is inferred; complete artifacts and guards establish
the completed run.

The paired 562-question deterministic tool-Agent result was:

```text
metric                         baseline   candidate   delta
verified average token F1      0.1946     0.2004     +0.0058
context gold-hit count             175        257         +82
gold citation count                151        183         +32
answerable refusal count             1          1           0
unanswerable false-answer count    191        191           0
request latency p95 seconds    0.129680   0.128900   -0.000780
```

The context and citation improvements are strong. Their paired bootstrap 95%
CI lower bounds are `0.178378` and `0.053986`. The F1 improvement is small and
not statistically stable: observed answerable delta is `0.005804`, with 95%
CI `[-0.001612, 0.013721]`.

Candidate F1 regressed in folds 1 and 2 and improved in folds 3, 4, and 5:

```text
fold      baseline   candidate   delta
fold_1    0.1896     0.1814     -0.0082
fold_2    0.1939     0.1870     -0.0069
fold_3    0.1951     0.2129     +0.0178
fold_4    0.1803     0.1886     +0.0083
fold_5    0.2172     0.2370     +0.0198
```

Nine of eleven frozen quality gates passed. The two failures were:

```text
f1_bootstrap_ci_lower_nonnegative:   false (-0.001612)
f1_fold_nonregression_4_of_5:        false (3/5)
```

Citation fold non-regression passed 5/5. The full checkpoint CPU probe covered
50 deterministic questions with mean/median/p95/max latency of
`0.574648/0.587466/0.665752/0.710443` seconds, passing the frozen one-second
p95 limit. The label-free real runtime smoke completed in `0.617925` seconds
with candidate depth 400, context depth 10, three tool calls, and zero retry or
fallback actions.

## Stage 178A Resources

```text
wall time:                         458.378767 seconds
candidate replay:                   93.757556 seconds
pair build and frozen scoring:      26.176787 seconds
five-fold OOF training:            148.533170 seconds
full checkpoint training:           31.416042 seconds
runtime resource build:             32.898284 seconds
paired tool-Agent E2E:              95.691908 seconds
CPU checkpoint validation:          29.648752 seconds
process working-set peak:             7.509 GiB
process private-usage peak:           11.712 GiB
minimum system available memory:       0.289 GiB
CUDA allocated/reserved peak:      2.954 / 5.541 GiB
```

The run completed without OOM, but the low minimum system-memory margin is
recorded rather than described as abundant. The five OOF fits and one full fit
used 9,714 complete scoring pairs; the full fit sampled 1,925 training pairs,
ran 128 optimizer steps, and reduced mean loss from `1.324835` to `1.002887`.
The full checkpoint was not used for quality metrics.

## Visual Verification

All 8/8 SVG files parse as XML. The F1, context/citation, fold F1, and quality
gate charts were independently rendered to PNG and visually inspected. They
are nonblank, fit their canvases, and agree with the public report. The first
headless Edge command used relative screenshot outputs; Edge could not resolve
those paths and wrote no PNGs. The commands were rerun with absolute output
paths and succeeded. Edge also emitted its existing QQBrowser profile warning.

## Stage 178 Decision

The formal status is `stage178a_listwise_tool_agent_e2e_insufficient`, with
`candidate_selected=false` and `stage178b_authorized=false`. Route C therefore
stops before the sharded Qwen dynamic-Agent experiment. The checkpoint remains
an ignored, authenticated research artifact but is not registered or enabled
as the default runtime. Development and test were not opened.

The current public report SHA-256 is
`e57e3f09bcc65657a3f8783e97e6767b690095e2cffd5d252d51e181eaf533c9`.

Final current-source repository verification:

```text
Stage 178 affected regression: 77 passed, 1 warning in 5.46s
Stage 178 file Ruff format check: passed
full repository Ruff lint: passed
full repository pytest: 1046 passed, 1 warning in 19.82s
full pytest PID / enclosing PowerShell exit: 1884 / 0
full pytest stderr bytes: 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. The
full suite ran in one pytest process and the same PowerShell command made one
`Wait-Process` call until natural completion. Its pytest output proves success
and the enclosing command returned `0`; the retained child `ExitCode` field
was blank, so no child exit-code value is claimed.
