# Stage 170 Frozen Iterative Router Prompt Comparison

## Objective

Stage 170 tests whether instruction design alone can repair the Stage 169
iterative router. It compares three frozen prompt profiles with the cached
`Qwen/Qwen3-VL-2B-Instruct` model on the RTX 5060 while retaining the compact
Stage 169 evidence representation and generation limits.

The stage uses synthetic cases for profile screening and the frozen train-only
calibration identities for the finalist comparison. It does not open
development or test, generate answers, evaluate F1 or citations, retry failed
generations, or activate a runtime default.

## Frozen Protocol

The three instruction profiles were fixed before model execution:

```text
ordered precedence
contrastive few-shot
phase gate
```

All profiles share the Stage 169 query-aware 200-character evidence windows,
final-phase document deduplication, 4,096 input-token limit, 32 output-token
limit, strict JSON parser, and greedy generation settings.

Each profile first receives the same 14 synthetic initial cases and four
synthetic final-phase calls. Synthetic quality alone ranks the profiles. The
top two then receive the same 50 frozen train cases, with one initial and one
counterfactual final call per case. The complete run therefore contains
exactly 254 model calls:

```text
3 profiles x 18 synthetic calls =  54
2 finalists x 50 x 2 train calls = 200
total                              254
```

The train cases retain their existing five-fold assignments so cross-fold
stability can be reported. The folds do not select or tune prompts after train
outcomes are observed.

## Synthetic Screen

The synthetic ranking was:

```text
profile                      action       clarification   exact path
contrastive few-shot         66.6667%     33.3333%        42.8571%
phase gate                   61.1111%     16.6667%        35.7143%
ordered precedence           27.7778%      0.0000%        14.2857%
```

Contrastive few-shot and phase gate advanced to the frozen train comparison.
No profile met the synthetic action or clarification threshold.

## Train-Only Results

The Stage 169 baseline passed 3/8 frozen quality gates. Both Stage 170
finalists passed 0/8:

```text
metric                              contrastive   phase gate   threshold
synthetic phase action                66.6667%     61.1111%    >= 80.0000%
synthetic clarification kind          33.3333%     16.6667%    >= 83.3333%
initial-visible compose                20.0000%     60.0000%    >= 70.0000%
alternate-only initial inspect         30.0000%      0.0000%    >= 50.0000%
alternate-only final compose           10.0000%     50.0000%    >= 70.0000%
alternate-only exact path               0.0000%      0.0000%    >= 40.0000%
insufficient final compose             33.3333%     36.6667%    <= 20.0000%
schema valid                           81.3559%     95.7627%    = 100.0000%
```

Contrastive few-shot won the synthetic ranking, but its train schema-valid
rate fell to 81.3559% and its initial-visible compose rate fell to 20%. Phase
gate preserved more schema validity and reached 60% initial-visible compose,
but never chose alternate inspection on any alternate-only case. Neither
profile completed a single alternate-only inspect-to-compose path.

The five-fold view also rejects a stable prompt-only improvement. Contrastive
schema validity ranges from 72.2222% to 83.3333%; phase-gate initial-visible
compose ranges from 0% to 100%, and its unsafe insufficient-evidence final
compose rate reaches 60% in fold 4. The observed behavior changes sharply by
case composition instead of enforcing the desired phase contract.

The formal decision is:

```text
stage170_prompt_family_insufficient
```

No Stage 170 prompt is activated in the default runtime.

## Resource Consumption

The complete comparison finished without CUDA OOM:

```text
wall time:                         343.770696 s
train evidence build:              93.895376 s
model load:                         5.322201 s
synthetic screen:                  31.343788 s
train finalist comparison:        210.076469 s
total GPU generation time:        237.859314 s
generation throughput:              1.067858 calls/s
total input/output tokens:     425321 / 4204
```

Resource peaks were:

```text
process working set:                7.323 GiB
process private usage:             13.273 GiB
minimum system available memory:    2.718 GiB
CUDA allocated:                     5.420 GiB
CUDA reserved:                      7.020 GiB
```

The event-driven sampler runs in the experiment process at phase boundaries
and after generation calls. There is no polling monitor. The formal PID `6552`
was started once and waited on by one direct PowerShell `Wait-Process` until it
exited naturally with code 0.

## Visualizations

Five SVGs report synthetic profile accuracy, frozen quality-gate passes,
train proxy rates, p95 latency, and process/GPU resource peaks. All five were
rendered with Edge headless. The key charts were visually checked for labels,
values, bar bounds, and overlap; all source SVGs contain complete titles and
parse as XML.

The first visual review appeared to show a truncated resource title at the
conversation image scale. Direct inspection of the SVG established that its
title and visible text were already complete. A regression assertion now
checks the exact resource chart title. This was a display-review false alarm,
not a corrected experimental value.

## Process Corrections

The first Stage 170 static check found one import-order violation in the
extended router test. It was corrected before any model run; the focused suite
then passed 37 tests.

A report-summary PowerShell command used `ConvertFrom-Json -Depth`, which this
Windows PowerShell version does not support. Removing that read-only option
successfully extracted the same report. Another read-only `rg` command used a
double-quoted pattern containing `$0`; PowerShell expanded it and produced an
invalid regex. Reissuing the command with a single-quoted pattern succeeded.
Neither failure changed code, reports, or model output.

A repository-wide `ruff format --check` found 311 historical files that the
current Ruff version would reformat. Those unrelated files were not changed.
The six Stage 170 Python files were formatted and linted independently. Edge
wrote all five PNG inspection renders successfully; stderr contained existing
QQBrowser-profile messages and one transient disk-cache directory message,
with exit code 0.

## Artifacts

The public report is
`artifacts/primeqa_hybrid_iterative_router_prompt_comparison_stage170.json`,
SHA-256:

```text
d74abda6a8455ab1946504096654a1238c5aa5ad2b4d5d3f4aa917e6badb5ef2
```

The report contains no raw question, answer, document text, or generated model
text. All 14 process guards passed, including exact call count, local-only
model loading, no development/test access, no retries, and unchanged runtime
default.

## Next Boundary

Prompt wording is now empirically exhausted for this router contract. Stage
171 should design a constrained two-level router: deterministic or separately
calibrated phase/action eligibility first, followed by a smaller model choice
only among actions legal for that state. It should retain train-only selection,
five-fold stability reporting, the compact evidence representation, and the
same quality thresholds. Development, test, answer E2E, and default runtime
activation remain closed until that structure passes the frozen gates.

## Repository Verification

```text
Stage 170 focused pytest: 37 passed in 1.42s
six-file Ruff format and lint: passed
full repository pytest: 970 passed, 1 warning in 11.75s
full pytest PID / exit code: 19900 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation warning.
The full suite used one PowerShell-launched PID and one direct `Wait-Process`
until natural completion, with no polling, wait timeout, kill, retry, or
fallback.
