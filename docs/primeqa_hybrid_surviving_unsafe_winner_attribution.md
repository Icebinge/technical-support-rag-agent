# Stage 197 Surviving Unsafe-Winner Attribution

## Objective

Stage 197 determines why unsafe actions still win after the Stage 196 cap-16
safety pool and safest-prefix filter. It uses only train-side inner-OOF
contexts. Development and test data remain closed.

The experiment reconstructs only the published first-ranked top-inner spec
from each of the five Stage 196 outer contexts. It does not search a new
configuration, fit an outer model, select a full-train policy, run runtime E2E,
or introduce a fallback.

## Focused Reconstruction

Each of the 20 inner partitions fits exactly the four models needed by its
published spec: two safety heads, one LambdaMART gain ranker, and one unsafe
classifier. This reduces reconstruction from Stage 196's 480 actual fits to
80 focused fits while preserving the exact top-inner predictions and metrics.

All five reconstructed evaluations and diagnostic reports exactly match the
published Stage 196 top-inner evidence within the frozen `1e-6` numeric
tolerance.

```text
inner partitions:               20
model fits:                     80
pool safety / LambdaMART:       40 / 20
unsafe-head fits:               20
LightGBM trees:             12,000
private predictions:       196,768
top-inner reconstructions:       5 / 5 exact
process guards:                 29 / 29 passed
```

## Exact Attribution

The 1,480 inner-OOF question contexts contain 465 unsafe winners, an aggregate
rate of `0.314189`. Every unsafe winner also has a strict opportunity. Their
mutually exclusive mechanism partition is exact:

| Mechanism | Count | Meaning |
| --- | ---: | --- |
| final gain dominance | 181 | A lower-risk strict action is already in the frontier, but pure gain selects unsafe |
| risk ordering failure | 172 | A strict action is in the frontier, but the unsafe winner is ranked at least as safe |
| risk frontier exclusion | 97 | Strict exists in the complete pool but is removed by risk ordering |
| safety pool exclusion | 15 | Strict exists globally but is absent from the cap-16 safety pool |

All 465 unsafe winners are gain rank 1 inside their frontier. Their complete-pool
risk ranks are `145` at rank 2, `195` at ranks 3-4, `105` at ranks 5-8, and
`20` at rank 9 or later; none is rank 1 because the baseline is safer in these
contexts.

Of the unsafe winners, `431` are F1-only regressions, `17` are citation-only,
and `17` lose both citation and F1. A gold-labelled offline oracle finds a
strict frontier alternative for `353/465` (`0.759140`) and a lower-risk strict
alternative for `181/465` (`0.389247`). This oracle is an attribution upper
bound only and is not a deployable rule or runtime candidate.

The unsafe head remains weak: ROC AUC is `0.589114` and average precision is
`0.562431` across 49,192 action contexts with unsafe prevalence `0.478289`.
`final_gain_dominance` is the largest exact mechanism, but it exceeds
`risk_ordering_failure` by only nine contexts. The evidence therefore supports
a risk-aware final winner rule as the first focus while retaining unsafe-head
discrimination as a comparably important factor in the next frozen design.

## Formal Execution

The formal process used Python PID `16432`. One PowerShell command called
`Wait-Process` once for that PID and waited for natural completion, with no
polling or experiment timeout. PowerShell's child `ExitCode` field was empty
after the wait and is retained as unknown; the outer command completed, the
formal report was written, all guards passed, and stderr contained only model
weight-loading progress bars.

```text
Stage 182 reproduction:     202.664476 s
Stage 197 attribution:      203.914902 s
total wall:                 407.441980 s

peak working set:             3.748 GiB
peak private usage:           4.083 GiB
minimum system available:     3.078 GiB
CUDA allocated/reserved:      0 / 0
OOM:                          none
```

## Visual Verification

Ten SVG charts passed XML parsing and were rasterized with pinned
`resvg_py==0.3.3`, project-owned Poppins fonts, a white background, and no font
fallback. All ten PNGs were opened at original resolution. Titles, labels,
zero-value rows, counts, and all 29 guard names are visible without clipping,
overlap, or blank output.

```text
formal report SHA-256:
c56f4af1b408a07e295a10f7decd2c8a0313f814f16955fd149a170355646d9d

resvg manifest SHA-256:
308541fb7615165c5aef02bce23ecffe522207137ee38597fa05436d5fc31c7f
```

## Decision

The formal status is
`stage197_surviving_unsafe_winner_attribution_complete`. The diagnostic is
valid, but it does not authorize development/test access, full-train policy
selection, runtime E2E, replacement, or default activation.

The next stage should freeze a train-only factorial protocol that separates
two effects: risk-aware final winner rules and improved unsafe-head
discrimination. The primary comparison must test whether adding risk to the
final decision reduces the 181 gain-dominance failures without sacrificing
strict capture, while a crossed risk-head factor measures whether the 172
ordering failures and 97 frontier exclusions can also be reduced.

## Current-Source Verification

Repository-wide Ruff lint, seven changed Python-file format checks,
`pip check`, CLI help, and `git diff --check` passed. The focused Stage 195-197
regression set passed `23 tests in 9.67s`.

The complete pytest suite used the single Python PID `27212`. Its PowerShell
command called `Wait-Process` exactly once and waited for natural completion,
without polling or a pytest timeout. The result was `1180 passed, 1 warning in
39.38s`; stderr was empty. The warning is the existing FastAPI/Starlette
`TestClient` deprecation. PowerShell's post-wait child `ExitCode` field was
empty and is retained as unknown rather than being fabricated as zero.
