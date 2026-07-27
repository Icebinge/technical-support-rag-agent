# Stage 189 Gain-Sensitive Ranking Failure Attribution

## Objective

Stage 189 diagnoses why every Stage 188 outer context had zero inner-eligible
configuration. It reproduces the frozen Stage 188 train-only nested CV and
streams one public-safe aggregate snapshot per outer context. It does not fit
an additional model or persist action-level predictions.

For every strict-opportunity question context and each of the 32 Stage 188
configurations, the diagnostic assigns exactly one outcome:

```text
safety frontier exclusion
strict action retained but missed by the gain ranker
strict action selected
```

The three counts must partition all strict opportunities exactly. Development,
test, runtime E2E, full-train policy selection, fallback behavior, and default
runtime activation remain closed.

## Implementation

The Stage 188 selector now exposes immutable per-question decisions through an
optional diagnostic sink. The production selection path and the diagnostic use
the same decision builder, preventing a second implementation from drifting
away from the evaluated algorithm. Stage 189 consumes one outer snapshot at a
time and retains only aggregates by outer context, safety margin, safety
estimator, gain ranker, feature representation, and full configuration.

The formal run reproduced all 12 Stage 188 checks, 240 model fits, and 393,536
private predictions. It added zero attribution fits and wrote zero private
action or prediction rows.

## Formal Result

The top-ineligible trajectory covered 1,480 question contexts. Of 1,456 strict
opportunities, 1,280 were removed by the safety frontier, 27 survived the
frontier but were missed by the ranker, and 149 were selected:

```text
1456 = 1280 + 27 + 149
frontier strict-question recall:       0.120879
conditional ranker strict capture:     0.846591
actual strict-opportunity capture:     0.102335
strict-action retention rate:          0.030655
mean frontier size:                    1.725676
filter harm / rescue contexts:         688 / 14
```

The same pattern held across all five outer contexts. Frontier exclusions were
253, 256, 260, 257, and 254, while retained-opportunity ranker misses were only
4, 4, 6, 1, and 12. The primary bottleneck is therefore
`safety_frontier_exclusion`, not gain-ranker discrimination.

Increasing the relative-risk margin improved frontier strict-question recall
from `0.014423` at margin `0.00` to only `0.079499` at margin `0.10`. Pairwise
Pareto logistic had higher conditional capture than ListNet
(`0.754116` versus `0.509330`). The best family configuration reached frontier
recall `0.135989`, still far below a useful candidate-pool recall level.

Stage 189 consequently refused to freeze the predeclared ranker-focused branch.
It did not authorize Stage 190 training, policy replacement, runtime E2E,
development, test, or default activation.

## Terminology Correction

After the formal process completed, one public aggregate name was found to be
ambiguous: `strict_selection_precision` measured strict selections among
changes relative to each question's original baseline, whereas Stage 188's
frozen precision gate is relative to the Stage 182 reference. The field was
renamed to `baseline_change_strict_precision`, and its denominator to
`baseline_changed_context_count`.

This was a structural migration of the persisted aggregate report. It did not
recompute metrics, rerun a model, reload predictions, or change the decision.
The initial report SHA-256 was:

```text
2ef733d9e36fca2ee3f2d79a1521a405c343d6039b33ce0dbc08d73dd9a1a0d5
```

The corrected formal report SHA-256 is:

```text
48af548168e4e40972c4082fc24bec822ce264427f12c56b98a8d0966df2e5a0
```

The report contains a `post_run_label_correction` provenance object preserving
the initial hash, reason, and unchanged-computation flags.

## Execution And Resources

Formal PID `25024` was awaited to natural completion by one PowerShell
`Wait-Process` call. There was no polling, experiment timeout, retry, partial
continuation, fallback, or OOM.

```text
wall time:                         871.946186 s
process CPU time:                 1889.656250 s
peak working set:                   5.004 GiB
peak private usage:                 3.475 GiB
minimum system available memory:    3.456 GiB
CUDA allocated / reserved:          0 / 0
process guards:                    24 / 24 passed
```

All 12 SVG charts were deterministically rasterized with the pinned project
fonts and no font fallback, then opened at original resolution. Titles, labels,
values, bars, axes, and the long guard chart were legible without clipping or
overlap. The rasterization manifest SHA-256 is:

```text
886237bd108645cac015d43d5a8418f0e06fc322b364bbca425370cf4bebfcd7
```

## Conclusion

Stage 188 failed mainly because its safety frontier discarded strict actions
before ranking. The next experiment must first construct a materially broader,
measurably high-recall safety-ranked pool, then apply the existing gain ranker
inside that pool. Merely tuning the ranker would target the smaller failure
component.

## Current-Source Verification

The Stage 184-190 related regression suite passed `41 passed in 4.68s`.
Repository-wide Ruff lint passed, and all 11 changed Python files passed Ruff
format check. The full pytest suite used PID `26576` and one PowerShell
`Wait-Process` call, with no polling or test timeout:

```text
1131 passed, 1 warning in 23.21s
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. After
the process exited, PowerShell exposed an empty child `ExitCode`; it is recorded
as empty rather than fabricated as zero. The complete pytest output reached
100% and reported the passing summary above.
