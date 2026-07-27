# Stage 194: Safety-Constrained LambdaMART Train-Only Nested CV

## Scope

Stage 194 implemented the Stage 193 frozen train-only design:

- Stage 191 cap-16 safety pools with baseline unioned after the cap;
- LightGBM LambdaMART labels `unsafe/safe-zero/strict-gain = 0/1/2` and
  `label_gain = [0, 1, 4]`;
- an independent LightGBM unsafe-risk head;
- 64 shared-model configurations and grouped five-by-four nested CV;
- no dev/test, fallback, retry, weak-candidate substitution, runtime E2E,
  full-train selection, replacement decision, or default activation.

Formal report:

```text
artifacts/primeqa_hybrid_safety_constrained_lambdamart_stage194.json
SHA-256: c1208348e79fd404e7b360a49a3f4d4e9663a3e9bd61c3ef99cf3f9ac60ece57
```

## Dependencies

The frozen LightGBM wheel was downloaded and verified before installation:

```text
lightgbm-4.7.0-py3-none-win_amd64.whl
SHA-256: f42d1e5b32b6f170e606d7c689c6165671da98d7bf37f1addec2623efc8740c9
```

The verified installation exposed a real Stage 193 omission: LightGBM 4.7.0 requires
`narwhals>=1.15`. The user selected route A, so Stage 194 pinned and verified:

```text
narwhals-2.24.0-py3-none-any.whl
SHA-256: 42fdedf44e5b2ca7505630d45b4ac3058f38d8485cba9fe1652ca23152df7489
```

The formal process rechecked versions and wheel hashes and ran `pip check`, which returned
`No broken requirements found.` The project now exposes a reproducible `ranking` optional
dependency containing these exact versions.

## Resource Amendment

Stage 193 froze a 6 GiB preflight. The first actual preflight observed 5.599 GiB and did
not start a formal process. The user explicitly authorized proceeding below 6 GiB. Stage
194 did not rewrite the frozen value; the report preserves:

```text
frozen threshold:       6.000 GiB
formal available:       5.595 GiB
explicit user override: true
model grid reduced:     false
fallback enabled:       false
```

Historical evidence was Stage 188 peak working set 4.744 GiB with 2.953 GiB minimum free,
and Stage 191 peak working set 5.724 GiB with 4.247 GiB minimum free.

## Verification

A real sparse smoke fit with warnings promoted to errors produced exactly 8 model fits,
1,200 trees, 60 question groups, 180 rows, and a group-size sum of 180. It verified both
safety estimators, both tree profiles, sparse prediction, and the LightGBM 4.7.0 API detail
that `eval_at=[1]` belongs on `LGBMRanker.fit()`.

The formal process used PID `27624`. One PowerShell command started that PID and made one
`Wait-Process` call until natural completion. There was no polling, experiment timeout,
retry, partial continuation, fallback, OOM, or CUDA allocation.

```text
Stage 182 reproduction:       216.683980 s
Stage 194 nested CV:          353.060593 s
model fit time:               257.255739 s
total wall time:              570.604419 s
process CPU time:            1991.234375 s
```

The child completed the CLI, wrote the report and all charts, and all 33 process guards
passed. After `Wait-Process`, the PowerShell process object's `ExitCode` field was empty;
the wrapper command therefore raised after completion. The empty value is retained as an
observed monitoring limitation and is not represented as exit code 0.

## Counts And Resources

No inner candidate was eligible, so the no-substitution rule prevented every outer refit.
The exact inner-only counts were:

```text
inner partitions:                    20
pool-safety fits:                   160
LambdaMART fits:                     80
unsafe-head fits:                    80
all model fits:                     320 / 400 maximum
LightGBM trees:                  48,000
group-contract validations:          40
private predictions:            393,536
public training/prediction rows:       0
```

```text
peak working set:             3.762 GiB
peak private usage:           4.179 GiB
minimum system free:          3.756 GiB
CUDA allocated/reserved:      0 / 0 GiB
event-driven snapshots:       29
```

The user-authorized amendment was sufficient; no OOM occurred and the full grid remained.

## Result

```text
status:                     stage194_safety_constrained_lambdamart_insufficient
experiment valid:          true
candidate family accepted: false
inner-eligible configs:    0 / 64 in every outer context
outer folds evaluated:     0 / 5
```

Because no inner configuration was eligible, outer aggregate metrics are zero and the
paired bootstrap is unavailable. These zeroes mean **not evaluated**, not zero model
quality. No weaker candidate was substituted.

The pool is no longer the bottleneck. The leading inner candidates show:

| Outer context | Pool recall | Conditional capture | Strict precision | Unsafe rate |
| --- | ---: | ---: | ---: | ---: |
| fold 1 | 0.989691 | 0.670139 | 0.657343 | 0.318644 |
| fold 2 | 0.986207 | 0.636364 | 0.629496 | 0.342373 |
| fold 3 | 0.986348 | 0.629758 | 0.609756 | 0.331104 |
| fold 4 | 0.989510 | 0.681979 | 0.659420 | 0.300000 |
| fold 5 | 0.986486 | 0.654110 | 0.635135 | 0.325581 |

Fold 4 is the clearest boundary case. Its best configuration passed pool recall,
conditional capture, strict precision, citation/F1 aggregate nonregression, and per-fold
capture checks, but unsafe selection was 0.300000 instead of at most 0.25. Other contexts
also missed capture or precision. LambdaMART approached the gain target, but the risk rank
did not suppress unsafe winners enough. Increasing rank penalty alone did not satisfy all
constraints simultaneously.

## Visuals And Corrections

Twelve SVG charts were XML-validated and rasterized with fixed `resvg_py==0.3.3`, bundled
Poppins fonts, white background, and no fallback. All PNGs were opened at original
resolution and checked for clipping, overlap, and blank output.

```text
resvg manifest SHA-256:
6e24d26c7aaf61844e3f662d1457986bf228eb7b54340d3fa7a65691de0baccb
```

Real process corrections, none of which caused a second formal model run:

1. Invalid PowerShell `Select-Object -Single` stopped before installation.
2. LightGBM import then exposed missing `narwhals`; the user selected version 2.24.0.
3. A file preflight used the wrong raw-data root and stopped before creating a PID.
4. The corrected 5.599 GiB preflight stopped below the frozen threshold; the user then
   explicitly authorized the amendment.
5. The formal process completed, but the outer command treated its empty post-wait
   `ExitCode` as failure after artifacts were written.
6. The first rasterization command omitted `--input-dir`; the corrected command rendered
   all 12 files. A multi-image preview displayed one chart incompletely, but direct SVG and
   single-image inspection confirmed the deterministic file was complete.

## Next Stage

Stage 194 does not authorize dev/test, full-train fitting, runtime E2E, replacement, or
default activation. Stage 195 should remain train-only and freeze a safety-first constrained
selection study comparing cost-sensitive unsafe heads and deterministic safest-prefix
frontiers while preserving the cap-16 pool and LambdaMART gain scores. It should test
whether unsafe rate can fall below 0.25 without losing the approximately 0.68 conditional
capture boundary. The current thresholds must not be relaxed and called a pass.

## Final Validation

Full-repository Ruff lint passed, all five changed Python files passed Ruff format check,
`pip check`, CLI help, formal artifact hash checks, and `git diff --check` passed. The first
attempt to orchestrate these independent checks used an invalid JavaScript string and
failed before any nested command started; corrected quoting was then used and all checks
passed.

Full pytest used the single Python PID `12924` and one PowerShell `Wait-Process` call until
natural completion, with no polling or pytest timeout:

```text
1157 passed, 1 warning in 27.65s
stderr: empty
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. The post-wait child
`ExitCode` field was empty and is retained as empty rather than represented as zero.
