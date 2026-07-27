# Stage 199 Joint Risk-Signal and Winner-Rule Nested CV

## Objective

Stage 199 executes the Stage 198 train-only factorial design. It tests whether
changing the unsafe-risk signal, the final winner rule, or both can satisfy the
existing Stage 196 safety and quality constraints. Development and test remain
closed throughout this experiment.

## Experiment

The five outer contexts retain their frozen Stage 196 source pool, gain model,
risk profile, risk weight, and prefix size. Each of the 20 inner partitions
fits two citation/F1 safety heads, one gain LambdaMART model, the source unsafe
classifier, and one pairwise safety LambdaMART model. These shared predictions
evaluate the full factorial grid:

```text
risk signals:                         4
winner rules:                         7
cells per outer context:             28
inner partitions:                    20
observed model fits:                100
observed LightGBM trees:         18,000
private predictions:            245,960
outer refits:                         0
fallback or retry:                    0
```

The exact `source_weighted_classifier x gain_only` control reproduced the
Stage 196 evidence in all five outer contexts. No private training row or
prediction row was written to the public report.

## Results

All 28 cells were evaluated in every outer context, but none satisfied the
full inner eligibility contract. Eligible cell counts were therefore
`0, 0, 0, 0, 0`, no outer refit was permitted, and paired outer bootstrap
metrics are correctly unavailable. The zero-valued outer aggregate means
"not evaluated"; it is not a measured zero-quality policy.

The top-inner candidate diagnostics show the remaining conflict:

```text
conditional strict capture:  0.647059 to 0.685512
unsafe selection rate:        0.272414 to 0.327759
required capture:             >= 0.68
required unsafe rate:         <= 0.25
```

Some contexts reached the capture boundary, but none simultaneously met the
unsafe-rate and all other per-fold constraints. The first-stage pool remained
high recall, so the dominant failure is still final discrimination and
selection inside the retained pool rather than missing strict opportunities
before ranking.

Complete-pool unsafe discrimination improved across the new signals, but
remained modest:

```text
source weighted classifier:              ROC AUC 0.590635
decomposed loss risk:                     ROC AUC 0.597567
pairwise safety ranker:                   ROC AUC 0.598221
decomposed + pairwise rank fusion:        ROC AUC 0.603601
```

The fusion signal had the strongest mean conditional capture (`0.539760`) and
the best complete-pool AUC. The source classifier had the lowest marginal mean
unsafe rate (`0.282915`). This split confirms that better global risk ordering
alone did not resolve the per-question capture/safety frontier.

Winner-rule marginals make the tradeoff explicit. `gain_only` achieved mean
capture `0.615705` at unsafe rate `0.338514`. A top-4 gain shortlist followed
by minimum risk reduced unsafe rate to `0.182601`, but capture collapsed to
`0.328700`. Increasing rank-utility risk weight showed the same monotonic
safety-versus-capture exchange rather than a jointly superior operating point.

## Decision

The formal status is `stage199_joint_risk_winner_insufficient`. The experiment
is valid, but the candidate family is rejected. Stage 199 does not authorize
full-train policy selection, replacement, runtime E2E, development, test, or
default runtime activation.

The next protocol should attribute inner eligibility failures by constraint,
fold, action-loss type, and question context. It should determine whether the
remaining conflict needs a stronger question-conditional safety model, a
different learning objective, or a revised candidate representation. It must
not relax the frozen gates or promote a top-ineligible cell.

## Execution Record

The first formal PID `14676` completed the 100 inner fits and emitted the final
decision to stdout, then failed during visualization with `KeyError: 'metric'`.
The real gate contract uses `name`; no formal JSON report was persisted from
that run. The defect was fixed by consuming the real contract and atomically
persisting core evidence before visualization postprocessing. A regression
test now proves that a visualization exception cannot erase the core report.

Formal rerun PID `25716` used one PowerShell `Wait-Process` call and waited for
natural completion. There was no polling, experiment timeout, retry, fallback,
OOM, or CUDA allocation. The reported wall time was `486.661838` seconds:

```text
dependency and memory authorization:   0.831653 s
Stage 182 reproduction:               203.679400 s
Stage 199 nested CV:                  282.150785 s
```

Peak working set was `3.759 GiB`, peak private usage was `4.088 GiB`, and
minimum system-available memory was `2.912 GiB`. All `34/34` process guards
passed. PowerShell's post-wait child `ExitCode` field was empty and is retained
as unknown; report integrity, decision validity, and all guards were verified
from the persisted output.

## Visual Verification

All 15 SVGs passed XML parsing and were rasterized with pinned
`resvg_py==0.3.3`, project-owned Poppins fonts, a white background, and no font
fallback. All PNGs were opened at original resolution. Titles, labels, values,
17 gates, 34 process guards, zero values, and negative deltas are visible
without clipping or overlap. The selected-factor chart is intentionally empty
because no outer context had an eligible cell; it is not a rendering failure.

Representative views:

- [complete-pool risk AUC](../artifacts/primeqa_hybrid_joint_risk_winner_stage199_visuals_png/stage199_complete_pool_risk_auc.png)
- [winner-rule capture](../artifacts/primeqa_hybrid_joint_risk_winner_stage199_visuals_png/stage199_winner_rule_capture.png)
- [winner-rule unsafe rate](../artifacts/primeqa_hybrid_joint_risk_winner_stage199_visuals_png/stage199_winner_rule_unsafe.png)
- [advancement gates](../artifacts/primeqa_hybrid_joint_risk_winner_stage199_visuals_png/stage199_advancement_gates.png)

Artifact hashes before final source verification:

```text
formal report SHA-256:
5b933f524fff1bceb4d4d842e4f3a1aec3160aa3ed337131444ec1b7c2699fee

resvg manifest SHA-256:
af31be88acd15b9c57bb983fef0a2b7bcd8b7abd3851c5ddf56416b015fb0e28
```

## Current-Source Verification

Repository-wide Ruff lint, all five changed Python-file format checks,
`pip check`, CLI help, and `git diff --check` passed. The Stage 194-199 focused
regression set passed `42 tests in 13.05s`.

The complete pytest suite used the single Python PID `26548`. Its PowerShell
command called `Wait-Process` exactly once and waited for natural completion,
without polling or a pytest timeout. The result was
`1197 passed, 1 warning in 40.75s` with exit code `0`. The warning is the
existing FastAPI/Starlette `TestClient` deprecation.
