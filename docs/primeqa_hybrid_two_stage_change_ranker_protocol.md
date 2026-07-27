# Stage 205: two-stage change/abstain gate and conditional ranker protocol

## Scope

Stage 205 freezes the Stage 206 train-only experiment selected by the user as route A. It
reads only the public Stage 204 failure-attribution report and the public Stage 202 source
protocol. It does not load train, development, test, or document rows; fit a model; generate
predictions; evaluate a policy; change runtime behavior; relax a quality gate; retry; or add a
fallback.

## Evidence

Stage 204 observed that the Stage 203 grouped softmax mixed two decisions: whether to keep the
baseline and which answer-bearing action to choose. Across precision-adjacent comparisons,
strict count changed by `-1,934`, unsafe count by `-1,105`, and baseline count by `+3,154`.
The dominant changing transition was `strict_success -> baseline`. Meanwhile, `1,439 / 1,480`
question contexts contained a strict opportunity, with a mean of `8.043919` strict actions in
the fixed pool. These are train-only associations under the Stage 203 grid, not universal causal
claims.

## Frozen architecture

The candidate pool remains the Stage 196 cap-16 pool with baseline unioned afterward. The
conditional ranker fits, normalizes, and selects only among nonbaseline actions. Baseline is not
allowed into its rows, softmax, or winner selection.

Two grouped LambdaMART ranker targets form a controlled ablation:

1. `strict_binary`: unsafe and safe-zero `0`, strict-success `1`, `label_gain=[0,1]`.
2. `strict_safety_graded`: unsafe `0`, safe-zero `1`, strict-success `2`,
   `label_gain=[0,1,4]`.

After a ranker chooses one nonbaseline action, a separate binary LightGBM gate predicts whether
that selected action is strict-success. The gate sees only runtime-available winner, baseline,
difference, within-question min-max normalized rank-margin, and source safety features. Raw
absolute LambdaMART scores are excluded because independently fitted cross-fit rankers need not
share a comparable margin scale. It never receives gold labels or outcome features at runtime.
It changes to the conditional winner when the score reaches the learned threshold; otherwise it
deliberately keeps baseline.

## Leakage control

Gate training rows may not use winners produced by a ranker fitted on those same questions.
Inside every inner training partition, four question-grouped cross-fit rankers each predict only
their held-out questions. Their winners create exactly one OOF gate row per question. Only after
all OOF gate rows exist may the full inner ranker be fit to select winners for the inner held-out
partition. The four-way assignment is the deterministic SHA-256 of outer context, inner held-out
context, stable question identifier, and seed 205 modulo four. Outer refit repeats the same
cross-fitting pipeline.

The two source safety heads follow the same four cross-fit boundaries and predict only cross-fit
held-out questions when their scores construct gate-training pools and features. Their OOF
predictions are shared by both ranker families. Same-fit source safety predictions are forbidden
for gate-training winners; this stricter amendment was explicitly selected by the user after the
initial Stage 205 freeze.

This extra layer is required because a same-fit ranker would make the gate learn from optimistically
selected winners and would leak ranking fit behavior into gate supervision.

## Candidate grid and validation

Each ranker family uses target change coverages `{0.25, 0.40, 0.55, 0.70, 0.85}`. A threshold is
the training-OOF score order statistic for the requested coverage and is reused on held-out data
without tuning. No probability-calibration claim is made. The factorial has 10 two-stage policies
plus one exact Stage 196 control, for 11 candidates per outer context.

Stage 206 retains 5 outer folds and 4 inner folds, with every question group kept intact. The
original 13 inner eligibility constraints and 17 advancement gates remain numerically unchanged.
If an outer context has no eligible candidate, it records failure and does not substitute a weaker
candidate.

## Budget and authorization

Each inner partition has at most 4 full source fits, 8 cross-fit source-safety fits, 10
conditional-ranker fits, and 2 gate fits. Across 20 inner partitions and at most 5 selected outer
refits, the maximum is 570 model fits and 96,000 LightGBM trees. The extra safety estimators are
not LightGBM models, so the LightGBM tree bound is unchanged. These are Stage 206 upper bounds,
not Stage 205 observed usage.

The formal Stage 206 process must use one PowerShell `Wait-Process` call for one PID until natural
exit, without polling or an experiment timeout. Stage 205 authorizes only the Stage 206 train-only
experiment. Development, test, full-train selection, replacement, runtime E2E, Stage 178B, and
default activation remain closed.

## Formal freeze result

The first local draft correctly failed its budget guard because the initial expected tree bound
was `94,500`; recomputation showed that outer refit also includes two source LightGBM models, so
the correct bound is `96,000`. That invalid draft was removed and was never treated as the formal
protocol. A final pre-commit audit then found that raw absolute LambdaMART scores were still
listed as gate features even though separately fitted cross-fit rankers need not share a score
scale. The contract was tightened to use only a within-question normalized margin and gained two
guards for raw-score exclusion and deterministic cross-fit assignment. The intermediate report
hash `2f9db604bb4733270e9b522686d7f8ef24a0d0ad36bdea5ddb9dda2e1e6a7b69` and manifest hash
`57ad74ac245773d9f9427d79f5fefde3ee707af57c086290d8e2c7c758c8ae8d` were superseded.

The user subsequently selected the stricter source-safety OOF amendment before Stage 206 began.
This supersedes the earlier `370-fit` report: the two source safety heads now cross-fit in
every gate fold, shared across ranker families. The prior report hash
`a967893d26d1ca58164893f8cd804ea1883d870863d6c599da42d447fa59df81` and manifest hash
`817fba82e86299dff4b0299cd5d4b1842b4c5eafa55e1d0c1b19f4cedc441678` are retained only as
history. The amended formal freeze is:

```text
status:        stage205_two_stage_change_ranker_protocol_frozen
guards:        84 / 84 passed
model fits:    0
private rows:  0
dev/test:      closed / closed
visuals:       10 SVG + 10 PNG
report hash:   0988f97e7e30e6772cc7a7c9738a9e1d285f698629b5b0ccf48c1701e36de02a
manifest hash: 531d66f4a7730cea099576ee5cf1d63b6bb455cf408b7881cba4c9ba8c2ad372
```

Repository-wide Ruff, format checks for the three changed Python files, `pip check`, CLI help, and
`git diff --check` passed. Full pytest on the revision immediately before the final raw-score
contract tightening reached 100% with `1253 passed, 1 warning in 45.74s`; stderr was empty and the
warning remains the pre-existing FastAPI/Starlette `TestClient` deprecation. The Python process was
launched once and the PowerShell wrapper issued one `Wait-Process`, but the Codex command channel
terminated that wrapper after about 14 seconds. Pytest continued to natural completion and wrote
the complete result above; the wrapper's post-wait exit-code and PID text were not returned and are
therefore recorded as unknown rather than reconstructed. After the final metadata-only protocol
tightening and the strict source-safety amendment, the focused Stage 202-205 current-source
regression set passed `40` tests. The full suite was not rerun merely to manufacture the missing
wrapper fields.
