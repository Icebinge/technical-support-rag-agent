# Stage 198 Joint Risk-Signal and Winner-Rule Protocol

## Objective

Stage 198 freezes the user-confirmed route A for Stage 199: a train-only
factorial ablation that independently varies the risk signal and the final
winner rule. It addresses both dominant Stage 197 findings rather than
assuming that either the risk model or pure-gain selection is the sole cause.

This stage is a protocol freeze only. It reads the public Stage 197 aggregate
report, loads no train/development/test rows or documents, imports no LightGBM
runtime, fits no model, produces no predictions, and evaluates no policy.

## Fixed Source Trajectories

Each outer context retains its published Stage 196 top-inner pool builder,
gain ranker, risk representation/profile/weight, and safest-prefix size. All
five trajectories were exactly reconstructed by Stage 197. Stage 199 must not
reopen those old factors.

The exact control cell is:

```text
source weighted unsafe classifier x gain-only winner
```

It must reproduce the corresponding Stage 196 top-inner evidence before any
new result is accepted.

## Risk-Signal Factor

Four risk signals are frozen:

1. `source_weighted_classifier`: exact Stage 196 binary-classifier control.
2. `decomposed_loss_risk`: `max(P(citation loss), P(F1 loss))` from the source
   safety heads, requiring no additional fit.
3. `pairwise_safety_ranker`: question-grouped LightGBM LambdaMART with
   `unsafe=0`, `non_unsafe=1`, `label_gain=[0,1]`, and truncation level 16.
4. `decomposed_pairwise_rank_fusion`: mean deterministic within-question
   normalized risk-rank fraction from signals 2 and 3.

Every signal is consumed only as a within-question order. No absolute risk
probability threshold or calibration claim is permitted.

## Winner-Rule Factor

Seven winner rules are frozen:

- one exact `gain_only` control;
- rank utility with risk penalties `0.25`, `0.50`, `1.00`, and `2.00`;
- gain shortlist sizes `2` and `4`, followed by lowest-risk selection.

Rank utility minimizes normalized gain-rank fraction plus the risk penalty
times normalized risk-rank fraction. It never adds incomparable raw
LambdaMART margins and classifier probabilities. Shortlist policies do not
force baseline into the shortlist; all deterministic ties end with canonical
action order.

Each risk signal rebuilds the fixed-size safest prefix from the source cap-16
pool, then unions baseline. This lets the risk-only cells measure frontier
changes, while the winner-only cells retain the source risk signal.

## Factorial Ablation

```text
risk signals:                  4
winner rules:                  7
policy cells per outer:       28

control:                       source risk x gain only
risk-only ablation:            alternate risk x gain only
winner-only ablation:          source risk x alternate winner
joint ablation:                alternate risk x alternate winner
```

Stage 199 must publish factor aggregates and paired deltas against the exact
control. Models are shared across policy cells; 28 cells do not mean 28 model
fits per partition.

## Execution Budget

Each partition fits two source safety heads, one source gain ranker, one source
unsafe classifier, and one pairwise safety ranker.

```text
inner partitions:                  20
maximum outer refits:               5
fits per partition:                 5
maximum model fits:               125
LightGBM models per partition:      3
maximum LightGBM trees:        22,500
minimum preflight memory:        4 GiB
```

The formal Stage 199 process must use one PowerShell `Wait-Process` call for
one PID until natural completion. Insufficient memory stops the launch; it
does not authorize reducing the factorial grid, retrying, or adding fallback.

## Selection and Gates

All 13 inner eligibility constraints and 17 outer advancement gates remain
numerically unchanged from Stage 196. In particular:

```text
strict-success precision:              >= 0.65
first-stage pool recall:                >= 0.95
conditional strict capture:            >= 0.68
unsafe selection rate:                  <= 0.25
new F1 regression rate:                 <= 0.02
```

An outer context with no eligible inner configuration records no-eligible and
does not substitute a weaker candidate.

## Formal Freeze

The formal freeze read the exact Stage 197 report SHA-256 and completed all
`62/62` guards. Its status is
`stage198_joint_risk_winner_protocol_frozen`. Only Stage 199 train-only nested
CV is authorized. Development, test, full-train policy selection, runtime E2E,
replacement selection, Stage 178B, and default activation remain unauthorized.

```text
load public report:     0.000653 s
freeze and guard:       0.000053 s
total:                  0.000706 s

formal report SHA-256:
62658919388603cdd2c85432399d45dfd0f50148ea4e521119f35a0d2e2e3330

resvg manifest SHA-256:
e6c3e97ba4c16efb7bee768a820ba5831af088c22e83813c6b86116354f5c593
```

Ten SVGs passed XML parsing and were rasterized with pinned
`resvg_py==0.3.3`, project-owned Poppins fonts, a white background, and no font
fallback. All PNGs were opened at original resolution. Titles, values, 17
advancement gates, 62 guard rows, and false/zero states are visible without
clipping, overlap, or blank output.

The stated model counts are frozen budgets, not observed Stage 199 execution
or effectiveness results. No claim is made yet that any of the 28 cells lowers
unsafe selection.

## Current-Source Verification

Repository-wide Ruff lint, all three changed Python-file format checks,
`pip check`, CLI help, and `git diff --check` passed. The Stage 195/197/198
focused regression set passed `18 tests in 1.14s`.

The complete pytest suite used the single Python PID `8144`. Its PowerShell
command called `Wait-Process` exactly once and waited for natural completion,
without polling or a pytest timeout. The result was `1186 passed, 1 warning in
39.15s`; stderr was empty. The warning is the existing FastAPI/Starlette
`TestClient` deprecation. PowerShell's post-wait child `ExitCode` field was
empty and is retained as unknown rather than being fabricated as zero.
