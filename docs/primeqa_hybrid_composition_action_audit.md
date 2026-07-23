# Stage 181 Counterfactual Composition Action Audit

## Objective

Stage 181 determines which answer-composition modifications improve a frozen
Stage 178A top-three answer without trading away answer fidelity. It is a
train-only diagnostic experiment. Development and test remain closed, no
runtime policy is selected, and no default behavior is changed.

The user-confirmed strict expected-action definition is:

```text
gold citation must not decrease
answer token F1 must not decrease
at least one of citation or F1 must strictly improve
```

Gold document ids and answers are used only to assign offline outcome labels.
The OOF predictability model receives runtime-visible action features only.

The Stage 180 baseline reproduction guard requires an exact gold-citation
count and exact paired citation/F1 deltas. After two stable formal attempts
reproduced those values but measured baseline F1 `0.199876` against the
four-decimal historical value `0.2004`, the user approved an absolute
cross-run baseline-F1 tolerance of `0.001`. The observed difference is
`0.000524`. This tolerance applies only to the historical absolute baseline
check; it does not relax the strict per-action expected label or paired delta
reproduction.

## Frozen Action Boundary

Each answerable train question starts from the frozen Stage 178A top-three
composition. Stage 181 enumerates these interpretable atomic actions:

```text
delete one baseline slot
replace one baseline slot with an alternate from the first 12 eligible sentences
append one alternate when the baseline has fewer than three sentences
keep only the first baseline sentence
keep the first two baseline sentences
take the first three distinct-document sentences
preserve the baseline lead and fill from distinct documents
reproduce the Stage 180 nested-OOF selected action
```

Equivalent selected-sentence sequences are deduplicated. Their family names
are retained as aliases, and the Stage 180 match is retained as an explicit
flag. The unchanged baseline is retained only as the comparison control; it
is not a positive action and is excluded from classifier fitting.

## Outcome Taxonomy

Every unique nonbaseline action is composed and verified against the same
frozen candidate pool as the baseline. It receives exactly one label:

```text
dual_gain
citation_gain_f1_tied
f1_gain_citation_preserved
citation_gain_f1_loss
citation_loss
citation_preserved_f1_loss
neutral
```

The first three classes satisfy the strict expected-action definition. A
floating-point tolerance of `1e-12` is used only to suppress arithmetic noise
around exact equality; it does not permit a substantive F1 loss.

## Five-Fold OOF Predictability

The existing frozen five train folds remain intact at question level. A fixed
class-balanced logistic classifier is fit five times: four folds train and
the fifth fold predicts. Features describe the route, action family, changed
slots, preserved lead, selected and added sentence ranks, document coverage,
retrieval scores, sentence scores, token counts, and query-overlap signals.
No question id, answer text, gold document id, or gold-derived value is a
model feature.

The report includes action-level ROC AUC and average precision, per-fold
metrics, per-question top-1/top-3/top-5 expected-action retrieval, and fixed
question-coverage slices at 10%, 25%, 50%, and 100%. At each slice, questions
outside the slice keep the baseline only for offline measurement. These
slices are diagnostic operating points, not fallback behavior and not a
runtime policy.

## Required Outputs

The public report contains only aggregate statistics:

```text
action and question counts
strict expected-action prevalence
outcome counts by action family, route, and modification pattern
oracle headroom under the strict definition
Stage 180 selected-action attribution
five-fold OOF predictability and coverage curves
resource, timing, source-authorization, and process guards
```

Visualizations cover outcome classes, family success rates, route coverage,
the citation/F1 family plane, OOF fold metrics, coverage precision, and oracle
headroom. Raw questions, answers, document ids, action features, and per-row
labels are forbidden from the public report.

## Decision Boundary

A valid run ends as `stage181_counterfactual_action_audit_complete`. Stage 181
does not select a policy or authorize Stage 182 automatically. Its measured
oracle headroom and OOF separability are presented to the user before any
selective composition policy is designed. Invalid source, split, fold,
reproduction, leakage, retry, fallback, or public-safety guards produce
`stage181_counterfactual_action_audit_invalid`.

## Formal Result

The third formal attempt completed the audit with all 22 process guards
passing. It evaluated 11,928 unique nonbaseline actions over 370 answerable
train questions. Per-question action counts ranged from 32 to 34. All 370
baseline answers reproduced directly, the Stage 180 citation delta reproduced
at `+31`, and its answer-F1 delta reproduced at `-0.000493`.

The strict outcome distribution was:

```text
f1_gain_citation_preserved       5,408
dual_gain                          257
citation_gain_f1_tied                3
neutral                             378
citation_gain_f1_loss               104
citation_loss                       721
citation_preserved_f1_loss        5,057
```

Therefore 5,668 actions (`0.475184`) met the user-approved strict definition,
and 364 of 370 questions had at least one such action. However, 5,408 of those
5,668 actions improved only F1 while preserving citation. Just 260 actions
both safely increased citation and avoided F1 regression, equal to `2.1797%`
of all nonbaseline actions. The strict oracle changed 364 questions, gained
58 gold citations, and increased mean answerable F1 by `0.111694` when
questions without an eligible action retained the baseline.

The reconstructed Stage 180 action was strict-expected for 179 of 370
questions (`0.483784`). It produced 26 dual gains, one citation gain with tied
F1, 152 F1-only gains, 10 citation gains with F1 loss, six citation losses,
144 citation-preserved F1 losses, and 31 neutral outcomes. This explains how
the aggregate policy gained 31 citations while slightly reducing F1.

## Action Patterns

`delete_slot_2` had the highest family-level strict expected rate
(`0.537838`), but its mean citation delta was `-0.059459`; success was entirely
F1-only. The three single-replacement families had strict expected rates from
`0.467267` to `0.484084`. `document_coverage` produced positive mean citation
delta (`+0.043796`) but negative mean F1 delta (`-0.031401`). The reconstructed
Stage 180 family was the only family with both positive mean citation
(`+0.096573`) and positive mean F1 (`+0.001273`) inside its 321 unique primary
actions, while still containing substantial per-question regressions.

The highest route-level strict rate was `error_or_log` at `0.534461`. The two
`security_bulletin_remediation` questions had the lowest rate (`0.258065`) and
mean F1 delta `-0.069156`; that count is too small for a broad route claim.

## OOF Predictability

The fixed five-fold OOF classifier achieved action-level ROC AUC `0.551504`
and average precision `0.513067` against prevalence `0.475184`. Fold ROC AUCs
were `0.490306`, `0.597378`, `0.574898`, `0.560955`, and `0.541652`; fold 1 was
below chance. Per-question expected-action retrieval was `212/370` at top 1,
`293/370` at top 3, and `325/370` at top 5.

Selecting the top-scored action for every question produced citation delta
`+5` and mean F1 delta `+0.007358`, with F1 nonregression in four of five
folds. Fixed 10%, 25%, 50%, and 100% coverage points had strict expected
precision `0.648649`, `0.526882`, `0.583784`, and `0.572973`; their citation
deltas were `-2`, `+2`, `+6`, and `+5`. The 10% result demonstrates that the
single expected/not-expected score does not rank citation-safe gain reliably,
even at its highest-confidence slice.

The diagnostic conclusion is that useful counterfactual actions exist in
nearly every question, but the current runtime feature set weakly separates
them and the binary label is dominated by F1-only gains. A later policy study
should model citation gain and F1 risk separately. Stage 181 itself selects no
policy and does not authorize Stage 182 automatically.

## Attempt And Verification Audit

Attempt 1, PID `22760`, and attempt 2, PID `23288`, both completed naturally
but were invalid because the historical absolute baseline-F1 guard compared
current `0.199876` with the Stage 180 four-decimal value `0.2004`. Citation
reproduced exactly at `183`, and paired Stage 180 deltas reproduced exactly in
both runs. Attempt 2 added the missing actual/expected diagnostics. Their
reports, visualizations, and logs remain in explicit attempt directories.

After the user selected option A, the protocol adopted the bounded `0.001`
historical absolute-F1 tolerance described above. Attempt 3, PID `17468`, then
completed naturally with all guards passing. Each attempt used one
PowerShell `Wait-Process` call on its PID and no polling. Child `ExitCode`
remained blank; the outer commands returned zero, `HasExited=True`, and stderr
contained only model-weight loading progress.

Attempts 1/2 produced OOF AUC `0.551511`, while attempt 3 produced `0.551504`;
top-1 strict hits changed from 214 to 212. The fixed model random state and
data were unchanged, so this small CPU numerical/tie instability is retained
as observed rather than hidden. It does not change the weak-separability
conclusion.

The completed analysis wall time was `176.994119` seconds. Peak working set
and private usage were `4,029,288,448` and `5,472,903,168` bytes; minimum
system-available memory was `1,789,669,376` bytes. CUDA allocation and
reservation were zero. The report SHA-256 is:

```text
a9c557d7346eb2b4958cddd2505937eba828556c7671d7e936bf883d80cfe88b
```

All seven SVGs parsed. Browser rendering exposed overlapping scatter labels;
the chart was changed to unlabeled points plus a fixed right-hand legend and
regenerated without changing the JSON report. A fresh isolated Edge profile
rendered the corrected title, axes, points, and legend without overlap.

Final current-source verification produced:

```text
Stage 181 focused tests: 10 passed in 1.00s
five Stage 181 Python files: Ruff format check passed
full repository Ruff lint: passed
full repository pytest: 1070 passed, 1 warning in 20.98s
full pytest PID / outer command exit: 11608 / 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation.
