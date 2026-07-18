# Stage 162: Conservative Context-Swap Nested Training

## Status

Stage 162 completed a train-only nested cross-validation experiment for four
conservative learned swap families. Untouched Stage116 RRF Top10 is the primary
safety baseline. Development and test were not loaded.

The corrected formal run passes all 18 process guards, but no model family
passes every strict outer-fold quality guard. No configuration is selected, no
model artifact is written, and runtime behavior remains unchanged.

The corrected public report SHA-256 is
`ff126db5efc2b117ab77cf99a62ec5c399110a938b3a37ea449055e76e622d93`.

## Protocol

Stage162 authorizes these exact sources before train is parsed:

```text
Stage161 report: a13b8ee5538581f0eb87a649c48fdf4ae715b6cfa8a43a97b5115001f9cd1197
Stage80 report:  2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f
train split:     cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155
technote corpus: f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b
```

Stage161 must have all 18 process guards passing and decision status
`primeqa_hybrid_protected_context_selector_no_train_cv_safe_config`, with no
selected model and both dev and test closed.

The train split contains 562 rows: 370 answerable and 192 unanswerable. It uses
the established normalized-question plus answer-document or `UNANSWERABLE`
grouping. Five inherited outer folds contain `113/113/112/112/112` rows and
have assignment SHA-256
`a41cc9c1d00c057c774d9d7e55390c8dfa56699d19513fff205b6d184e7988a8`.

The candidate pool is the exact original Stage116 RRF Top200. All 562 pools
have depth 200, producing 112,400 in-memory candidate records. Gold is present
for `345/370` answerable rows. Raw candidate and case rows are not written.

The four families are:

```text
protected RRF prefix 8, promotion budget 2, pairwise logistic
protected RRF prefix 8, promotion budget 2, histogram GBDT
protected RRF prefix 9, promotion budget 1, pairwise logistic
protected RRF prefix 9, promotion budget 1, histogram GBDT
```

Each selector starts from untouched RRF Top10. A model scores only the
unprotected rank-9/10 incumbents and rank-11-to-200 challengers. A challenger
replaces an incumbent only when its score margin is strictly greater than the
selected threshold. Selected documents retain original RRF order.

For each outer fold, threshold selection uses only the other four folds. Four
inner fits each train on three inherited folds and predict the fourth. The
threshold grid is fixed zero plus positive inner-OOF score-margin quantiles
`Q50/Q70/Q80/Q90/Q95`. The outer validation fold never contributes to its
threshold. Each family therefore runs 20 inner fits and five outer refits.

Inner selection is lexicographic by hit count, all-answerable F1, citations,
refusals, unanswerable false answers, swap count, and conservative threshold.
This selects a hyperparameter inside outer train; it does not replace strict
outer evaluation.

Final family selection requires a strict context-hit improvement over RRF,
non-regressing aggregate F1, citations, refusals, and unanswerable false
answers, exact protected prefixes, no promotion-budget violation, and no hit
or F1 regression in any outer fold. There is no fallback or guard relaxation.

## Corrected formal result

The corrected formal process ran once with explicit CUDA query encoding and
ended naturally with exit code 0. It had no runtime limit, monitoring deadline,
automatic termination, restart, retry, or fallback. Total time was
`380.621178s`, including `287.516420s` for four nested configurations.

Controls reproduce Stage161 exactly:

| Control | Gold in Top10 | F1, all answerable | Gold citations | Answerable refusals |
| --- | ---: | ---: | ---: | ---: |
| Current query overlap | 175 | 0.194072 | 151 | 1 |
| Untouched RRF | 255 | 0.201990 | 177 | 1 |

Nested OOF results are:

| Family | Gold in Top10 | F1, all | Citations | Swaps | Minimum fold hit delta | Minimum fold F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix8/budget2 pairwise | 255 | 0.202566 | 177 | 698 | -0.040000 | -0.000917 |
| prefix8/budget2 histogram | 251 | 0.199708 | 177 | 325 | -0.053333 | -0.010966 |
| prefix9/budget1 pairwise | 257 | 0.202710 | 178 | 217 | 0.000000 | -0.002368 |
| prefix9/budget1 histogram | 255 | 0.201990 | 177 | 6 | 0.000000 | 0.000000 |

The closest family is prefix9/budget1 pairwise. Relative to RRF it gains two
Top10 gold documents, raises aggregate F1 by `0.000720`, adds one gold
citation, and reduces answerable refusals from one to zero. It never regresses
outer-fold hit rate, but one outer fold has F1 delta `-0.002368`; the strict
every-fold F1 guard rejects it.

Prefix9/budget1 histogram performs only six swaps across 562 rows and exactly
matches RRF quality. It passes every non-regression guard but fails the required
strict hit improvement. The other two families regress outer-fold hit and F1.

All four candidates keep unanswerable false answers at `191/192`. This is only
a non-regression result and is not acceptable final answer behavior.

Selector-only average latency is `0.783390ms` and `0.798980ms` for the two
pairwise families, versus `1.666222ms` and `1.837422ms` for histogram. These
figures exclude retrieval-channel construction, candidate-pool construction,
and answer generation.

The final decision is:

```text
process guards: 18 / 18
completed nested families: 4 / 4
selectable families: 0 / 4
selected family: none
dev/test loaded: false / false
model artifact written: false
runtime/fallback changed: false / false
```

## Visualizations

Ten public-safe SVGs cover context hits, verified F1, citations, average swaps,
selector latency, selected thresholds, minimum outer-fold hit and F1 deltas,
family guard status, and process guards. All ten parse as XML. Pixel-level
screenshot review is not claimed.

## Progress-label correction

The first formal algorithm run ended naturally with exit code 0, passed 18/18
guards, and produced identical frozen thresholds and quality results. Its
report SHA-256 is
`8b7c4b1669643b4ab69e43a27b06d943ed6f92ac397e2d628d9d6b9dfb210999`.

During that run, the reused Stage161 candidate builder emitted 23 candidate
progress events with `stage: Stage 161`. The process, algorithm, output path,
and later events were Stage162, so this was an observability-label defect rather
than a data or metric defect. The run was not interrupted or restarted.

The user was offered three explicit choices and selected A: preserve the first
run, add an explicit progress-stage parameter, retain Stage161 as the default,
pass Stage162 from the new training runner, and run one corrected formal. The
first report, logs, exit evidence, and ten SVGs remain in ignored
`artifacts/stage162/pre_correction_*` paths.

The corrected run contains 23 candidate-build progress events, all labeled
Stage162, and zero Stage161 events. Its four selected threshold sets, stable
quality metrics, swap audits, and family guards exactly match the first run.
Only measured timing and selector latency differ.

An initial comparison audit incorrectly included runtime latency in an exact
equality requirement and printed four false mismatches. It then attempted to
decode PowerShell output as UTF-8 and failed with `UnicodeDecodeError`. That
audit did not alter the formal result. A corrected audit compares only frozen
thresholds, quality fields, swap audits, and guards, and scans logs as bytes;
all four families match and the corrected log has no traceback.

Corrected formal stderr is nonempty at 658 bytes. It contains successful model
weight-loading and PowerShell first-use progress, not a Python traceback.

Current-source verification after documentation is:

```text
Stage161/162 six-file Ruff format check: passed
full repository Ruff lint: passed
Stage161/162 targeted pytest: 28 passed in 2.94s
full repository pytest: 843 passed, 1 existing warning in 12.88s
full pytest exit code: 0
```

The warning is the existing FastAPI/Starlette `TestClient` deprecation. The
single full pytest process had no test timeout, monitoring deadline,
termination, or restart and exited naturally. Its 382-byte stderr contains
only PowerShell first-use CLIXML progress and no traceback or failed test.

## Next direction

The learned swap family should stop here under the strict protocol. The
train-only evidence now favors untouched RRF as the generation-context
baseline: it is much stronger than current query overlap, while learned swaps
either regress a fold or reduce to near identity.

The next eligible stage is to freeze untouched RRF Top10 as a fixed candidate
policy and run one independent development comparison against current query
overlap. Test, runtime defaultization, fallback, query rewrite, and second
retrieval remain closed.
