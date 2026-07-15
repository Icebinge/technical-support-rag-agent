# PrimeQA Hybrid Validation-Failure Pattern Analysis

This document records Stage 107.

## Scope

Stage 107 freezes and executes a public-safe validation-failure diagnostic for
the current PrimeQA hybrid answer pipeline.

This stage reads only the saved public-safe Stage102, Stage105, and Stage106
JSON reports. It does not load train/dev/test split files, does not load corpus
documents, does not run retrieval or answer metrics, does not run final metrics,
does not select from dev-only observations, does not add fallback strategies,
and does not change runtime defaults.

The analysis was explicitly confirmed for this run:

```text
route_id: primeqa_hybrid_validation_failure_pattern_analysis
confirmed: true
confirmation_note: user confirmed Stage107 frozen validation-failure pattern analysis on 2026-07-16; read saved Stage102/105/106 public-safe reports only; test locked; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\analyze_primeqa_hybrid_validation_failure_patterns.py ^
  --user-confirmed-analysis ^
  --confirmation-note "user confirmed Stage107 frozen validation-failure pattern analysis on 2026-07-16; read saved Stage102/105/106 public-safe reports only; test locked; runtime defaults unchanged; no fallback strategies"
```

The run completed successfully and wrote console output to:

```text
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107.console.txt
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107.json
```

## Frozen Diagnostic Protocol

```text
protocol_id: primeqa_hybrid_validation_failure_pattern_analysis_v1
validation_split: dev
source reports only: Stage102, Stage105, Stage106
load split files: false
load corpus documents: false
run retrieval metrics: false
run answer metrics: false
run final test metrics: false
select config from dev: false
retune thresholds on dev: false
case-level rows written: false
aggregate counts only: true
```

Failure buckets are all Stage102 buckets except
`answer_supported_and_cited`.

## Validation Failure Pattern

Stage107 found that the dev failure is broad rather than a single isolated
route issue:

```text
dev rows: 121
dev failure rows: 117 / 121 = 0.9669
answerable dev failure rows: 76 / 76 = 1.0000
unanswerable false-answer rows: 41 / 45 = 0.9111
```

Top dev failure buckets:

| Bucket | Count | Overall rate | Conditional rate |
| --- | ---: | ---: | ---: |
| answerability_false_answer | 41 | 0.3388 | 41 / 45 unanswerable = 0.9111 |
| gold_span_beats_selected_answer | 41 | 0.3388 | 41 / 76 answerable = 0.5395 |
| retrieval_context_miss | 23 | 0.1901 | 23 / 76 answerable = 0.3026 |
| evidence_selection_miss | 12 | 0.0992 | 12 / 76 answerable = 0.1579 |

Answerable dev flow:

```text
answerable rows: 76
gold context absent from top10: 23 / 76 = 0.3026
gold context present in top10: 53 / 76 = 0.6974
context-present evidence selection miss: 12 / 53 = 0.2264
context-present gold span beats selected answer: 41 / 53 = 0.7736
answerable supported-and-cited rows: 0
```

This means retrieval recall still matters, but even when the gold context is in
top10, the current answer pipeline fails to turn it into a supported answer.

## Train-Dev Similarity

The dev bucket rates are close to train for the main failure buckets:

| Bucket | Train rate | Dev rate | Dev - train |
| --- | ---: | ---: | ---: |
| answerability_false_answer | 0.3203 | 0.3388 | +1.85 pp |
| gold_span_beats_selected_answer | 0.3096 | 0.3388 | +2.92 pp |
| retrieval_context_miss | 0.2224 | 0.1901 | -3.23 pp |
| evidence_selection_miss | 0.1192 | 0.0992 | -2.00 pp |

The pattern is therefore not merely a dev-only anomaly.

## Route Concentration

Top dev route failure counts:

| Route | Total | Failures | Failure rate | Dominant failure bucket |
| --- | ---: | ---: | ---: | --- |
| other | 46 | 45 | 0.9783 | gold_span_beats_selected_answer |
| error_or_log | 25 | 25 | 1.0000 | answerability_false_answer |
| install_upgrade_config | 17 | 16 | 0.9412 | answerability_false_answer |
| how_to_or_lookup | 15 | 14 | 0.9333 | gold_span_beats_selected_answer |
| security_bulletin_vulnerability_detail | 14 | 13 | 0.9286 | gold_span_beats_selected_answer |

The highest absolute bucket is `other`, but every visible route has a high
failure rate. The next redesign should therefore target failure mechanisms, not
only one route label.

## Stage105 Candidate Failure Pattern

Stage105 selected `amg_bm25_evidence8_rank3_v1` on train. Stage107 confirms it
was a dev no-op:

```text
selected_train_weighted_target_delta: 0.0
dev_weighted_target_delta: 0.0
dev_changed_answer_count: 0
dev target bucket deltas: all 0
dev metric deltas: all 0.0
```

The candidate family split into three behavior clusters:

| Cluster | Configs | Dev changed answers | Best dev target delta | Train selectable |
| --- | ---: | ---: | ---: | --- |
| train_selectable_noop_or_near_noop | 2 | 0 | 0.00 | true |
| train_nonselectable_low_change | 1 | 4 | -3.10 | false |
| train_nonselectable_high_change | 6 | 695 total | -29.15 | false |

All seven dev-better non-selectable configs failed the train answerable-refusal
guard; four also failed the train gold-citation guard.

## Decision

```text
status: primeqa_hybrid_validation_failure_pattern_analysis_completed
protocol_id: primeqa_hybrid_validation_failure_pattern_analysis_v1
validation_split: dev
dev_failure_count: 117
dev_failure_rate: 0.9669
answerable_failure_rate: 1.0
unanswerable_false_answer_rate: 0.9111
stage105_selected_config_was_dev_noop: true
recommended_next_direction: failure_pattern_driven_train_dev_redesign_protocol
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Stage107 is diagnostic only. It does not justify runtime defaultization and does
not open the final-test gate.

## Guard Checks

All `19 / 19` Stage107 guard checks passed.

## Visualizations

```text
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_dev_failure_bucket_counts.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_train_dev_bucket_rate_drift.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_dev_route_failure_counts.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_dev_answerable_failure_flow.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_stage105_candidate_behavior.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_decision_flags.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_guard_check_status.svg
```

## Validation

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_validation_failure_pattern_analysis.py scripts\analyze_primeqa_hybrid_validation_failure_patterns.py tests\test_primeqa_hybrid_validation_failure_pattern_analysis.py
pytest -q tests\test_primeqa_hybrid_validation_failure_pattern_analysis.py
python scripts\analyze_primeqa_hybrid_validation_failure_patterns.py --user-confirmed-analysis ...
```

Result:

```text
ruff: passed
pytest: 2 passed
Stage107 run: passed
guard checks: 19 / 19 passed
```

Full validation is recorded in `docs\learning_journal.md`.

## Next Step

Stage108 should happen only after user confirmation. It should freeze a
train/dev-only failure-pattern-driven redesign protocol. The next protocol
should target the observed failure mechanisms without selecting from dev, keep
test locked, keep runtime defaults unchanged, and add no fallback strategies.
