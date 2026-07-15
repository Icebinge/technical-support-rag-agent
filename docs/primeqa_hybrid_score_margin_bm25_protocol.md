# PrimeQA Hybrid Score-Margin BM25 Protocol

This document records Stage 94.

## Scope

Stage 94 confirms and freezes the train/dev-only protocol for
`score_margin_bm25_normalization_gate_design` after Stage93 stopped the
section-signal guarded expansion route.

This is a protocol-freeze checkpoint. It reads only the public-safe Stage84 and
Stage93 reports, does not load train/dev/test split files, does not run
retrieval metrics, does not run final metrics, does not use source `DOC_IDS` as
runtime retrieval evidence, does not choose runtime thresholds from dev-only
observations, and does not change runtime defaults.

The candidate was explicitly confirmed for this run:

```text
confirmed: true
confirmed_candidate_id: score_margin_bm25_normalization_gate_design
confirmation_note: user confirmed Stage94 score-margin BM25 protocol freeze in current turn
```

## Command

```text
python scripts\freeze_primeqa_hybrid_score_margin_bm25_protocol.py ^
  --user-confirmed-candidate ^
  --confirmed-candidate-id score_margin_bm25_normalization_gate_design ^
  --confirmation-note "user confirmed Stage94 score-margin BM25 protocol freeze in current turn" ^
  --output artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94.json ^
  --visualization-dir artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals
```

The run completed in `0.000s`.

## Frozen Protocol

```text
protocol_id: score_margin_bm25_normalization_gate_train_dev_v1
candidate_id: score_margin_bm25_normalization_gate_design
protocol_status: frozen_requires_user_confirmation_before_metric_run
development_splits: train, dev
forbidden_final_splits: test
```

Stage84 target metric contract:

```text
primary: train-selected rule must improve dev hit@10
secondary: rank 11-50 near misses should decrease
guard: dev-only b=0.95 observations cannot select a runtime rule
```

Historical Stage82 policy:

```text
source_signal: bm25_k1_b_grid
dev_only_b095_observation_can_select_runtime_rule: false
allowed_use: motivation for predeclaring train-selected score-margin and document-length gates
```

## Candidate Config Grid

```text
smbn_rank11_20_long_doc_b095_margin_v1
  challenger BM25: k1=1.5, b=0.95
  eligible baseline rank: 11-20
  challenger rank max: 10
  max score margin to rank10: 0.05
  length gate: long_document_only, min length ratio 1.20
  max top10 promotions per query: 1

smbn_rank21_50_long_doc_b095_high_confidence_v1
  challenger BM25: k1=1.5, b=0.95
  eligible baseline rank: 21-50
  challenger rank max: 15
  max score margin to rank10: 0.03
  length gate: long_document_only, min length ratio 1.50
  max top10 promotions per query: 1

smbn_rank11_20_short_doc_b055_margin_v1
  challenger BM25: k1=1.5, b=0.55
  eligible baseline rank: 11-20
  challenger rank max: 10
  max score margin to rank10: 0.04
  length gate: short_document_only, max length ratio 0.85
  max top10 promotions per query: 1

smbn_rank11_50_dual_length_band_margin_v1
  challenger BM25: k1=1.5, b=0.55 for short docs and b=0.95 for long docs
  eligible baseline rank: 11-50
  challenger rank max: 12
  max score margin to rank10: 0.02
  length gate: outside_length_band_short_or_long
  branch rule: use b=0.55 when length ratio <= 0.75; use b=0.95 when length ratio >= 1.35
  max top10 promotions per query: 1
```

Train selection rule:

```text
Select the candidate config on train only by hit@10, then fewer rank 11-50
near misses, fewer top10 regressions, hit@5, hit@1, MRR@10, lower top10
promotion budget, then config_id. Dev is validation only.
```

## Runtime Evidence Contract

Allowed runtime feature groups:

```text
baseline_bm25_features:
  baseline_bm25_rank
  baseline_bm25_score
  baseline_score_margin_to_rank10
  baseline_score_margin_to_previous

challenger_bm25_features:
  challenger_bm25_rank
  challenger_bm25_score
  challenger_score_margin_to_rank10
  normalization_view_id

document_length_features:
  document_token_count
  average_document_token_count
  document_length_ratio_to_average
  document_length_bucket

gate_state_features:
  eligible_baseline_rank_bucket
  challenger_rank_bucket
  top10_promotion_budget_remaining
  promotion_reason_code
```

Explicit exclusions:

```text
Do not use source DOC_IDS as runtime retrieval evidence.
Do not use answer document IDs or gold ranks as runtime features.
Do not choose candidate configs from dev-only performance.
Do not choose candidate configs from Stage82 dev-only b=0.95 observations.
Do not load or evaluate the frozen test split.
Do not write raw question text, answer text, document titles, document body text, query terms, or matched token strings to the report.
Do not change runtime defaults in this stage.
Do not add behavior outside the predeclared score-margin config grid.
```

## Decision

```text
status: primeqa_hybrid_score_margin_bm25_protocol_frozen
protocol_id: score_margin_bm25_normalization_gate_train_dev_v1
candidate_id: score_margin_bm25_normalization_gate_design
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_run_train_dev_metrics_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

Stage94 only freezes the protocol. It does not run metrics and does not make the
score-margin rule a runtime default.

## Guard Checks

All `31 / 31` guard checks passed:

```text
source_stage84_report_is_stage84: passed
source_stage93_report_is_stage93: passed
user_confirmed_score_margin_protocol: passed
stage93_stopped_section_signal_route: passed
confirmed_candidate_matches_stage93_next_candidate: passed
stage93_next_candidate_summary_matches: passed
stage93_requires_confirmation_before_next_protocol: passed
stage93_final_test_metrics_locked: passed
stage93_final_test_gate_closed: passed
stage93_forbids_test_tuning: passed
stage93_runtime_default_unchanged: passed
stage84_final_test_metrics_locked: passed
stage84_forbids_test_tuning: passed
stage84_runtime_default_unchanged: passed
stage84_candidate_is_recommended_for_protocol_design: passed
stage84_candidate_contract_requires_train_selected_dev_hit10_gain: passed
stage84_candidate_contract_requires_rank_11_to_50_reduction: passed
stage84_candidate_guard_blocks_dev_only_b095_selection: passed
protocol_id_is_fixed: passed
candidate_config_grid_is_predeclared: passed
candidate_config_grid_contains_length_and_margin_gates: passed
train_selection_rule_forbids_dev_and_test_selection: passed
historical_stage82_signal_is_motivation_only: passed
score_margin_feature_contract_uses_runtime_scores_only: passed
source_doc_ids_forbidden_in_runtime_features: passed
answer_doc_ids_forbidden_in_runtime_features: passed
report_fields_are_public_safe: passed
source_doc_ids_oracle_blocked_candidate_not_selected: passed
stage94_freezes_protocol_without_metrics: passed
stage94_final_test_metrics_not_run: passed
stage94_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_config_b_values.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_margin_thresholds.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_length_thresholds.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_feature_group_counts.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_protocol_decision_flags.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_guard_check_status.svg
```

Stage94 JSON SHA256:

```text
FB6E8B0E8EDBA4BBB8C9DD6E684F4D807AE26CD91A658D3B0AA1F694B202F2E8
```

Visualization SHA256:

```text
stage94_score_margin_bm25_config_b_values.svg: 26849CE3714FD3FF04401D0127546E9D476244E80D8FBD1C537045724A0570FD
stage94_score_margin_bm25_feature_group_counts.svg: B655D1A429489F430647550168428BB8B2D27551B72613C9F1AB2DE13F1B5187
stage94_score_margin_bm25_guard_check_status.svg: 724AF26B2A06F6AB26E93C1CAA839CD058CE5CAB962528028355EB80CB56E6E5
stage94_score_margin_bm25_length_thresholds.svg: 8F32B109D1BC082A314AE2EA76AB58B6139CB94F0BC9B77D0A160E743B1DC09C
stage94_score_margin_bm25_margin_thresholds.svg: 660DEFEF47832B9768CC37212C4E5950E9FA95384CCCED601F7BFE4CADE18A2F
stage94_score_margin_bm25_protocol_decision_flags.svg: 2F6DE073A1F41B240BF4890EFDD5713A103A011088F581920B4385BBE3093B72
```

## Validation

Completed local validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_score_margin_bm25_protocol.py scripts\freeze_primeqa_hybrid_score_margin_bm25_protocol.py tests\test_primeqa_hybrid_score_margin_bm25_protocol.py: passed
pytest -q tests\test_primeqa_hybrid_score_margin_bm25_protocol.py: 3 passed
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage94 JSON: no matches
git check-ignore Stage94 JSON and SVG artifacts: ignored by .gitignore
```

Full repository validation:

```text
ruff check .: passed
pytest -q: 247 passed
git diff --check: passed
```

## Next Step

Stage95 ran the frozen train/dev-only score-margin BM25 normalization gate
comparison after user confirmation. The train-selected config was
`smbn_rank11_20_long_doc_b095_margin_v1`, but it had dev hit@10 delta `0.0000`
and dev rank 11-50 count delta `0`, so the route does not advance.

The current next step is Stage96: stop score-margin BM25 normalization as a
retrieval-recall route unless a new train/dev-only protocol is explicitly
confirmed. The frozen test split remains locked, final metrics must not be run,
source `DOC_IDS` must not be used as runtime retrieval evidence, dev-only
observations must not select runtime rules, and runtime defaults remain
unchanged.
