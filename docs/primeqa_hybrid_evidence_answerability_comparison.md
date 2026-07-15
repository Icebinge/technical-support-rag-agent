# PrimeQA Hybrid Evidence/Answerability Comparison

This document records Stage 105.

## Scope

Stage 105 runs the frozen train/dev-only evidence-answerability candidate
comparison from Stage104. It compares all 9 frozen candidate configs against the
Stage102 verified BM25 top10 answer-pipeline baseline.

This stage loads only the frozen Stage68 train/dev split files and the PrimeQA
training/dev document file. It does not load the frozen test split, does not run
final metrics, does not use dev for threshold selection, does not use source
`DOC_IDS` as runtime retrieval evidence, does not write raw question, answer,
document, token, or document-identifier fields in changed-case samples, does
not add fallback strategies, and does not change runtime defaults.

The comparison was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: user confirmed Stage105 train/dev candidate comparison on 2026-07-15; test locked; runtime defaults unchanged; no fallback strategies
```

## Command

```text
python scripts\run_primeqa_hybrid_evidence_answerability_comparison.py ^
  --user-confirmed-comparison ^
  --confirmation-note "user confirmed Stage105 train/dev candidate comparison on 2026-07-15; test locked; runtime defaults unchanged; no fallback strategies"
```

Console output was captured in:

```text
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105.console.txt
```

The JSON report was written to:

```text
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105.json
```

The run completed successfully in `368.557s` according to the Stage105 timing
block.

## Loaded Data

```text
documents: 28482
train rows: 562
train answerable rows: 370
train unanswerable rows: 192
dev rows: 121
dev answerable rows: 76
dev unanswerable rows: 45
test_split_loaded: false
```

## Baseline

```text
baseline_config_id: stage102_verified_bm25_top10_answer_pipeline
train weighted target score: 645.2
dev weighted target score: 143.4

train verified:
  answerable_refusal_rate: 0.0459
  unanswerable_refusal_rate: 0.0625
  gold_doc_citation_rate: 0.4958
  average_token_f1: 0.2017

dev verified:
  answerable_refusal_rate: 0.1053
  unanswerable_refusal_rate: 0.0889
  gold_doc_citation_rate: 0.6029
  average_token_f1: 0.2040
```

Stage105 guard checks confirmed that the recomputed baseline bucket counts and
verified metrics matched the saved Stage102 report.

## Train Selection

Train objective:

```text
1.55 * answerability_false_answer
+ 1.45 * gold_span_beats_selected_answer
+ 1.70 * evidence_selection_miss
```

Train selectability guards:

```text
max_train_answerable_refusal_rate_delta: 0.05
max_train_average_token_f1_drop: 0.01
max_train_gold_doc_citation_rate_drop: 0.03
```

Train ranking:

| Rank | Config | Train delta | Selectable | Changed answers |
| ---: | --- | ---: | --- | ---: |
| 1 | amg_bm25_evidence8_rank3_v1 | 0.00 | true | 1 |
| 2 | amg_bm25_evidence9_rank3_v1 | 0.00 | true | 1 |
| 3 | jgw_answer_window_mcpd5_evidence8_rank2_v1 | -133.90 | false | 542 |
| 4 | ewr_answer_window_mcpd3_evidence7_rank3_v1 | -89.25 | false | 546 |
| 5 | ewr_answer_window_mcpd5_evidence7_rank3_v1 | -89.25 | false | 546 |
| 6 | jgw_answer_window_mcpd3_evidence8_rank3_v1 | -89.25 | false | 546 |
| 7 | amg_bm25_evidence8_rank2_v1 | -29.25 | false | 37 |
| 8 | ewr_hybrid_window_mcpd3_evidence7_rank3_v1 | -16.60 | false | 550 |
| 9 | jgw_hybrid_window_mcpd3_evidence8_rank3_v1 | -16.60 | false | 550 |

Train-selected config:

```text
selected_config_id: amg_bm25_evidence8_rank3_v1
selected_candidate_id: answerability_margin_gate_candidate_v1
selected_train_weighted_target_delta: 0.0
selectable_config_count: 2 / 9
```

Important interpretation: the configs with large negative train target deltas
were not selectable because they violated the pre-frozen train guards. Most of
the strongest target-score improvements came with much higher answerable refusal
or citation degradation, so they cannot be advanced by this protocol.

## Dev Validation

Dev is validation only. It was not used for config selection or threshold
tuning.

Train-selected config on dev:

```text
selected_config_id: amg_bm25_evidence8_rank3_v1
dev_weighted_target_score: 143.4
dev_weighted_target_delta: 0.0
dev_changed_answer_count: 0
dev_validation_passed: false
```

Dev target bucket deltas:

```text
answerability_false_answer: 0
gold_span_beats_selected_answer: 0
evidence_selection_miss: 0
```

Dev metric deltas:

```text
answerable_refusal_rate: 0.0
unanswerable_refusal_rate: 0.0
gold_doc_citation_rate: 0.0
average_token_f1: 0.0
```

Some non-selectable configs had better dev weighted target deltas, including
`jgw_answer_window_mcpd5_evidence8_rank2_v1` at `-29.15`, but those configs were
already blocked by train selectability guards and therefore cannot be selected
from dev.

## Decision

```text
status: primeqa_hybrid_evidence_answerability_comparison_completed_dev_guard_failed
selected_config_id: amg_bm25_evidence8_rank3_v1
selected_candidate_id: answerability_margin_gate_candidate_v1
selectable_config_count: 2
dev_validation_passed: false
dev_weighted_target_delta: 0.0
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_direction: evidence_answerability_stop_decision
recommended_next_stage: Stage106: record a stop or redesign decision because the train-selected config did not pass dev validation; keep test locked and runtime defaults unchanged.
```

Stage105 therefore does not justify a runtime/default change and does not open
the final test gate.

## Guard Checks

All `29 / 29` Stage105 guard checks passed.

Key checks:

```text
user_confirmed_stage105_comparison: passed
only_train_dev_splits_loaded: passed
test_split_not_loaded: passed
all_stage104_configs_were_run: passed
baseline_bucket_counts_match_stage102: passed
baseline_verified_metrics_match_stage102: passed
train_selection_uses_train_split: passed
dev_validation_not_used_for_selection: passed
stage105_final_test_metrics_not_run: passed
stage105_default_runtime_policy_unchanged: passed
stage105_fallback_strategies_not_added: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_train_weighted_target_scores.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_dev_weighted_target_scores.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_train_target_score_deltas.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_dev_target_score_deltas.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_train_selectability_guards.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_changed_answer_counts.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_guard_check_status.svg
```

## What I Learned

- A lower weighted target score is not enough for advancement; train
  selectability guards are doing real work here and blocked the aggressive
  answer-window configs.
- The two train-selectable AMG configs were effectively no-op candidates for the
  target buckets. They changed only one train answer and zero dev answers, so
  dev validation correctly failed the improvement check.
- Dev contained tempting improvements among non-selectable configs, but the
  protocol must not select from dev. The honest conclusion is stop or redesign,
  not runtime adoption.
- The next useful work is not final-test evaluation. It is a Stage106
  stop/redesign decision that preserves test lock and runtime defaults.

## Next Step

Stage106: record a stop or redesign decision for the Stage103/104
evidence-answerability candidate family. The decision should explain why the
train-selected config failed dev validation, why non-selectable configs cannot
be chosen from dev, and what redesign direction, if any, remains train/dev-only
without fallback strategies or runtime default changes.
