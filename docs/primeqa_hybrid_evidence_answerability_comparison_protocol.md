# PrimeQA Hybrid Evidence/Answerability Comparison Protocol

This document records Stage 104.

## Scope

Stage 104 freezes the train/dev-only comparison grid for the Stage103
evidence-answerability candidate family.

This stage reads only the saved public-safe Stage103 protocol report. It does
not load train/dev/test split files, does not load corpus documents, does not
run retrieval metrics, does not run answer metrics, does not run final metrics,
does not use oracle document identifiers or gold answers as runtime evidence,
does not add fallback strategies, and does not change runtime defaults.

The protocol was explicitly confirmed for this run:

```text
confirmed: true
confirmation_note: Stage104 user-confirmed comparison-grid freeze after Stage103 protocol
```

## Command

```text
python scripts\freeze_primeqa_hybrid_evidence_answerability_comparison_protocol.py ^
  --user-confirmed-protocol ^
  --confirmation-note "Stage104 user-confirmed comparison-grid freeze after Stage103 protocol"
```

The run completed successfully:

```text
stage104_exit_code=0
```

Console output was written to:

```text
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104.console.txt
```

## Stage103 Premise

```text
decision_status: primeqa_hybrid_evidence_answerability_candidate_protocol_frozen
design_id: evidence_selection_and_answerability_candidate_design_v1
recommended_direction: evidence_answerability_train_dev_candidate_comparison
recommended_execution_order:
1. joint_gate_then_window_candidate_v1
2. evidence_window_reselector_candidate_v1
3. answerability_margin_gate_candidate_v1
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Stage104 only reads:

```text
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103.json
```

## Baseline Reference

```text
baseline_id: stage102_verified_bm25_top10_answer_pipeline
retriever: BM25
bm25_k1: 1.5
bm25_b: 0.75
retrieval_top_k: 10
evidence_selector_name: bm25_sentence
max_candidates_per_document: 3
composition_policy_name: top_k
max_sentences: 3
min_sentence_score: 2.0
verifier_min_evidence_score: 7.0
verifier_max_citation_rank: 3
verifier_min_citations: 1
source_stage: Stage 102
```

## Frozen Grid

The following threshold values are predeclared engineering design parameters.
They are not fitted from train labels, dev labels, test labels, or Stage104
metric results. Stage104 does not run any metrics.

| Config | Candidate | Selector | MCPD | Min evidence | Max citation rank |
| --- | --- | --- | ---: | ---: | ---: |
| amg_bm25_evidence8_rank3_v1 | answerability_margin_gate_candidate_v1 | bm25_sentence | 3 | 8.0 | 3 |
| amg_bm25_evidence9_rank3_v1 | answerability_margin_gate_candidate_v1 | bm25_sentence | 3 | 9.0 | 3 |
| amg_bm25_evidence8_rank2_v1 | answerability_margin_gate_candidate_v1 | bm25_sentence | 3 | 8.0 | 2 |
| ewr_answer_window_mcpd3_evidence7_rank3_v1 | evidence_window_reselector_candidate_v1 | answer_window | 3 | 7.0 | 3 |
| ewr_hybrid_window_mcpd3_evidence7_rank3_v1 | evidence_window_reselector_candidate_v1 | hybrid_window | 3 | 7.0 | 3 |
| ewr_answer_window_mcpd5_evidence7_rank3_v1 | evidence_window_reselector_candidate_v1 | answer_window | 5 | 7.0 | 3 |
| jgw_answer_window_mcpd3_evidence8_rank3_v1 | joint_gate_then_window_candidate_v1 | answer_window | 3 | 8.0 | 3 |
| jgw_hybrid_window_mcpd3_evidence8_rank3_v1 | joint_gate_then_window_candidate_v1 | hybrid_window | 3 | 8.0 | 3 |
| jgw_answer_window_mcpd5_evidence8_rank2_v1 | joint_gate_then_window_candidate_v1 | answer_window | 5 | 8.0 | 2 |

Grid coverage:

```text
answerability_margin_gate_candidate_v1: 3 configs
evidence_window_reselector_candidate_v1: 3 configs
joint_gate_then_window_candidate_v1: 3 configs
total: 9 configs
```

Selector mix:

```text
bm25_sentence
answer_window
hybrid_window
```

## Selection Contract

Stage105 must select thresholds on train only.

Train objective:

```text
Minimize weighted target bucket score:
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

Tie breakers:

```text
1. lower train answerability_false_answer count
2. lower train gold_span_beats_selected_answer count
3. lower train evidence_selection_miss count
4. higher train verified average token F1
5. higher train gold document citation rate
6. lower train changed answer count
7. lexicographic config_id
```

Dev validation:

```text
dev_used_for: single validation of train-selected config
must_report_all_configs_on_dev: true
dev_retuning_allowed: false
dev_selection_allowed: false
test_access_allowed: false
```

## Decision

```text
status: primeqa_hybrid_evidence_answerability_comparison_protocol_frozen
protocol_id: evidence_answerability_candidate_train_dev_comparison_v1
recommended_direction: run_evidence_answerability_train_dev_candidate_comparison
requires_user_confirmation_before_train_dev_metric_run: true
can_continue_train_dev_development: true
can_run_train_dev_candidate_comparison_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_stage: Stage105: after user confirmation, run the frozen train/dev-only evidence-answerability candidate comparison against the Stage102 verified baseline; select thresholds on train only, validate once on dev, keep test locked, and keep runtime defaults unchanged.
```

## Guard Checks

All `31 / 31` guard checks passed:

```text
stage103_source_is_expected_stage: passed
user_confirmed_stage104_protocol: passed
stage103_design_id_matches: passed
stage103_protocol_is_frozen: passed
stage103_recommends_candidate_comparison: passed
stage103_execution_order_matches_protocol: passed
stage103_candidates_present: passed
stage103_allows_train_dev_comparison_after_confirmation: passed
stage103_final_test_gate_locked: passed
stage103_forbids_test_tuning: passed
stage103_fallback_disabled: passed
stage103_runtime_defaults_unchanged: passed
stage103_contract_id_matches_stage104_protocol: passed
protocol_id_is_fixed: passed
protocol_status_requires_confirmation_before_metric_run: passed
candidate_config_grid_has_nine_configs: passed
candidate_config_ids_are_unique: passed
candidate_grid_covers_all_stage103_candidates: passed
candidate_grid_has_three_configs_per_candidate: passed
candidate_grid_uses_allowed_selectors: passed
candidate_grid_keeps_composition_top_k: passed
candidate_grid_forbids_dev_threshold_tuning: passed
candidate_grid_forbids_test_access: passed
candidate_grid_forbids_runtime_default_change: passed
grid_derivation_uses_no_metric_labels: passed
train_selection_rule_is_train_only: passed
train_selection_guards_are_frozen: passed
dev_validation_forbids_retuning: passed
public_output_contract_has_no_forbidden_fields: passed
stage104_exclusions_lock_test_runtime_fallback: passed
fallback_policy_disabled: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_config_counts_by_candidate.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_config_min_evidence_scores.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_config_max_citation_ranks.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_selector_mix.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_train_selection_guard_thresholds.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_protocol_decision_flags.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_guard_check_status.svg
```

Stage104 JSON SHA256:

```text
F50A7054085BB8ACF852A89070BAB65551F58CA5BFA8978BF84B340BB660C974
```

Visualization SHA256:

```text
stage104_config_counts_by_candidate.svg: 9E610C00CFDB2C9D26A918AC25438D05D3F0BF69550469AC7975334A086A6F0A
stage104_config_max_citation_ranks.svg: 61B07DDADB145622387D26F4CD3AB05D1512B71FCD3E16480A98E4AE26595F95
stage104_config_min_evidence_scores.svg: 24BB9CD321C88F1A918294C21EEC50ED530E01F391597556674185A5049FF077
stage104_guard_check_status.svg: 00EA679B04F15AA7BCF3E5A72011FDC28891AC1C97A9A4BF9551318D2D661476
stage104_protocol_decision_flags.svg: 2F95E29F12CE71BA541427FB9A08889478D93D2C1304EDA23A4130B231D7B2E0
stage104_selector_mix.svg: 42A54C7D0C60AE8662FE566D214540152BBD02D18411DF035D23BAB03F5C8299
stage104_train_selection_guard_thresholds.svg: 13E5743BB7156355AB2F1E88D9733BDC3340F3322BBA9B8D02D27CB7EC8AA0DA
```

## Validation

Targeted validation:

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_comparison_protocol.py scripts\freeze_primeqa_hybrid_evidence_answerability_comparison_protocol.py tests\test_primeqa_hybrid_evidence_answerability_comparison_protocol.py
pytest -q tests\test_primeqa_hybrid_evidence_answerability_comparison_protocol.py
python scripts\freeze_primeqa_hybrid_evidence_answerability_comparison_protocol.py ...: stage104_exit_code=0
```

Result:

```text
ruff: passed
pytest: 5 passed
Stage104 run: passed
guard checks: 31 / 31 passed
```

Full validation:

```text
ruff check .: passed
pytest -q: 284 passed
git diff --check: passed
```

Artifact safety checks:

```text
Allowed output fields intersect forbidden fields: []
Source file recorded by Stage104: artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103.json
No split-file or corpus-document paths are recorded in Stage104 source_files.
No private fixture or split/corpus path patterns matched in the Stage104 JSON scan.
```

## Conclusion

Stage104 can advance to Stage105 after user confirmation. Stage105 should run
the frozen train/dev-only evidence-answerability candidate comparison against
the Stage102 verified baseline, select thresholds on train only, validate once
on dev, keep test locked, keep fallback strategies disabled, and keep runtime
defaults unchanged.
