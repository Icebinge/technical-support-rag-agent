# PrimeQA Hybrid Retrieval Recall Exhaustion Summary

This document records Stage 83.

## Scope

Stage 83 summarizes the full Stage76 retrieval-recall candidate set after the
Stage77-Stage82 train/dev-only runs.

This stage reads only saved public-safe Stage76-Stage82 reports. It does not
load the frozen test split, does not run new retrieval metrics, does not run
final metrics, does not use source `DOC_IDS` as runtime retrieval evidence, and
does not change runtime defaults.

The report stores candidate-level outcomes, guard checks, next-route options,
source report fingerprints, and visualization paths. It does not output raw
question text, raw answer text, document titles, or document body text.

## Command

```text
python scripts\summarize_primeqa_hybrid_retrieval_recall_exhaustion.py ^
  --output artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83.json ^
  --visualization-dir artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals
```

The run completed in `0.002s`.

## Candidate Outcomes

| Candidate | Source stage | Selected ID | Dev hit@10 delta | Top10 net | Outcome |
| --- | --- | --- | ---: | ---: | --- |
| query_view_ablation_full_title_dedup | Stage 77 | full_question_dedup_terms | -0.0395 | -3 | not advanced |
| fielded_title_text_bm25_score_fusion | Stage 78 | fielded_title_0_25_text_1_00 | 0.0000 | 0 | not advanced |
| section_bm25_doc_rollup_train_dev_probe | Stage 79 | section_bm25_max_section_rollup | -0.0527 | -4 | not advanced |
| dense_sparse_rrf_train_dev_probe | Stage 81 | dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix | -0.0132 | -1 | not advanced |
| bm25_k1_b_grid_train_to_dev | Stage 82 | full_document_bm25_baseline | 0.0000 | 0 | not advanced |

Aggregate summary:

```text
allowed_stage76_candidate_count: 5
allowed_candidate_outcome_count: 5
allowed_candidates_completed: true
blocked_candidate_count: 1
runtime_advancing_candidate_count: 0
best_selected_dev_hit10_delta: 0.0
stage76_retrieval_recall_set_exhausted: true
```

The blocked candidate remains blocked:

```text
candidate_id: source_doc_ids_oracle_union_blocked
status: blocked_from_train_dev_experiment
target_miss_count: 148
target_miss_count_by_split:
  dev: 23
  train: 125
```

## Dev-Only Observations

Stage82 found two `b=0.95` BM25 grid configs with dev hit@10 above the baseline:

| Config | Dev hit@10 | Baseline dev hit@10 | Delta |
| --- | ---: | ---: | ---: |
| bm25_grid__k1_1_20__b_0_95 | 0.7105 | 0.6974 | +0.0131 |
| bm25_grid__k1_1_50__b_0_95 | 0.7105 | 0.6974 | +0.0131 |

These configs were not selected by the train-only rule. Using them would be
dev-set selection, so they are recorded only as follow-up evidence and cannot
advance to runtime.

## Guard Checks

All `10 / 10` Stage83 guard checks passed:

```text
source_reports_are_expected_stages: passed
stage76_allowed_candidates_all_accounted_for: passed
source_doc_ids_candidate_remains_blocked: passed
no_allowed_candidate_advanced_to_runtime: passed
all_source_decisions_keep_final_test_locked: passed
all_source_decisions_forbid_test_tuning: passed
all_source_decisions_keep_runtime_defaults_unchanged: passed
stage83_runs_summary_only_no_new_retrieval_metrics: passed
stage83_final_test_metrics_not_run: passed
stage83_default_runtime_policy_unchanged: passed
```

## Visualizations

```text
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals\stage83_candidate_dev_hit10_deltas.svg
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals\stage83_candidate_top10_net_changes.svg
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals\stage83_candidate_advancement_status.svg
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals\stage83_next_route_options.svg
```

Stage83 JSON SHA256:

```text
8A775C08F867C0E6C6DFB2E62606CF04BD53AC449EA0C85B75A674450FA8C7B4
```

Visualization SHA256:

```text
stage83_candidate_advancement_status.svg: C5E8A4B567E6D60ED6A6A3A2AF9260787EF09D06302C1E9CC5AF4CAC2740537E
stage83_candidate_dev_hit10_deltas.svg: FFF299B7E74364832D0FE14FEE6A96E182845EA676268D29D458E3C55C8B4069
stage83_candidate_top10_net_changes.svg: 95528E7642BAEE818491BC91E07E1FEE63908CB55681877425AABCF13FA86BE2
stage83_next_route_options.svg: 7BE04D14943CEE3DF2657E004D33DC163FA974CB58C24043DDA3AB1B080C0049
```

## Validation

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_recall_exhaustion_summary.py scripts\summarize_primeqa_hybrid_retrieval_recall_exhaustion.py tests\test_primeqa_hybrid_retrieval_recall_exhaustion_summary.py: passed
pytest -q tests\test_primeqa_hybrid_retrieval_recall_exhaustion_summary.py: 2 passed
Select-String raw question / answer / document / snippet field patterns over Stage83 JSON: no matches
git check-ignore Stage83 JSON and SVG artifacts: ignored by .gitignore
ruff check .: passed
pytest -q: 215 passed
git diff --check: passed
```

## Decision

```text
status: primeqa_hybrid_retrieval_recall_exhaustion_summary_completed
stage76_allowed_candidates_exhausted: true
runtime_advancing_candidate_count: 0
can_continue_train_dev_development: true
requires_user_confirmation_before_next_route: true
recommended_next_route_option: second_wave_retrieval_candidate_design
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage84 should confirm the next train/dev-only route before any new metrics are
run. The recommended route is `second_wave_retrieval_candidate_design`: aggregate
Stage75 and Stage77-82 changed-case evidence, then design a second-wave
retrieval candidate set. The frozen test split remains locked, final metrics
must not be run, source `DOC_IDS` must not be used as runtime retrieval
evidence, and runtime defaults remain unchanged.
