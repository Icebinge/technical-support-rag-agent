# PrimeQA Hybrid Retrieval/Index Redesign Comparison

## Scope

Stage 114 runs the frozen Stage113 retrieval/index redesign candidates on the
Stage68 train/dev development splits only.

It uses train grouped cross-validation for selection and reports one dev pass
without dev selection or retuning. It does not load the test split, does not run
final metrics, does not change runtime defaults, and does not add fallback
strategies.

## Command

```text
python scripts\run_primeqa_hybrid_retrieval_index_redesign_comparison.py --user-confirmed-comparison --confirmation-note "user confirmed Stage114 train grouped-CV retrieval/index redesign comparison on 2026-07-16; train selection only; one dev report; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

## Inputs

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113.json
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_train.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_dev.jsonl
data\raw\primeqa_techqa\TechQA\training_and_dev\training_dev_technotes.sections.json
```

## Data Summary

```text
documents: 28482
sections: 216648

train rows: 562
train answerable: 370
train unanswerable: 192

dev rows: 121
dev answerable: 76
dev unanswerable: 45

train grouped-CV folds: 5
fold row counts: 113, 113, 112, 112, 112
raw group values written: false
```

## Baseline

The baseline is the Stage102 verified BM25 top10 answer pipeline.

```text
train-CV gold doc hit: 245 / 370
train-CV gold doc miss: 125 / 370
train-CV gold doc recall@10: 0.6622
train-CV verified F1: 0.2017
train-CV gold citation rate: 0.4958

dev gold doc hit: 53 / 76
dev gold doc miss: 23 / 76
dev gold doc recall@10: 0.6974
dev verified F1: 0.2040
dev gold citation rate: 0.6029
```

## Train-CV Result

No candidate passed train-CV selectability.

```text
selectable configs: 0 / 8
decision status: primeqa_hybrid_retrieval_index_redesign_completed_no_train_cv_selectable_config
recommended next stage: Stage115 stop decision for the frozen Stage113 retrieval/index redesign family
```

Ranking by train-CV objective:

```text
1. evc_special_token_title_heading_boost_v1
   objective delta: -14.0
   retrieval_context_miss delta: -4
   gold doc recall@10 delta: +0.0108
   changed answer rate: 0.7278
   failed guards: evidence_selection_miss, gold_span_beats_selected_answer, changed_answer_rate

2. evc_special_token_exact_boost_v1
   objective delta: -14.0
   retrieval_context_miss delta: -4
   gold doc recall@10 delta: +0.0108
   changed answer rate: 0.1833
   failed guards: answerability_false_answer, evidence_selection_miss

3. thw_title3_heading2_body1_doc_bm25_v1
   objective delta: -10.5
   retrieval_context_miss delta: -3
   gold doc recall@10 delta: +0.0081
   changed answer rate: 0.8327
   failed guards: evidence_selection_miss, gold_span_beats_selected_answer, changed_answer_rate

4. thw_title2_heading2_body1_doc_bm25_v1
   objective delta: -10.5
   retrieval_context_miss delta: -3
   gold doc recall@10 delta: +0.0081
   changed answer rate: 0.7171
   failed guards: evidence_selection_miss, changed_answer_rate

5. slr_section_top3_rrf_doc_rollup_v1
   objective delta: +31.5
   retrieval_context_miss delta: +9
   gold doc recall@10 delta: -0.0244
   changed answer rate: 0.9804

6. thw_title_heading_query_view_rrf_v1
   objective delta: +73.5
   retrieval_context_miss delta: +21
   gold doc recall@10 delta: -0.0568
   changed answer rate: 0.9893

7. slr_heading_section_title_rollup_v1
   objective delta: +80.5
   retrieval_context_miss delta: +23
   gold doc recall@10 delta: -0.0622
   changed answer rate: 0.9875

8. slr_section_top1_doc_rollup_v1
   objective delta: +91.0
   retrieval_context_miss delta: +26
   gold doc recall@10 delta: -0.0703
   changed answer rate: 0.9840
```

## Dev Report

Because no train-CV-selectable candidate exists, Stage114 did not select a
candidate for dev validation. Dev remains report-only and is not used for
selection.

For reference, candidate dev deltas were:

```text
thw_title2_heading2_body1_doc_bm25_v1:
  retrieval_context_miss delta: 0
  recall@10 delta: +0.0000
  F1 delta: +0.0057
  gold citation delta: -0.0114
  changed answer rate: 0.7107

thw_title3_heading2_body1_doc_bm25_v1:
  retrieval_context_miss delta: +1
  recall@10 delta: -0.0132
  F1 delta: +0.0052
  gold citation delta: -0.0335
  changed answer rate: 0.8512

evc_special_token_exact_boost_v1:
  retrieval_context_miss delta: 0
  recall@10 delta: +0.0000
  F1 delta: +0.0041
  gold citation delta: +0.0058
  changed answer rate: 0.2231

evc_special_token_title_heading_boost_v1:
  retrieval_context_miss delta: 0
  recall@10 delta: +0.0000
  F1 delta: +0.0067
  gold citation delta: -0.0196
  changed answer rate: 0.7686
```

The section-rollup and RRF candidates did not improve dev retrieval recall.

## Visualizations

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_train_cv_retrieval_context_miss_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_train_cv_gold_doc_recall_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_train_cv_average_token_f1_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_train_cv_selectability.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_dev_retrieval_context_miss_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_dev_gold_doc_recall_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_decision_flags.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_guard_check_status.svg
```

## Guard Checks

All Stage114 guard checks passed:

```text
16 / 16 passed
```

The experiment itself completed successfully, but the candidate family did not
produce a selectable config.

## Interpretation

Stage114 confirms that the current frozen retrieval/index candidate family is
not safe to promote.

The best train-CV retrieval movement came from special-token candidates:

```text
retrieval_context_miss delta: -4
gold doc recall@10 delta: +0.0108
```

However, the gains were too small and unstable relative to answer-pipeline
risk. The exact special-token boost had an acceptable changed-answer rate but
increased answerability false answers and evidence-selection misses. The
title-heading special-token boost recovered the same number of misses but
changed too many answers and introduced additional downstream bucket risk.

The section-level candidates are not viable in this implementation: they
increased train-CV retrieval_context_miss and sharply changed answer behavior.

## Next Step

Stage115 should record a stop decision for the frozen Stage113 retrieval/index
redesign family. Test remains locked, runtime defaults remain unchanged, and no
fallback strategies are added.
