# Evaluation Strategy

This document records the current evaluation strategy after Stage 80.

The active route is now the project-owned PrimeQA/TechQA hybrid split
`primeqa_hybrid_stage68_v1`. Stage 68 froze local split artifacts, and Stage 69
rebuilt PrimeQA-compatible question files plus train/dev candidate artifacts.
Stage 70 ran train/dev development baselines and candidate artifact checks.
Stage 71 ran train-only candidate-reranker cross-validation and train-to-dev
guarded policy validation. Stage 72 reviewed the candidate-reranker dev changed
cases and generated visualization artifacts. Stage 73 ran a train/dev-only
top10 answer proxy diagnostic. Stage 74 stopped the current reranker-policy
development route as non-actionable for now. Stage 75 analyzed BM25 top10
misses and showed that retrieval recall is now the blocking issue. Stage 76
designed train/dev-only retrieval-recall candidate experiments from those miss
drivers. Stage 77 ran the first candidate, query-view ablation, and found that
it underperforms the full-question BM25 baseline. Stage 78 ran the second
candidate, fielded title/text BM25 score fusion, and found no dev hit@10 gain.
Stage 79 ran the third candidate, section BM25 max-section document rollup, and
found a dev hit@10 regression. Stage 80 checked dense+sparse RRF feasibility
and found two compatible local dense caches, but requires user confirmation
before a train/dev run. The frozen test split remains locked for future final
evaluation.

## Current Facts

- Stage 51 remains a non-default candidate policy:
  `candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker`.
- PrimeQA train/dev experiments through Stage 51 are development evidence, not
  final held-out evidence.
- Stage 53 blocked NVIDIA TechQA-RAG-Eval `train.json` as an independent
  held-out source:
  - NVIDIA rows: 910;
  - exact overlap questions against PrimeQA train/dev: 910;
  - exact overlap pairs: 974;
  - unhandled overlap questions: 910.
- Stage 57 froze an MSQA project evaluation split with 3,301 rows and 0 exact or
  near-duplicate overlaps against PrimeQA train/dev at token Jaccard `0.9`.
- Stage 58 recorded MSQA answer-source baselines, but they are not
  PrimeQA-style document-citation metrics.
- Stage 64 ran the capped MSQA Stage51 adapter comparison and showed F1
  regression.
- Stage 65 reviewed changed cases and blocked defaultization from MSQA adapter
  evidence:

```text
consistency_checks_passed: true
changed_answer_count: 719
top3_regression_count: 57
top3_improvement_count: 20
citation_gained_count: 3
citation_lost_count: 0
decision: msqa_stage51_changed_case_review_blocks_defaultization
```

- Stage 66 searched for another external dataset and recommended HQA-Data only
  as a schema-probe candidate. It did not download HQA, run HQA metrics, or
  change runtime defaults.
- After discussing the final document-style RAG target, the user chose to rebuild
  a project-owned PrimeQA/TechQA split instead of continuing with a new external
  dataset.
- Stage 67 planned that split as a dry run from local PrimeQA/TechQA files.
- Stage 68 froze the split as `primeqa_hybrid_stage68_v1`.
- Stage 69 rebuilt train/dev candidate artifacts from the frozen split without
  using test rows.
- Stage 70 reran BM25 development baselines on train/dev and audited the Stage
  69 candidate artifact:

```text
train evaluated questions: 370
train hit@1: 0.4243
train hit@5: 0.6054
train hit@10: 0.6622
train MRR: 0.5023

dev evaluated questions: 76
dev hit@1: 0.4342
dev hit@5: 0.6579
dev hit@10: 0.6974
dev MRR: 0.5331

candidate rows: 5993
candidate rows with test split: 0
final test metrics: not run
default runtime policy: unchanged
```

- Stage 72 reviewed the Stage 71 train/dev changed cases and kept the final test
  gate closed:

```text
best dev top3 policy: logistic_best_candidate / candidate_score_gte_60
best dev top3 delta: +0.0004
best dev top3 regressions: 0
best dev top3 gold citation delta: +0

logistic candidate_score_gte_60 vs stage36_main changed cases: 4 / 76
logistic candidate_score_gte_60 better/tied/worse vs stage36_main: 1 / 2 / 1

ridge candidate_score_gte_60 vs stage36_main changed cases: 5 / 76
ridge candidate_score_gte_60 better/tied/worse vs stage36_main: 1 / 2 / 2

can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
default_runtime_policy: unchanged
```

- Stage 73 ran a top10 answer proxy diagnostic on train/dev only:

```text
train-only CV top10:
  logistic stage36_main delta: +0.0000, regressions: 0
  logistic candidate_score_gte_60 delta: +0.0000, regressions: 0
  ridge stage36_main delta: +0.0000, regressions: 0
  ridge candidate_score_gte_60 delta: +0.0000, regressions: 0

dev holdout top10:
  logistic stage36_main delta: +0.0000, regressions: 0
  logistic candidate_score_gte_60 delta: +0.0000, regressions: 0
  ridge stage36_main delta: +0.0000, regressions: 0
  ridge candidate_score_gte_60 delta: +0.0000, regressions: 0

candidate rows with test split: 0
final test metrics: not run
default runtime policy: unchanged
```

- Stage 74 stopped the current candidate-reranker policy route:

```text
status: candidate_reranker_policy_route_stopped_as_non_actionable
reason: top3 dev signal is only +0.0004 and top10 train/dev signal is +0.0000
current_reranker_policy_defaultization: blocked
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

- Stage 75 analyzed BM25 top10 misses on train/dev only:

```text
train evaluated questions: 370
train hit@10: 0.6622
train miss count: 125
train miss rate: 0.3378

dev evaluated questions: 76
dev hit@10: 0.6974
dev miss count: 23
dev miss rate: 0.3026

cross split evaluated questions: 446
cross split hit@10: 0.6682
cross split miss count: 148
cross split miss rate: 0.3318

cross split gold_doc_not_found_within_top50: 110
cross split gold_doc_rank_21_to_50: 24
cross split gold_doc_rank_11_to_20: 14
final test metrics: not run
default runtime policy: unchanged
```

- Stage 76 designed retrieval-recall candidates on top of the public-safe
  Stage75 report:

```text
recommended execution order:
  1. query_view_ablation_full_title_dedup
  2. fielded_title_text_bm25_score_fusion
  3. section_bm25_doc_rollup_train_dev_probe
  4. dense_sparse_rrf_train_dev_probe
  5. bm25_k1_b_grid_train_to_dev

query_view_ablation_full_title_dedup:
  priority score: 196
  target misses: 143
  dev targets: 22

fielded_title_text_bm25_score_fusion:
  priority score: 195
  target misses: 143
  dev targets: 23

section_bm25_doc_rollup_train_dev_probe:
  priority score: 163
  target misses: 119
  dev targets: 17

source_doc_ids_oracle_union_blocked:
  blocked target misses: 148
  reason: source DOC_IDS are dataset metadata, not runtime user-query evidence

final test metrics: not run
default runtime policy: unchanged
```

- Stage 77 ran query-view ablation on train/dev only:

```text
train full_question_baseline hit@10: 0.6622
train title_only hit@10: 0.6054
train full_question_dedup_terms hit@10: 0.6432

dev full_question_baseline hit@10: 0.6974
dev title_only hit@10: 0.6184
dev full_question_dedup_terms hit@10: 0.6579

train-selected challenger: full_question_dedup_terms
selected challenger dev hit@10 delta: -0.0395
selected challenger dev top10 improvements/regressions: 1 / 4

decision: query_view_ablation_does_not_advance
final test metrics: not run
default runtime policy: unchanged
```

- Stage 78 ran fielded title/text BM25 score fusion on train/dev only:

```text
train full_document_bm25_baseline hit@10: 0.6622
train fielded_title_0_25_text_1_00 hit@10: 0.6378
train fielded_title_0_25_text_1_00 hit@10 delta: -0.0244

dev full_document_bm25_baseline hit@10: 0.6974
dev fielded_title_0_25_text_1_00 hit@10: 0.6974
dev fielded_title_0_25_text_1_00 hit@10 delta: +0.0000
dev fielded_title_0_25_text_1_00 hit@1 delta: +0.0395
dev fielded_title_0_25_text_1_00 MRR delta: +0.0219
dev top10 improvements/regressions: 1 / 1

train-selected challenger: fielded_title_0_25_text_1_00
decision: fielded_bm25_fusion_does_not_advance
final test metrics: not run
default runtime policy: unchanged
```

- Stage 79 ran section BM25 max-section document rollup on train/dev only:

```text
section index documents: 28482
section index sections: 216648
section rollup: max_section_score_per_parent_document

train full_document_bm25_baseline hit@10: 0.6622
train section_bm25_max_section_rollup hit@10: 0.5919
train hit@10 delta: -0.0703
train top10 improvements/regressions: 8 / 34
train not_found@50 delta: +1

dev full_document_bm25_baseline hit@10: 0.6974
dev section_bm25_max_section_rollup hit@10: 0.6447
dev hit@10 delta: -0.0527
dev top10 improvements/regressions: 1 / 5
dev not_found@50 delta: +1

decision: section_bm25_doc_rollup_does_not_advance
final test metrics: not run
default runtime policy: unchanged
```

- Stage 80 checked dense+sparse RRF feasibility without running train/dev
  metrics:

```text
required cached-RRF packages: available
existing dense/hybrid code: available
FAISS: not installed and not required for existing NumPy path
compatible local dense cache count: 2

local cache 1:
  model: intfloat/e5-small-v2
  document_text_max_chars: 512
  document_prefix: passage:
  embedding_shape: 28482 x 384
  can_run_without_model_download: true

local cache 2:
  model: sentence-transformers/all-MiniLM-L6-v2
  document_text_max_chars: 1600
  document_prefix: empty
  embedding_shape: 28482 x 384
  can_run_without_model_download: true

train/dev metrics: not run
final test metrics: not run
default runtime policy: unchanged
requires_user_confirmation_before_train_dev_run: true
```

- Stage 71 ran train-only candidate-reranker grouped CV and train-to-dev guarded
  policy validation for both `logistic_best_candidate` and
  `ridge_candidate_token_f1`:

```text
train-only CV best model: ridge_candidate_token_f1
ridge train-CV selected F1: 0.2652
ridge train-CV delta: +0.0383
logistic train-CV selected F1: 0.2523
logistic train-CV delta: +0.0254

logistic dev top3 best policy: candidate_score_gte_60
logistic dev top3 best delta: +0.0004
logistic dev top3 best regressions: 0

ridge dev top3 best policy: stage36_main
ridge dev top3 best delta: +0.0003
ridge dev top3 best regressions: 1

candidate rows with test split: 0
final test metrics: not run
default runtime policy: unchanged
```

## Rejected Path

Do not use `data/raw/nvidia_techqa_rag_eval/train.json` as the current held-out
defaultization test. It has complete normalized question overlap with PrimeQA
train/dev, so any quality metric reported as held-out would be misleading.

## Chosen Path

### Project-Owned PrimeQA/TechQA Hybrid Split

Status: Stage 80 completed dense+sparse RRF feasibility and found local cached
dense options, but no train/dev dense+sparse run has been approved yet; final
metrics not run.

This route preserves the final target: document-style RAG over TechQA technotes.
It accepts that old Stage 31-66 model-selection evidence cannot be treated as
final-test evidence once the split boundary is rebuilt.

The Stage 67 route used:

```text
1A: fully isolate 10% of answer documents into a document-disjoint test subtype
2A: split the remaining grouped rows into 70% train, 15% dev, 15% random test
3A: include PrimeQA validation_reference rows in the planning pool
```

The grouping rule is:

```text
normalized_question + answer_doc_id
normalized_question + UNANSWERABLE
```

The strict document-disjoint rule is:

```text
If any row in a group has candidate DOC_IDS intersecting a selected answer
document, the whole group goes to test/document_disjoint.
```

Stage 67/68 split result:

```text
input rows: 930
input groups: 889
duplicate groups: 40
answerable rows: 621
unanswerable rows: 309
unique answer docs: 496
selected answer docs for strict isolation: 50
document-disjoint rows: 126
document-disjoint groups: 121
candidate-intersection-only document-disjoint groups: 54

train rows: 562
dev rows: 121
test rows: 247
test/document_disjoint rows: 126
test/group_random_test rows: 121
```

Stage 67 leakage checks preserved by Stage 68:

```text
normalized_question_answer_doc_groups_do_not_cross_splits: passed
selected_document_answer_docs_only_in_document_disjoint_test: passed
selected_document_candidate_doc_ids_only_in_document_disjoint_test: passed
```

Stage 68 decision:

```text
split_name: primeqa_hybrid_stage68_v1
protocol_version: primeqa_hybrid_split_v1
status: primeqa_hybrid_split_frozen_for_rebuild
split_files_finalized: true
can_run_final_metrics_now: false
can_rebuild_training_and_dev_artifacts_next: true
default_runtime_policy: unchanged
```

Required next step:

Stage 81 should run only after confirming the dense model/cache protocol. The
recommended option is `compare_existing_cached_dense_models`, which compares the
two existing local dense caches on train/dev only, selects by train, validates
on dev, does not download models, and does not use test for evaluation or
tuning.

## Parked Paths

### HQA-Data External Probe

Status: parked after the user chose the PrimeQA/TechQA hybrid split route.

Stage 66 recommended HQA-Data from Ubuntu Dialogue Corpus for a possible schema
probe because the public pages expose CSV/JSON, context/span structure, and a CC
BY 4.0 license. The same stage also recorded that HQA questions and answers are
generated from dialogue contexts, not natural user questions paired with human
accepted answers.

HQA was not downloaded and is not approved as final evaluation evidence.

### MSQA External Adapter Evidence

Status: parked as external-risk evidence.

MSQA remains useful for understanding cross-dataset adapter risk, but Stage 64
and Stage 65 blocked Stage51 defaultization from that path. Do not run more MSQA
Stage51 comparisons without a new frozen protocol and explicit user approval.

### Freeze Without Defaultization

Status: available but not chosen.

This would keep Stage 51 as a documented non-default research result and keep
top-k as the default runtime. It cannot support a defaultization decision.

## Current Decision Boundary

The PrimeQA/TechQA hybrid split route is selected, Stage 68 froze local split
JSONL files, Stage 69 rebuilt train/dev candidate artifacts, and Stage 70
completed train/dev development checks. Stage 71 completed candidate-reranker
development on train/dev. Stage 72 completed changed-case review on dev only.
Stage 73 completed a train/dev-only top10 diagnostic. Stage 74 stopped the
current candidate-reranker policy route. Stage 75 completed BM25 top10 miss
analysis and identified retrieval recall as the next blocking issue. Stage 76
designed allowed train/dev retrieval-recall candidates and blocked source
`DOC_IDS` oracle union as non-deployable. Stage 77 completed query-view ablation
and did not advance that route because both challenger views underperformed the
baseline. Stage 78 completed fielded title/text BM25 score fusion and did not
advance that route because the train-selected challenger produced no dev hit@10
gain. Stage 79 completed section BM25 max-section document rollup and did not
advance that route because dev hit@10 regressed. Until a future stage explicitly
opens final evaluation, and until Stage81 protocol is confirmed:

- do not run final metrics;
- do not change the default runtime;
- do not defaultize the current candidate-reranker policy;
- do not continue the current reranker-policy route without a new user-confirmed
  train/dev-only plan;
- do not use the frozen test split while running Stage 81 dense/sparse checks or
  retrieval-recall experiments;
- do not download models or choose dense retrieval dependencies silently;
- do not use source `DOC_IDS` as runtime retrieval evidence;
- do not tune Stage 51 against the frozen test split;
- do not use NVIDIA `train.json` as held-out evidence;
- do not treat PrimeQA validation rows as independent held-out evidence;
- do not treat MSQA answer-source metrics as PrimeQA-style document-citation
  metrics;
- do not continue HQA download/probe work unless the user explicitly redirects
  back to that route.

## Artifacts

```text
artifacts/evaluation_strategy_stage54_review.json
artifacts/evaluation_strategy_stage54_visuals/
artifacts/nvidia_heldout_leakage_stage53.json
artifacts/nvidia_heldout_leakage_stage53_visuals/
artifacts/external_eval_dataset_discovery_stage55.json
artifacts/external_eval_dataset_discovery_stage55_visuals/
artifacts/msqa_schema_probe_stage56.json
artifacts/msqa_schema_probe_stage56_visuals/
artifacts/msqa_evaluation_split_stage57.json
artifacts/msqa_evaluation_split_stage57.jsonl
artifacts/msqa_evaluation_split_stage57_visuals/
artifacts/msqa_topk_baseline_stage58.json
artifacts/msqa_topk_baseline_stage58_visuals/
artifacts/msqa_stage51_compatibility_stage59.json
artifacts/msqa_stage51_compatibility_stage59_visuals/
artifacts/msqa_stage51_protocol_stage60.json
artifacts/msqa_stage51_protocol_stage60_visuals/
artifacts/msqa_stage51_candidate_adapter_stage61.json
artifacts/msqa_stage51_candidate_adapter_stage61_candidates.jsonl
artifacts/msqa_stage51_candidate_adapter_stage61_visuals/
artifacts/msqa_stage51_candidate_distribution_stage62.json
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/
artifacts/msqa_stage51_candidate_adapter_stage63_capped.json
artifacts/msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl
artifacts/msqa_stage51_candidate_adapter_stage63_capped_visuals/
artifacts/msqa_stage51_candidate_distribution_stage63_capped.json
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/
artifacts/msqa_stage51_adapter_comparison_stage64.json
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/
artifacts/msqa_stage51_changed_case_review_stage65.json
artifacts/msqa_stage51_changed_case_review_stage65_visuals/
artifacts/external_eval_dataset_rediscovery_stage66.json
artifacts/external_eval_dataset_rediscovery_stage66_visuals/
artifacts/primeqa_hybrid_split_stage67.json
artifacts/primeqa_hybrid_split_stage67_assignments.jsonl
artifacts/primeqa_hybrid_split_stage67_visuals/
artifacts/primeqa_hybrid_split_stage68_freeze.json
artifacts/primeqa_hybrid_split_stage68_splits/
artifacts/primeqa_hybrid_split_stage68_visuals/
artifacts/primeqa_hybrid_rebuild_stage69.json
artifacts/primeqa_hybrid_rebuild_stage69_questions/
artifacts/primeqa_hybrid_rebuild_stage69_candidates.jsonl
artifacts/primeqa_hybrid_rebuild_stage69_candidates.summary.json
artifacts/primeqa_hybrid_rebuild_stage69_visuals/
artifacts/primeqa_hybrid_development_checks_stage70.json
artifacts/primeqa_hybrid_development_checks_stage70_visuals/
artifacts/primeqa_hybrid_candidate_reranker_development_stage71.json
artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/
artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72.json
artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/
artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73.json
artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals/
artifacts/primeqa_hybrid_bm25_top10_miss_analysis_stage75.json
artifacts/primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals/
artifacts/primeqa_hybrid_retrieval_recall_candidate_design_stage76.json
artifacts/primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals/
artifacts/primeqa_hybrid_query_view_ablation_stage77.json
artifacts/primeqa_hybrid_query_view_ablation_stage77_visuals/
artifacts/primeqa_hybrid_fielded_bm25_fusion_stage78.json
artifacts/primeqa_hybrid_fielded_bm25_fusion_stage78_visuals/
artifacts/primeqa_hybrid_section_bm25_doc_rollup_stage79.json
artifacts/primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals/
artifacts/primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json
artifacts/primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals/
artifacts/primeqa_hybrid_dense_sparse_rrf_comparison_stage81.json
artifacts/primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals/
artifacts/primeqa_hybrid_bm25_k1_b_grid_stage82.json
artifacts/primeqa_hybrid_bm25_k1_b_grid_stage82_visuals/
artifacts/primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83.json
artifacts/primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals/
artifacts/primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts/primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals/
artifacts/primeqa_hybrid_lexical_cluster_diversity_protocol_stage85.json
artifacts/primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals/
artifacts/primeqa_hybrid_lexical_cluster_diversity_comparison_stage86.json
artifacts/primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals/
artifacts/primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87.json
artifacts/primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals/
artifacts/primeqa_hybrid_structured_query_protocol_stage88.json
artifacts/primeqa_hybrid_structured_query_protocol_stage88_visuals/
artifacts/primeqa_hybrid_structured_query_comparison_stage89.json
artifacts/primeqa_hybrid_structured_query_comparison_stage89_visuals/
artifacts/primeqa_hybrid_structured_query_stop_decision_stage90.json
artifacts/primeqa_hybrid_structured_query_stop_decision_stage90_visuals/
artifacts/primeqa_hybrid_section_signal_protocol_stage91.json
artifacts/primeqa_hybrid_section_signal_protocol_stage91_visuals/
artifacts/primeqa_hybrid_section_signal_comparison_stage92.json
artifacts/primeqa_hybrid_section_signal_comparison_stage92_visuals/
artifacts/primeqa_hybrid_section_signal_stop_decision_stage93.json
artifacts/primeqa_hybrid_section_signal_stop_decision_stage93_visuals/
artifacts/primeqa_hybrid_score_margin_bm25_protocol_stage94.json
artifacts/primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals/
artifacts/primeqa_hybrid_score_margin_bm25_comparison_stage95.json
artifacts/primeqa_hybrid_score_margin_bm25_comparison_stage95.console.txt
artifacts/primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals/
artifacts/primeqa_hybrid_score_margin_bm25_stop_decision_stage96.json
artifacts/primeqa_hybrid_score_margin_bm25_stop_decision_stage96.console.txt
artifacts/primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals/
artifacts/primeqa_hybrid_selective_dense_sparse_protocol_stage97.json
artifacts/primeqa_hybrid_selective_dense_sparse_protocol_stage97.console.txt
artifacts/primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals/
```

The current Stage 67 protocol is recorded in:

```text
docs/primeqa_hybrid_split.md
```

The current Stage 68 freeze is recorded in:

```text
docs/primeqa_hybrid_split_freeze.md
```

The current Stage 69 rebuild is recorded in:

```text
docs/primeqa_hybrid_rebuild.md
```

The current Stage 70 development checks are recorded in:

```text
docs/primeqa_hybrid_development_checks.md
```

The current Stage 71 candidate-reranker development run is recorded in:

```text
docs/primeqa_hybrid_candidate_reranker_development.md
```

The current Stage 72 candidate-reranker changed-case review is recorded in:

```text
docs/primeqa_hybrid_candidate_reranker_changed_case_review.md
```

The current Stage 73 candidate-reranker top10 diagnostic is recorded in:

```text
docs/primeqa_hybrid_candidate_reranker_top10_diagnostic.md
```

The current Stage 74 candidate-reranker stop decision is recorded in:

```text
docs/primeqa_hybrid_candidate_reranker_stop_decision.md
```

The current Stage 75 BM25 top10 miss analysis is recorded in:

```text
docs/primeqa_hybrid_bm25_top10_miss_analysis.md
```

The current Stage 76 retrieval-recall candidate design is recorded in:

```text
docs/primeqa_hybrid_retrieval_recall_candidate_design.md
```

The current Stage 77 query-view ablation is recorded in:

```text
docs/primeqa_hybrid_query_view_ablation.md
```

The current Stage 78 fielded BM25 fusion experiment is recorded in:

```text
docs/primeqa_hybrid_fielded_bm25_fusion.md
```

The current Stage 79 section BM25 doc-rollup experiment is recorded in:

```text
docs/primeqa_hybrid_section_bm25_doc_rollup.md
```

The current Stage 80 dense+sparse RRF feasibility check is recorded in:

```text
docs/primeqa_hybrid_dense_sparse_rrf_feasibility.md
```

The current Stage 81 dense+sparse RRF train/dev comparison is recorded in:

```text
docs/primeqa_hybrid_dense_sparse_rrf_comparison.md
```

The current Stage 82 BM25 k1/b grid train/dev experiment is recorded in:

```text
docs/primeqa_hybrid_bm25_k1_b_grid.md
```

The current Stage 83 retrieval-recall exhaustion summary is recorded in:

```text
docs/primeqa_hybrid_retrieval_recall_exhaustion_summary.md
```

The current Stage 84 second-wave retrieval candidate design is recorded in:

```text
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
```

The current Stage 85 lexical cluster diversity protocol freeze is recorded in:

```text
docs/primeqa_hybrid_lexical_cluster_diversity_protocol.md
```

The current Stage 86 lexical cluster diversity train/dev comparison is recorded in:

```text
docs/primeqa_hybrid_lexical_cluster_diversity_comparison.md
```

The current Stage 87 lexical cluster diversity stop decision is recorded in:

```text
docs/primeqa_hybrid_lexical_cluster_diversity_stop_decision.md
```

The current Stage 88 structured query protocol freeze is recorded in:

```text
docs/primeqa_hybrid_structured_query_protocol.md
```

The current Stage 89 structured query train/dev comparison is recorded in:

```text
docs/primeqa_hybrid_structured_query_comparison.md
```

The current Stage 90 structured query stop decision is recorded in:

```text
docs/primeqa_hybrid_structured_query_stop_decision.md
```

The current Stage 91 section signal protocol freeze is recorded in:

```text
docs/primeqa_hybrid_section_signal_protocol.md
```

The current Stage 92 section signal train/dev comparison is recorded in:

```text
docs/primeqa_hybrid_section_signal_comparison.md
```

The current Stage 93 section signal stop decision is recorded in:

```text
docs/primeqa_hybrid_section_signal_stop_decision.md
```

The current Stage 94 score-margin BM25 protocol freeze is recorded in:

```text
docs/primeqa_hybrid_score_margin_bm25_protocol.md
```

The current Stage 95 score-margin BM25 train/dev comparison is recorded in:

```text
docs/primeqa_hybrid_score_margin_bm25_comparison.md
```

The current Stage 96 score-margin BM25 stop decision is recorded in:

```text
docs/primeqa_hybrid_score_margin_bm25_stop_decision.md
```

The current Stage 97 selective dense+sparse protocol freeze is recorded in:

```text
docs/primeqa_hybrid_selective_dense_sparse_protocol.md
```

The current Stage 98 selective dense+sparse train/dev comparison is recorded in:

```text
docs/primeqa_hybrid_selective_dense_sparse_comparison.md
```

The current Stage 99 selective dense+sparse stop decision is recorded in:

```text
docs/primeqa_hybrid_selective_dense_sparse_stop_decision.md
```

The current Stage 100 second-wave route exhaustion summary is recorded in:

```text
docs/primeqa_hybrid_second_wave_route_exhaustion_summary.md
```

The current Stage 101 answer-pipeline error decomposition protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_answer_pipeline_error_decomposition_protocol.md
```

The current Stage 102 answer-pipeline error decomposition analysis is recorded
in:

```text
docs/primeqa_hybrid_answer_pipeline_error_decomposition.md
```

The current Stage 103 evidence/answerability candidate protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_evidence_answerability_candidate_protocol.md
```

The current Stage 104 evidence/answerability comparison-grid protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_evidence_answerability_comparison_protocol.md
```

The current Stage 105 evidence/answerability train/dev comparison is recorded
in:

```text
docs/primeqa_hybrid_evidence_answerability_comparison.md
```

The current Stage 106 evidence/answerability stop decision is recorded in:

```text
docs/primeqa_hybrid_evidence_answerability_stop_decision.md
```

The current Stage 107 validation-failure pattern analysis is recorded in:

```text
docs/primeqa_hybrid_validation_failure_pattern_analysis.md
```

The current Stage 108 failure-pattern redesign protocol freeze is recorded in:

```text
docs/primeqa_hybrid_failure_pattern_redesign_protocol.md
```

The current Stage 109 failure-pattern redesign train-CV/dev comparison is
recorded in:

```text
docs/primeqa_hybrid_failure_pattern_redesign_comparison.md
```

The current Stage 110 failure-pattern redesign stop decision is recorded in:

```text
docs/primeqa_hybrid_failure_pattern_redesign_stop_decision.md
```

The current Stage 111 retrieval-context-miss root-cause audit protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_retrieval_context_miss_audit_protocol.md
```

The current Stage 112 retrieval-context-miss root-cause audit is recorded in:

```text
docs/primeqa_hybrid_retrieval_context_miss_root_cause_audit.md
```

The current Stage 113 retrieval/index redesign protocol freeze is recorded in:

```text
docs/primeqa_hybrid_retrieval_index_redesign_protocol.md
```

The current Stage 114 retrieval/index redesign train-CV/dev comparison is
recorded in:

```text
docs/primeqa_hybrid_retrieval_index_redesign_comparison.md
```

The current Stage 115 retrieval/index redesign stop decision is recorded in:

```text
docs/primeqa_hybrid_retrieval_index_redesign_stop_decision.md
```

The current Stage 116 high-recall first-stage union candidate-pool comparison
is recorded in:

```text
docs/primeqa_hybrid_high_recall_union_comparison.md
```

The current Stage 117 second-stage reranking protocol freeze is recorded in:

```text
docs/primeqa_hybrid_second_stage_reranking_protocol.md
```

The current Stage 118 second-stage reranking train-CV/dev validation is
recorded in:

```text
docs/primeqa_hybrid_second_stage_reranking_validation.md
```

The current Stage 119 second-stage reranking stop decision is recorded in:

```text
docs/primeqa_hybrid_second_stage_reranking_stop_decision.md
```

The current Stage 120 fast-filter plus alternate-screening protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_fast_filter_screening_protocol.md
```

The current Stage 121 fast-filter plus alternate-screening train-CV/dev
validation is recorded in:

```text
docs/primeqa_hybrid_fast_filter_screening_validation.md
```

The current public-source notes on high-recall RAG agent retrieval patterns are
recorded in:

```text
docs/rag_high_recall_agent_research_notes.md
```

The current Stage 122 fast-filter screening changed-case review is recorded in:

```text
docs/primeqa_hybrid_fast_filter_screening_changed_case_review.md
```

The current Stage 123 first-stage recall expansion protocol freeze is recorded
in:

```text
docs/primeqa_hybrid_first_stage_recall_expansion_protocol.md
```

The current Stage 124 first-stage recall expansion train-CV/dev validation is
recorded in:

```text
docs/primeqa_hybrid_first_stage_recall_expansion_validation.md
```

The current Stage 125 Stage116 prefix-preserving recall expansion protocol
freeze is recorded in:

```text
docs/primeqa_hybrid_prefix_preserving_recall_expansion_protocol.md
```

The current Stage 126 Stage116 prefix-preserving recall expansion train-CV/dev
validation is recorded in:

```text
docs/primeqa_hybrid_prefix_preserving_recall_expansion_validation.md
```

The current Stage 127 selected-config review for the Stage116
prefix-preserving recall expansion is recorded in:

```text
docs/primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review.md
```

The current Stage 128 agent retrieval integration protocol freeze for the
selected prefix-preserving expansion is recorded in:

```text
docs/primeqa_hybrid_agent_retrieval_integration_protocol.md
```

The current Stage 129 agent retrieval integration train-CV/dev validation is
recorded in:

```text
docs/primeqa_hybrid_agent_retrieval_integration_validation.md
```

The current Stage 130 Stage129 agent-integration failure-pattern review is
recorded in:

```text
docs/primeqa_hybrid_agent_integration_failure_review.md
```

The current Stage 131 append-candidate evidence shortlist redesign protocol
freeze is recorded in:

```text
docs/primeqa_hybrid_append_candidate_evidence_shortlist_protocol.md
```

The current Stage 132 append-candidate evidence shortlist train-CV/dev
validation is recorded in:

```text
docs/primeqa_hybrid_append_candidate_evidence_shortlist_validation.md
```

The current Stage 133 append-candidate evidence shortlist selected-config
review is recorded in:

```text
docs/primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review.md
```

The current Stage 134 Stage116 answer-context plus Stage128 sidecar-observation
agent protocol freeze is recorded in:

```text
docs/primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol.md
```

The current Stage 135 Stage116 answer-context plus Stage128 sidecar-observation
train grouped-CV/dev report-only validation is recorded in:

```text
docs/primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation.md
```

Stage135 separates interface validity from evidence-recovery effectiveness.
Primary-context identity, sidecar isolation, record completeness, and signal
availability are train grouped-CV integrity checks; dev is report-only. The
adapter passed those checks, but its three-slot sidecar captured none of the 9
train and 1 dev incremental append-region gold opportunities. This result
permits train/dev agent-interface implementation, but it is not evidence of
answer-quality improvement, citation-verification effectiveness, runtime
defaultization, or readiness to open the final-test gate.

The current Stage 136 Stage116-primary plus Stage128-sidecar agent orchestrator
and public-safe trace protocol freeze is recorded in:

```text
docs/primeqa_hybrid_sidecar_agent_orchestrator_protocol.md
```

Stage136 is an implementation and protocol-freeze result, not an evaluation
result. Its 21/21 guards confirm the answer/verification channel routing,
sidecar isolation policy, public-safe trace schema, inherited Stage135 safety
facts, and locked test/default/fallback boundaries. Stage137 is frozen as train
five-fold grouped-CV integrity validation followed by one dev report-only pass;
dev cannot select or retune the orchestrator.

The current Stage 137 Stage116-control versus sidecar-agent train grouped-CV/dev
report-only validation is recorded in:

```text
docs/primeqa_hybrid_sidecar_agent_orchestrator_validation.md
```

Stage137 runs both paths on every train/dev row and records the actual generator
and verifier inputs in memory. Train uses five grouped folds for integrity
checks; dev is reported once without selection or retuning. All context, answer,
verification-reason, metric, isolation, and public-trace deltas were zero, but
the sidecar again captured none of 9 train and 1 dev append opportunities.
Therefore agent integration is validated while sidecar effectiveness remains
`safe_but_neutral`; test and runtime-default gates remain closed.

The current Stage 138 optional sidecar-agent entrypoint and executable
action-state protocol freeze is recorded in:

```text
docs/primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol.md
```

Stage138 executes only the deterministic state machine, covering accepted and
refused terminal paths plus invalid-transition rejection. Its 31/31 guards
inherit Stage137 integration and isolation facts while keeping the runtime
entrypoint unimplemented. This is protocol evidence, not train/dev answer
evaluation or runtime action-order validation. Stage139 must implement the
optional adapter and revalidate per-row action traces and Stage137 answer-path
invariance before any runtime integration decision.

The current Stage 139 optional sidecar-agent entrypoint train grouped-CV/dev
report-only validation is recorded in:

```text
docs/primeqa_hybrid_optional_sidecar_agent_entrypoint_validation.md
```

Stage139 executes the retrieval port and entrypoint on all 683 train/dev rows.
Every row has exactly five legal transitions, every dependency is called once,
terminal states match verified refusal, all answer/context/isolation deltas are
zero, and candidate-pool plus split aggregates match the saved Stage137 report
exactly. Therefore the implementation and runtime action order are validated
for an explicit future non-default activation path. This does not authorize
runtime defaultization, final-test evaluation, retries, fallback strategies, or
claims of sidecar effectiveness.

The current Stage 140 online candidate-pool performance and exact-identity
validation is recorded in:

```text
docs/primeqa_hybrid_online_candidate_pool_performance_validation.md
```

Stage140 postpones runtime activation and treats latency as an independent
engineering gate. It vectorizes the unchanged BM25 formulas, reuses the
full-document result for the derived special-token route, and moves index/model
ownership outside the request path. All 683 online candidate pools exactly
match the legacy sequence, and train/dev Recall@10/50/100/200/400 exactly match
Stage127. The observed train/dev P95 latency is 0.450798/0.293909 seconds. Dev is
report-only, test remains locked, and no product latency SLO is inferred from
these measurements.

The current Stage 141 strict non-default runtime activation protocol is
recorded in:

```text
docs/primeqa_hybrid_nondefault_runtime_activation_protocol.md
```

Stage141 freezes the user-selected warm single-request limits of P95 <= 0.300
seconds and P99 <= 1.000 seconds. The future gate requires three complete warm
train passes, all five grouped folds plus the pooled train aggregate to pass,
and then one dev report-only pass. Stage140's train P95 fails the strict target
and train/dev P99 were not measured, so current evidence is explicitly
ineligible. Stage141 freezes policy only; no runtime flag or service entrypoint
is implemented, concurrent serving is unauthorized, and test remains locked.

The current Stage 142 strict warm single-request latency validation is recorded
in:

```text
docs/primeqa_hybrid_strict_latency_validation.md
```

Stage142 uses exact Top-K boundary selection while retaining all score and tie
semantics, then compares every measured optimized pool against a validation-
only historical full-sort pool. Three complete train passes, all five folds in
each pass, pooled train, all pooled folds, and one dev report-only pass meet
P95 <= 0.300 seconds and P99 <= 1.000 seconds. Combined train P95/P99 is
0.111715/0.322262 seconds and dev P95/P99 is 0.094916/0.120182 seconds. Recall
and Agent regression remain unchanged. This authorizes implementation of
non-default single-request wiring, not runtime activation or defaultization.

The current Stage 143 optional runtime wiring validation is recorded in:

```text
docs/primeqa_hybrid_optional_sidecar_runtime_validation.md
```

Stage143 validates disabled, rejected, and eligible startup behavior, builds
the eligible process-scoped resources exactly once, and executes one complete
train grouped-five-fold runtime pass before a single dev report-only pass.
Train/dev runtime and entrypoint trace violations are zero; runtime retrieval
P95/P99 is 0.104243/0.152497 seconds on train and 0.094431/0.123178 seconds on
dev. Recall matches Stage142 and verified F1/gold citations match Stage139.
This validates explicit single-request activation only; concurrency,
defaultization, and test remain closed.

The current Stage 144 strict practical concurrent-runtime validation protocol
is recorded in:

```text
docs/primeqa_hybrid_concurrent_runtime_validation_protocol.md
```

Stage144 freezes profile B for one warm process with four in-flight requests,
synchronized four-request bursts, deterministic `0/7/13/20ms` jitter, and
end-to-end P95/P99 limits of `0.800/1.500s`. Future Stage145 train evidence must
cover three complete 562-row passes per pattern, 39 grouped-fold and pooled
latency scopes, and a five-request overload probe that admits four and rejects
one before any downstream call. Dev remains one mixed report-only pass after
the train gate. Stage144 itself loads no split rows and runs no concurrent
requests; test, defaults, queues, retries, and fallback remain closed.

The current Stage 145 bounded concurrent-runtime validation is recorded in:

```text
docs/primeqa_hybrid_concurrent_runtime_validation.md
```

Stage145 builds the heavy resource graph once, performs one label-free warmup,
and validates a nonblocking concurrency-four research runtime against the
frozen Stage144 profile B workload. Six complete train passes produce 3,372
accepted requests across 39 independently gated latency scopes. Global train
end-to-end P95/P99 is `0.569697/0.763205s`; the worst individual scope is
P95 `0.682807s` and P99 `0.875067s`, so all scopes meet the
`0.800/1.500s` limits. The five-request overload probe admits four and rejects
one with a typed error before any downstream call.

The final barrier-synchronized harness records actual arrival fidelity rather
than assuming thread submission equals arrival. Synchronized bursts have a
maximum observed offset of `0.6999ms`. Deterministic jitter retains its exact
target `0/7/13/20ms` schedule, while actual absolute offset error has P99
`15.815290ms` and one Windows scheduling outlier of `286.9019ms`; Stage144 did
not define this diagnostic as a decision gate. Dev is loaded once only after
the train gate and matches Stage143 behavior with end-to-end P95/P99
`0.591977/0.695942s`. Test, application activation, defaultization, queues,
retries, and fallback remain closed.

The current Stage 146 concurrent application-activation validation is recorded
in:

```text
docs/primeqa_hybrid_concurrent_runtime_activation_validation.md
```

Stage146 adds a separate `TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT` setting that
defaults to false and is mutually exclusive with the single-request setting.
The bootstrap recomputes Stage145 evidence and compares it with the saved
evidence before any resource build. Disabled and synthetic rejected cases build
zero resources; eligible startup builds once, performs one Top400 warmup, and
returns the runtime used for the complete 3,372-request train workload,
five-request overload probe, and gated dev pass.

The final result passes `43/43` guards and all 39 latency scopes. Global train
end-to-end P95/P99 is `0.559442/0.755804s`; worst-scope P95/P99 is
`0.687264/0.866313s`; dev is `0.539259/0.695918s`. Recall, F1, citations,
terminal states, and candidate depth match Stage145, with zero cross-request
contamination. Deterministic-jitter actual offset error has P99 `15.737221ms`
and maximum `303.5484ms`, retained as a machine-specific load-generator
diagnostic. Explicit evidence-gated activation is now available, but defaults,
network serving, test, queues, retries, and fallback remain closed.

The current Stage 147 application Agent request-facade protocol is recorded in:

```text
docs/primeqa_hybrid_agent_request_facade_protocol.md
```

Stage147 reads only the saved Stage146 public aggregate and freezes a
transport-neutral private call contract, an allowlisted public telemetry
contract, exact capacity-error mapping, cooperative pre-dispatch cancellation,
and the `accepting -> draining -> closed` lifecycle. Shutdown rejects new work
and waits naturally for in-flight calls without an implicit timeout or force
cancellation. The executable policy passes `34/34` guards; one compliant case
is eligible and four unsafe synthetic cases are rejected. No split, model,
index, or candidate pool is loaded. The facade itself, network serving,
defaultization, test evaluation, queues, retries, and fallback remain closed.

The current Stage 148 transport-neutral Agent request-facade validation is
recorded in:

```text
docs/primeqa_hybrid_agent_request_facade_validation.md
```

Stage148 separates the label-bearing offline `PrimeQAQuestion` from the new
three-field `PrimeQARuntimeQuery` and updates the online retrieval/answer path
to consume the structural label-free query contract. The implemented facade is
constructible only from an eligible active Stage146 bootstrap. It maps private
responses and citations, emits separate six-field facade and fourteen-field
runtime public telemetry, maps capacity rejection exactly, propagates other
downstream exceptions unchanged, and implements the monotonic
`accepting -> draining -> closed` lifecycle.

The formal aggregate/synthetic validator passes `37/37` guards in `0.001964s`.
Invalid and pre-cancelled requests make zero runtime calls. A blocking
concurrency case proves shutdown rejects new work while draining and waits for
one in-flight call to finish naturally, without timeout or force-cancel. A
separate synthetic Top400 integration test drives the label-free query through
the real online runtime and legally ends in verified refusal. The formal
validator loads no split, model, index, or candidate pool. Network serving,
defaultization, test, queues, retries, and fallback remain closed.

The current Stage 149 strict local Agent HTTP transport protocol is recorded
in:

```text
docs/primeqa_hybrid_agent_network_transport_protocol.md
```

Stage149 reads only the saved Stage148 public aggregate and freezes a
loopback-only HTTP/1.1 surface with one versioned answer route and separate
liveness/readiness routes. It requires strict JSON without coercion, a 32 KiB
raw-body hard cap, exact field limits, stable success/error schemas, and exact
`400/413/415/422/500/503` mappings. Verified refusal remains HTTP 200, while
capacity and lifecycle failures remain nonblocking 503 outcomes.

The protocol records pre-dispatch disconnect cancellation honestly and does
not claim in-flight hard cancellation. FastAPI lifespan must wait naturally
for facade in-flight work without an application timeout or force-cancel.
Default access logging is disabled in favor of an 18-field public allowlist.
The formal aggregate-only run passes `39/39` guards in `0.001029s`; one exact
case is eligible and five unsafe cases are rejected. No split, question,
document, model, index, candidate pool, network service, or port is opened.
FastAPI implementation is authorized next, but remote serving,
defaultization, test, queues, retries, and fallback remain closed.

The current Stage 150 disabled local FastAPI Agent transport validation is
recorded in:

```text
docs/primeqa_hybrid_agent_http_transport_validation.md
```

Stage150 implements the exact Stage149 loopback-only HTTP/1.1 adapter behind a
new strict setting that remains false by default and requires an eligible
concurrent-runtime bootstrap. The surface has exactly three routes, strict JSON
without coercion, a 32 KiB pre-parse body cap, stable status/error mapping, and
an 18-field public log allowlist. Synchronous facade work runs off the event
loop with four nonblocking admission permits and no application waiting queue.

The formal synthetic/ASGI plus real-loopback run passes `37/37` guards in
`0.300549s`. Four blocked calls complete and the fifth is immediately rejected
with 503; application waiting, queue, retry, and fallback counts remain zero. A
known pre-dispatch disconnect sends zero ASGI frames and makes zero runtime
calls. Natural shutdown observes draining, rejects new work, waits for the
in-flight call, and closes without timeout or force cancellation. The temporary
HTTP/1.1 server stops and its port can be rebound. No split, model, index, or
candidate pool is loaded; test remains locked. The next authorized direction
is a local service-entrypoint composition protocol, not persistent or remote
serving yet.

The current Stage 151 strict local Agent service-entrypoint composition
protocol is recorded in:

```text
docs/primeqa_hybrid_agent_service_entrypoint_protocol.md
```

Stage151 reads only the saved Stage150 public aggregate and freezes an exact
future invocation: `python -m ts_rag_agent.local_agent_service --port <PORT>`.
The port is mandatory, has no default, and is restricted to `1024..65535`;
host/source/reload/worker overrides are forbidden. Stage150 authorization and
both explicit activation flags precede Stage145 recomputation, one resource
build, and one built-in label-free synthetic warmup. No train/dev/test question
row may be used for service warmup.

The protocol requires one prebound loopback listener with no bind retry or
alternate-port fallback, one process/worker, Uvicorn signal ownership on the
main thread, and no shutdown timeout or force cancellation. Its corrected
shutdown order matches Uvicorn 0.51.0: stop accepting and wait HTTP tasks first,
then run FastAPI lifespan to close the transport, then release process
references. The final aggregate-only run passes `33/33` guards in `0.001723s`;
one exact policy case is eligible and five unsafe cases are rejected. No split,
document, model, index, candidate pool, signal handler, listener, or service is
opened. Stage152 may implement the entrypoint after refactoring bootstrap
warmup to `PrimeQARuntimeQuery`; remote serving, defaults, test, queues,
retries, and fallback remain closed.

The current Stage 152 non-default local Agent service-entrypoint validation is
recorded in:

```text
docs/primeqa_hybrid_agent_service_entrypoint_validation.md
```

Stage152 implements the frozen invocation, source ordering, stable exit codes,
18-field terminal event, single resource graph, label-free built-in warmup,
prebound loopback listener, and main-thread Uvicorn ownership. The concurrent
and optional bootstrap warmup signatures now require `PrimeQARuntimeQuery`, so
service startup no longer needs an evaluation-shaped question with emptied
gold fields.

Nine formal synthetic cases validate the source, activation, runtime, resource,
socket, server, and clean-return boundaries; a direct unit case separately
validates unexpected-composition exit code 1. The one formal real-resource
lifecycle loads the technote corpus and retrieval models, then observes
HTTP/1.1 `200/200/200` for live, ready, and answer on `127.0.0.1:18152`. It
returns three citations, closes the transport, releases the listener, and exits
zero after `51.098075s`. The final report passes `46/46` guards. No train, dev,
or test question split is loaded and no test metric is run. The service remains
off by default, loopback-only, nonpersistent after validation, and outside
runtime defaultization. Stage153 may freeze a tool-orchestration protocol;
remote serving, test, queues, retries, and fallback remain closed.

The current Stage 153 deterministic Agent tool-orchestration protocol is
recorded in:

```text
docs/primeqa_hybrid_agent_tool_orchestration_protocol.md
```

Stage153 reads only the saved Stage152 and Stage139 public aggregate reports
and freezes a nine-state, seven-node, eight-transition acyclic workflow. Its
three tools execute sequentially and at most once each: retrieve the exact
Top400 candidate pool, compose from the frozen Top10 generation context, and
verify against the frozen rank-200 prefix. The only branch occurs after
diagnostic observation and terminates at `complete` or `refuse`. The verified
answer remains the sole final response authority.

The protocol is honestly classified as a deterministic workflow rather than an
autonomous Agent: no LLM selects tools, no loop exists, and query rewrite,
second retrieval, memory, persistence, streaming, retry, and fallback are
closed. Each request owns an independent 13-field private state, while public
telemetry is restricted to 20 aggregate fields. The graph may compile once per
process but may not share request state.

Official LangGraph documentation and PyPI were researched on 2026-07-18. The
next implementation adapter selected for proof is
`langgraph.graph.StateGraph`; neither LangGraph nor LangChain is currently
installed, and Stage153 performed no dependency installation. The unconfirmed
preflight passes `45/46` guards with only the confirmation guard false. The
confirmed formal run passes `46/46` in `0.002492s`; one exact case is eligible
and six unsafe cases are rejected. Formal and preflight each produce ten
parseable SVG files. No split, document, model, index, candidate pool, socket,
or service is opened, and test remains locked. Stage154 may implement and prove
the deterministic workflow plus LangGraph adapter; remote serving,
defaultization, test, queues, retries, and fallback remain closed.

The current Stage 154 LangGraph Agent tool-workflow implementation is recorded
in:

```text
docs/primeqa_hybrid_agent_tool_workflow_validation.md
```

Stage154 pins the sole direct `agent` dependency to `langgraph==1.2.9` and
records the exact installed transitive versions. Full LangChain and
langchain-community are not direct dependencies. Installation changed
`websockets` from 16.1 to 15.0.1 under the LangGraph SDK constraint; `pip check`
passes.

The implementation uses one shared node executor behind a framework-neutral
reference engine and a `StateGraph` engine. The graph compiles once per
workflow instance, creates a new exact 13-field private state per request, has
seven nodes and one conditional terminal route, and attaches no checkpointer or
cache. The concurrent runtime factory now uses this graph, so the existing
facade, FastAPI adapter, and local service path execute it when their existing
explicit activation gates pass.

Complete and refuse outputs are equal across both engines. Successful requests
make exactly one retrieval, answer, and verification call over Top400, Top10,
rank-200, and three-sidecar contexts. Invalid node order is rejected before a
tool call. Four simultaneous graph invocations remain isolated with one compile.
The same original retrieval exception reaches the caller; only a public-safe
failure-stage snapshot is recorded and immediately consumed, with zero retry or
fallback actions.

The corrected unconfirmed preflight passes `46/54` guards. One confirmed
current-code real resource and loopback lifecycle ran once on port 18154 and
passes the supporting Stage152 `46/46` guards in `49.813499s`; HTTP/1.1 live,
ready, and answer are `200/200/200`, three citations are returned, and the
listener and transport close. That artifact does not save the real request's
candidate-pool depth, so no real-depth claim is made; separate synthetic HTTP
evidence directly observes Top400. The Stage154 formal report passes `54/54`
in `0.204824s`, and full repository validation is `700 passed` with one known
dependency warning. No evaluation split is loaded and test remains locked.
Stage155 may freeze graph runtime activation and operational observability;
remote serving, defaultization, test, queues, retries, and fallback remain
closed.

The current Stage 155 strict Agent activation and operational-observability
validation is recorded in:

```text
docs/primeqa_hybrid_agent_runtime_observability_validation.md
```

Stage155 adds Stage154 formal evidence and its four current source
fingerprints to the service startup chain. Missing, tampered, or stale evidence
is rejected with exit code 9 before Stage145 loading, resource construction,
warmup, or listener binding. The existing concurrent-runtime and local-HTTP
flags remain the only activation flags; both must be explicitly true. The
runtime remains nondefault and loopback-only.

Every graph invocation now emits one start event, one event for each of the
seven frozen nodes, and one terminal event. The exact 22-field event schema
contains only process-local sequence numbers, monotonic elapsed times, state,
counts, depths, failure stage, and in-flight level. It contains no wall-clock
timestamp, request identifier, question, answer, document identifier, or raw
content. Four simultaneous calls produce 36 isolated events, and a retrieval
failure produces four events while preserving the same original exception and
performing one retrieval with no retry or fallback.

The unconfirmed preflight passes `48/57` guards. The first formal real command
was externally interrupted by the shell tool's approximately 14-second default
limit and produced no formal artifact. After explicit user approval, exactly
one replacement process ran naturally on fixed port 18155. It passes `57/57`
in `40.741661s`, returns HTTP/1.1 `200/200/200`, fingerprints eleven sources,
releases the listener and transport, and emits 18 events for the warmup and
answer request. The warm answer request completes in `59.595ms`, of which
retrieval is `46.777ms` and composition is `6.684ms`.

Stage155 loads no evaluation rows and runs no test metric. Remote serving,
runtime defaultization, test access, observation sampling/batching/export,
queues, retries, fallback, LLM-selected tools, and multi-turn memory remain
closed. Final repository validation is Ruff passing and `712 passed` with one
existing FastAPI `TestClient` deprecation warning. The next direction is to design the local tool-selection and
multi-turn-state boundary before implementation.

The current Stage 156 bounded Agent tool-selection and volatile multi-turn-state
protocol is recorded in:

```text
docs/primeqa_hybrid_bounded_agent_state_protocol.md
```

Stage156 reads only the saved Stage155 public aggregate. It freezes one
structured model decision after the system-owned retrieval and context
preparation steps. The exact model actions are `compose_grounded_answer` and
`refuse_insufficient_evidence`. The model cannot own retrieval, rewrite the
query, request a second retrieval, create tools, execute parallel tools, loop,
or own the final answer. The compose branch runs composition, verification,
read-only diagnostics, and verified finalization once each. The early-refuse
branch skips those operations and uses a fixed system refusal constructor.

The executable `VolatileThreadStateLedger` isolates state by exact opaque thread
handle and retains only completed terminal turns in process memory. Candidate
pools, document contexts, unverified responses, diagnostics, exceptions, and
model reasoning are discarded at the turn boundary. Both turn-count and byte
limits are mandatory constructor inputs, but Stage156 does not claim production
values. Overflow rejects before mutation; no truncation, eviction, implicit
thread creation, reconstruction, retry, or fallback occurs. Explicit close
clears state and process restart loses it because checkpointers and persistent
stores remain disabled.

The preflight passes `42/43` guards with only confirmation false. The confirmed
formal artifact passes `43/43`; two bounded policy cases are eligible, six
unsafe cases are rejected, and all five synthetic thread-state cases pass.
Targeted validation is `24 passed`, and formal/preflight each produce ten
parseable SVGs. No model, split, corpus, index, candidate pool, socket, or
runtime is loaded or changed, and test remains locked. Stage157 may implement a
local structured decision-router adapter after explicit model/provider and
thread-limit selection.

The first full-suite command was externally stopped after about 14 milliseconds
because the shell interpreted `timeout_ms=0` as an immediate timeout rather
than unlimited execution. It produced no pytest result and left no pytest
process. After explicit user confirmation, exactly one hidden replacement
process ran naturally and completed with `736 passed, 1 warning in 12.72s`.
The warning is the existing FastAPI `TestClient` deprecation warning.
Formal and preflight visualization directories each contain ten parseable SVGs.
The formal artifact SHA-256 is
`1057cd70ed0ce872529bdc04d1182b84327a50cf6f9bcce9fedb76a4f2952a97`.

A pre-commit node audit then corrected the early-refusal diagnostics branch, so
the `12.72s` suite is retained only as pre-correction history. Exactly one
current-source hidden pytest process subsequently exited naturally with
`736 passed, 1 warning in 7.31s`.

The current Stage 157 local structured-router and bounded dynamic Agent runtime
validation is recorded in:

```text
docs/primeqa_hybrid_bounded_dynamic_agent_runtime_validation.md
```

Stage157 provisions a separate ignored `.venv` with exact CUDA packages:
PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, and Transformers `5.13.1`.
The verified device is an NVIDIA GeForce RTX 5060 with CUDA available and
compute capability `(12, 0)`. The local-files-only model is
`Qwen/Qwen3-VL-2B-Instruct` at revision
`89644892e4d85e24eaac8bacfd4f463576704203`. The model is the sole GPU
workload; existing dense retrievers remain on CPU.

The runtime is a separate, nondefault nine-node LangGraph. It performs exactly
one system-owned retrieval and one structured model decision. The only valid
actions remain `compose_grounded_answer` and
`refuse_insufficient_evidence`. The compose branch runs composition,
verification, diagnostics, and verified finalization once each. The refusal
branch uses the fixed system refusal and skips those three operations. The
strict JSON schema rejects malformed, fenced, extra-field, trailing-content,
and unauthorized-action output. One nonblocking GPU slot is exposed; there is
no queue, retry, fallback, loop, query rewrite, or second retrieval.

The production thread ledger is fixed at four completed terminal turns and
32 KiB per explicit process-local thread handle. Overflow rejects before
mutation. No checkpointer, persistent store, implicit thread creation,
truncation, or eviction is enabled. The selected prompt profile uses Top10
evidence, at most 600 characters per item, 12,288 input tokens, 32 output
tokens, greedy decoding, strict JSON, and rejection rather than truncation.

A generated synthetic GPU probe selected the compose action with valid schema,
used `696/9` input/output tokens, generated in `905.026ms`, loaded the model in
`12199.471ms`, and reached `4,463,856,128` peak allocated GPU bytes. This probe
contains no dataset split, document corpus, index, or gold label and is not a
quality evaluation.

The first formal process ran naturally through model loading, real technote
retrieval construction, one graph turn, and thread close, but report assembly
then failed because a `ContextVar` metric snapshot did not cross the LangGraph
node context. It produced no formal artifact. One separate monitor command was
externally stopped after its own ten-second sleep; the detached formal process
was unaffected. After explicit user approval, the metric snapshot was carried
in private graph state and exactly one corrected formal process ran naturally.

The corrected formal artifact passes `47/47` guards. On the single generated,
label-free real-corpus runtime query, the model selected
`refuse_insufficient_evidence`; the terminal was refusal, with retrieval/model
calls `1/1` and composition/verification/diagnostics calls `0/0/0`. It used
`2190/11` input/output tokens, generated in `1793.562ms`, loaded the model once,
generated once, and reached `5,358,983,168` peak allocated GPU bytes. Total
formal time was `60.651182s`, including `40.714994s` retrieval construction and
`13.835044s` model loading. Ten SVGs are XML-parseable. The artifact SHA-256 is
`2351015d2c7447e6a5e1c2fe99b6583f0b9067e126ef2bfdd87b0b80c725c3e1`.

Stage157 loads no train, dev, or test split; reads no gold label; and computes
no accuracy, F1, or false-refusal metric. The real turn proves execution only,
not router quality. HTTP integration, runtime defaultization, sockets, remote
models, persistence, queues, retries, fallback, loops, rewrite, and second
retrieval remain closed. Stage158 may integrate this separate runtime behind an
explicit local service activation boundary while preserving these constraints.

Final current-source validation uses one hidden pytest process with no pytest
timeout, monitoring deadline, termination, or restart. It exits naturally with
`758 passed, 1 warning in 15.73s`; stderr is empty, and the warning is the
existing FastAPI `TestClient` Starlette deprecation. All Stage157 Python files
pass Ruff formatting and the complete repository passes Ruff lint. A global
format-only audit also identifies 311 historical Python files that would be
reformatted; Stage157 does not rewrite those unrelated files.

The current Stage 158 explicit bounded dynamic Agent local service validation
is recorded in:

```text
docs/primeqa_hybrid_bounded_dynamic_agent_service_validation.md
```

Stage158 adds a separate two-flag activation boundary and exact open, turn, and
close HTTP routes. It does not change the existing answer route or register the
new runtime as default. Startup authorizes the exact Stage157 artifact,
router/runtime sources, and local model files before building one CPU retrieval
resource set, loading Qwen on CUDA, running one label-free warmup, closing the
temporary thread, composing FastAPI, and opening the loopback listener.

Whole-turn GPU admission is one global nonblocking slot acquired before the
single executor submission, so the executor cannot create an application
waiting queue. Same-thread parallel turn and close are `409`; a different
thread arriving while the slot is occupied is `503`; unknown and duplicate
thread lifecycle operations are `404` and `409`. Shutdown drains an admitted
turn naturally and clears process-local threads without an implicit timeout.

The current-source corrected formal real loopback lifecycle passes `51/51`
guards and returns HTTP `200/200/201/200/200` for live, ready, open, turn, and
close. Warmup and the real HTTP turn each execute one retrieval and one model
decision; both select `refuse_insufficient_evidence`. The real turn uses
`2117/11` input/output tokens and `1055.072ms` model generation. Resources build
once, model generation count is two, peak allocated GPU memory is
`5,358,983,168` bytes, the server thread joins, and port `18158` is released.
Formal total time is `68.564630s`, including `53.588927s` retrieval resource
construction and `7.472638s` model loading.

The capacity-rejection boundary is proven with deterministic synthetic overlap,
not two simultaneous real Qwen HTTP requests. The real request is generated and
label-free, so the result proves execution only. Train, dev, test, gold labels,
and quality metrics remain untouched. The artifact SHA-256 is
`12649c087c3140feeb4121837152b41ef4005922eb73931f3770a5fac83889b0`.

A pre-correction formal had already passed `51/51` with artifact SHA-256
`1358ce88bd494079dfc806ad3416e87c279f14947436122e89d6452e68d937b1`, and the
then-current full suite was `783 passed, 1 warning in 15.94s`. A pre-commit audit
then found inaccurate terminal progress for source-authorized startup failures.
The failure remained fail-closed, but the event could incorrectly report source
authorization as incomplete. The entrypoint and three failure-stage tests were
corrected. Because this changed current source, the old formal and full suite
are retained only as pre-correction history. After explicit user approval,
exactly one corrected formal ran naturally and replaced the artifact.

Current-source validation is Ruff passing, targeted `65 passed, 1 warning in
1.33s`, and full repository `786 passed, 1 existing warning in 10.80s`; full
pytest stderr is empty. Stage159 may measure warm multi-turn behavior and one
real two-request admission rejection on the locked development split only;
test remains locked.

The current Stage 159 full-development warm multi-turn service validation is
recorded in:

```text
docs/primeqa_hybrid_bounded_dynamic_agent_warm_service_validation.md
```

Stage159 authorizes the exact frozen Stage68 development file by SHA-256 and
projects only question title and text into runtime requests. The JSON parser
materializes each authorized dev object, but label fields are not used for
ordering, runtime projection, or metrics. Test is never loaded. Stable
SHA-256(sample identity) ordering creates 30 synthetic four-turn threads and
one synthetic one-turn thread; these groups are explicitly not claimed to be
natural conversations.

One Stage158 service preparation builds retrieval resources once and loads the
local Qwen model once. The warm process executes one startup generation, 121
development generations, and one real capacity-probe generation. All 121 dev
turns return HTTP 200, all 31 threads grow state monotonically and close, and
all 121 branches satisfy their call-count protocol. The model chooses compose
34 times and refusal 87 times. These are operating counts only: no dev gold
label is used, so answer/refusal distribution and 102 emitted citations are not
quality metrics.

Median all-turn end-to-end latency is `1977.732ms`, while p95 is
`11835.247ms`. Per-position average latency rises from `1971.820ms` on turn one
to `6269.388ms` on turn four; turn-four p95 is `16481.358ms`. Average input
tokens and retained state also rise by position. This establishes a real warm
multi-turn long tail but does not identify its semantic cause because the
public artifact stores no individual rows.

The real two-request capacity probe pauses the first admitted request before
runtime execution, observes the second request return HTTP 503
`gpu_capacity_exceeded` in `1.401ms` without downstream dispatch or GPU
admission, then releases the first request to complete real retrieval and Qwen
execution with HTTP 200. Final counters show 122 admitted/completed turns, one
capacity rejection, one maximum in-flight turn, and zero failures, queue,
retry, or fallback actions.

Formal validation passes `65/65` guards, produces ten parseable SVGs, releases
port `18159`, and leaves no process or thread open. The artifact SHA-256 is
`93eb319aeb0c2212f55df0bbb2c2b1790eeba02aa4ec20439464bc72a7f3bfe6`.
Current-source verification is Ruff passing, targeted `52 passed, 1 existing
warning in 1.84s`, and full repository `797 passed, 1 existing warning in
11.52s`; corrected full pytest stderr is empty.

Stage160 remains development-only. It may either analyze the refusal and
latency-tail patterns with private diagnostics or freeze the validated runtime
behavior before further integration. Test evaluation, runtime defaultization,
remote exposure, persistence, queues, retries, fallback, rewrite, and second
retrieval remain closed.

The selected Stage 160 path is documented in:

```text
docs/primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics.md
```

Stage160 runs the exact Stage159 warm workload once on the frozen 121-row
development split, then joins dev gold to validation-only private observations.
Gold is not projected into runtime. Train and test are not loaded. The grouped
five-fold layer contains 117 normalized-question plus answer-document groups
with row counts `25/24/24/24/24`; it reports stability only and performs no
model fitting, policy selection, or threshold tuning.

The formal service completes all 121 HTTP turns and passes `57/57` guards. The
answerable gold candidate-pool hit rate is `92.1053%`, while generation-Top10
hit rate is only `47.3684%`. Among 52 answerable refusals, 5 are candidate-pool
misses, 28 lose gold before generation Top10, and 19 show gold to the model but
still refuse. The primary measured loss is therefore the second-stage context
selection boundary, not first-stage candidate construction.

Average generation accounts for `93.0286%` of end-to-end latency. Router input
token count has Spearman correlation `0.915809` with generation latency,
compared with `0.627997` for retained state bytes and `0.566652` for synthetic
turn position. This is diagnostic association, not causal evidence.

The public artifact SHA-256 is
`e17e5fe5bbc5fef4e25e41234e47b89daf19ea4ef18f3c7270601f0fee7d9377`.
The ignored private artifact has 121 hashed rows, byte SHA-256
`3f10cffe245a4405dfc56044f2a3c0d364fdd0f8723e6cc3ae401260199652db`,
and canonical content SHA-256
`1c8aa4260be5427e13322cb3304e518dd3609c2e38f839cda4f10ce01c911a0d`.
Ten SVGs are XML-parseable and structurally valid. Pixel-level browser review
is not claimed because the local SVG viewer could not process the files and
the in-app browser blocked local-file navigation.

Current-source verification is Stage160 five-file Ruff format passing,
full-repository Ruff lint passing, targeted `70 passed, 1 existing warning in
2.00s`, and full-repository `815 passed, 1 existing warning in 13.84s`. The
full pytest process exits 0 with empty stderr and no imposed runtime deadline.

The next eligible experiment is train plus grouped-CV design for a fast
second-stage reranker or generation-context selector, with dev held out from
fit and test still locked. Runtime defaultization, remote exposure,
persistence, queueing, retry, fallback, query rewrite, and second retrieval
remain closed.
