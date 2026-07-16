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
