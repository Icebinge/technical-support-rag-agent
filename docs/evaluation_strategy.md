# Evaluation Strategy

This document records the current evaluation strategy after Stage 73.

The active route is now the project-owned PrimeQA/TechQA hybrid split
`primeqa_hybrid_stage68_v1`. Stage 68 froze local split artifacts, and Stage 69
rebuilt PrimeQA-compatible question files plus train/dev candidate artifacts.
Stage 70 ran train/dev development baselines and candidate artifact checks.
Stage 71 ran train-only candidate-reranker cross-validation and train-to-dev
guarded policy validation. Stage 72 reviewed the candidate-reranker dev changed
cases and generated visualization artifacts. Stage 73 ran a train/dev-only
top10 answer proxy diagnostic. The frozen test split remains locked for future
final evaluation.

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

Status: Stage 73 train/dev top10 candidate-reranker diagnostic completed; final
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

Stage 74 should choose whether to stop reranker-policy development as
non-actionable for now, or refine train/dev reranker gates using the top3/top10
diagnostics. Do not use test for evaluation or tuning.

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
Stage 73 completed a train/dev-only top10 diagnostic. Until a future stage
explicitly opens final evaluation:

- do not run final metrics;
- do not change the default runtime;
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
