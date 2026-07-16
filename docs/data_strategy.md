# Data Strategy

## Main Sources

### PrimeQA/TechQA

Use case: training and development source.

The original TechQA package is large and contains the TechQA question-answer
data plus a large IBM Technotes corpus. This project uses it for local corpus
construction and for development experiments after leakage checks.

Local path:

```text
data/raw/primeqa_techqa/TechQA.tar.gz
```

Current project-owned split:

Stage 67 planned a PrimeQA/TechQA hybrid split dry run from local
`training_Q_A.json`, `dev_Q_A.json`, and `validation_reference.json`. This is
the active route because the final target is document-style RAG over technotes.
Stage 68 then froze the split as `primeqa_hybrid_stage68_v1`.

The Stage 68 frozen split:

- selects 10% of unique answer documents for strict document-disjoint test
  isolation;
- sends any group whose candidate `DOC_IDS` contain a selected document to
  `test/document_disjoint`;
- splits the remaining grouped rows into 70% train, 15% dev, and 15% random
  test;
- groups by normalized question plus answer document, or by normalized question
  plus `UNANSWERABLE`;
- includes validation reference rows in the planning pool only, not as
  independent held-out evidence.

Stage 68 materialized local ignored train/dev/test JSONL artifacts. Stage 69
then rebuilt PrimeQA-compatible question files and train/dev candidate-reranker
artifacts from this frozen boundary. Stage 70 ran train/dev BM25 development
baselines and audited the Stage 69 train/dev candidate artifact. Stage 71 then
ran train-only candidate-reranker cross-validation and train-to-dev guarded
policy validation. Stage 72 reviewed the Stage 71 dev changed cases and
visualized the remaining train/dev-only risk. Stage 73 ran a train/dev-only
top10 answer proxy diagnostic for the candidate-reranker policies. Stage 74
stopped the current candidate-reranker policy route as non-actionable for now.
Stage 75 then analyzed BM25 top10 misses on train/dev only and showed that
retrieval recall is the next blocking issue. Stage 76 designed train/dev-only
retrieval-recall candidate experiments and explicitly blocked source `DOC_IDS`
oracle union because those IDs are dataset metadata rather than runtime
user-query evidence. Stage 77 ran the first retrieval-recall candidate,
query-view ablation, and found that title-only and deduplicated full-question
queries underperform the full-question BM25 baseline on train/dev.
Stage 78 ran the second retrieval-recall candidate, fielded title/text BM25
score fusion, and found no dev hit@10 gain from the train-selected challenger.
Stage 79 ran the third retrieval-recall candidate, section BM25 max-section
document rollup, and found a dev hit@10 regression.
Stage 80 checked dense+sparse RRF feasibility and found two compatible local
dense caches, but did not run train/dev dense+sparse metrics because the
model/cache protocol requires confirmation first.
The frozen test split remains locked and must not be used for tuning.

### nvidia/TechQA-RAG-Eval

Original intended use case: final evaluation set.

This dataset is a reduced TechQA-derived RAG evaluation dataset with question,
answer, answerability flag, and evidence contexts.

Local paths:

```text
data/raw/nvidia_techqa_rag_eval/train.json
data/raw/nvidia_techqa_rag_eval/corpus.zip
```

Current boundary:

Stage 53 leakage audit found that `train.json` is not an independent held-out
set for the current PrimeQA train/dev development history. All 910 NVIDIA rows
have exact normalized question overlap with PrimeQA train/dev, producing 974
heldout-development overlap pairs because some development questions normalize
to duplicate text.

Therefore, do not use `nvidia/TechQA-RAG-Eval/train.json` as a held-out
defaultization test for the current Stage 51 candidate unless a future workflow
first redesigns the data split and removes or redoes all affected tuning.

### Microsoft Q&A (MSQA)

Current use case: external evaluation candidate only.

Stage 55 reviewed public dataset-owner sources and recommends Microsoft Q&A
(MSQA) as the next schema-probe candidate because it is an external
technical-support QA dataset collected from Microsoft Q&A, with human-generated
accepted answers and an explicitly listed dataset license.

Current public source:

```text
https://github.com/microsoft/Microsoft-Q-A-MSQA-
```

Stage 55 source-backed facts:

- public GitHub archive;
- 32,252 rows according to the dataset README;
- Microsoft product and IT technical-problem QA domain;
- data files include `msqa-32k.csv` and `test_id.txt`;
- dataset license is listed as CDLA-Permissive-2.0 in the README.

Current boundary:

- Stage 56 downloaded the public MSQA repository into ignored local storage:
  `data/raw/msqa_repo/`.
- Stage 56 parsed `data/msqa-32k.csv` locally and found 32,236 rows, not the
  32,252 rows claimed in the README summary.
- Stage 56 found 100% row-level Microsoft Learn Q&A URL coverage.
- Stage 56 found 0 exact normalized question overlaps against local PrimeQA
  train/dev.
- Stage 56 did not run near-duplicate leakage search.
- Stage 56 did not implement the MSQA adapter contract.
- Stage 56 did not run top-k or Stage 51 metrics on MSQA.
- Stage 57 defined the MSQA adapter contract:
  - question: `QuestionText`;
  - answer: `ProcessedAnswerText`;
  - source URL: `Url`;
  - no fallback to `AnswerText` or `DoubleProcessedAnswerText`.
- Stage 57 ran near-duplicate leakage audit against PrimeQA train/dev at token
  Jaccard threshold `0.9` and found 0 exact or near-duplicate overlaps.
- Stage 57 froze `msqa_stage57_project_eval_v1` with 3,301 selected rows from
  the MSQA CSV `Split == test` source split.
- MSQA is now frozen for the next baseline evaluation step, but it is not yet
  evidence for changing the default runtime.

Before MSQA can support a defaultization decision, the project must have a task
contract that is fair for both the baseline and candidate. Stage 58 recorded the
frozen-split answer-source baseline, and Stage 59 reviewed compatibility with
Stage 51. Direct Stage 51 comparison is blocked because Stage 51 is a PrimeQA
document-grounded evidence composition policy, while the current MSQA task is
answer-source row retrieval.

The next data step is to design an MSQA source/citation adapter and comparison
protocol before any Stage 51 candidate run.

Stage 60 designed the recommended protocol:

```text
msqa_row_source_url + processed_answer_sentence_candidates
```

This protocol uses `QuestionId + AnswerId + Url` as the row-source citation
identity and `ProcessedAnswerText` answer sentences as evidence candidates. It
required user confirmation before implementation. The user confirmed this option,
and Stage 61 completed the adapter dry run:

```text
candidate_rows: 266647
samples_with_candidates: 3301 / 3301
samples_with_gold_source_candidate: 2023 / 3301
contract_checks_passed: 7 / 7
stage51_candidate_run_performed: false
```

Stage 62 reviewed candidate distribution and blocked direct Stage 51 comparison:

```text
Stage61 median candidates/query: 79
Stage61 p10 candidates/query: 51
Stage31 max candidates/question: 15
status: msqa_stage51_adapter_comparison_blocked_by_candidate_pool_mismatch
```

Stages 63-65 completed the Stage31-aligned MSQA cap, comparison, and changed-case
review. Stage 65 blocked defaultization from MSQA adapter evidence. Stage 66
then searched for another external dataset and recommended HQA-Data only as a
schema-probe candidate.

The user later chose the document-style PrimeQA/TechQA split route instead of
continuing the HQA download/probe route. MSQA remains useful external-risk
evidence, but it is not the current final-test source.

## Split Rule

Any future held-out dataset must not be used for:

- prompt tuning
- retriever parameter tuning
- reranker training
- answerability classifier training
- model selection after many repeated attempts

The PrimeQA data can be used for development, but any overlap with NVIDIA
evaluation questions must be removed before training or tuning.

For the current repository state, NVIDIA `train.json` has already been shown to
overlap completely with PrimeQA train/dev, so it is blocked as a held-out
evaluation source.

The current evaluation path options are recorded in:

```text
docs/evaluation_strategy.md
```

The current external dataset discovery snapshot is recorded in:

```text
docs/external_eval_datasets.md
```

The current MSQA local schema probe is recorded in:

```text
docs/msqa_schema_probe.md
```

The current MSQA adapter and frozen split are recorded in:

```text
docs/msqa_evaluation_split.md
```

The current MSQA top-k baseline is recorded in:

```text
docs/msqa_topk_baseline.md
```

The current MSQA Stage 51 compatibility review is recorded in:

```text
docs/msqa_stage51_compatibility.md
```

The current MSQA Stage 51 protocol design is recorded in:

```text
docs/msqa_stage51_protocol.md
```

The current MSQA Stage 51 candidate adapter dry run is recorded in:

```text
docs/msqa_stage51_candidate_adapter.md
```

The current MSQA Stage 51 candidate distribution review is recorded in:

```text
docs/msqa_stage51_candidate_distribution.md
```

The current Stage31-aligned MSQA candidate-pool cap is recorded in:

```text
docs/msqa_stage51_candidate_pool_cap.md
```

The current capped MSQA Stage51 adapter comparison is recorded in:

```text
docs/msqa_stage51_adapter_comparison.md
```

The current MSQA Stage51 changed-case review is recorded in:

```text
docs/msqa_stage51_changed_case_review.md
```

The current second external dataset rediscovery snapshot is recorded in:

```text
docs/external_eval_dataset_rediscovery.md
```

The current PrimeQA/TechQA hybrid split dry run is recorded in:

```text
docs/primeqa_hybrid_split.md
```

The current PrimeQA/TechQA hybrid split freeze is recorded in:

```text
docs/primeqa_hybrid_split_freeze.md
```

The current PrimeQA/TechQA hybrid loader and train/dev artifact rebuild is
recorded in:

```text
docs/primeqa_hybrid_rebuild.md
```

The current PrimeQA/TechQA hybrid train/dev development checks are recorded in:

```text
docs/primeqa_hybrid_development_checks.md
```

The current PrimeQA/TechQA hybrid candidate-reranker development run is recorded
in:

```text
docs/primeqa_hybrid_candidate_reranker_development.md
```

The current PrimeQA/TechQA hybrid candidate-reranker changed-case review is
recorded in:

```text
docs/primeqa_hybrid_candidate_reranker_changed_case_review.md
```

The current PrimeQA/TechQA hybrid candidate-reranker top10 diagnostic is
recorded in:

```text
docs/primeqa_hybrid_candidate_reranker_top10_diagnostic.md
```

The current PrimeQA/TechQA hybrid candidate-reranker stop decision is recorded
in:

```text
docs/primeqa_hybrid_candidate_reranker_stop_decision.md
```

The current PrimeQA/TechQA hybrid BM25 top10 miss analysis is recorded in:

```text
docs/primeqa_hybrid_bm25_top10_miss_analysis.md
```

The current PrimeQA/TechQA hybrid retrieval-recall candidate design is recorded
in:

```text
docs/primeqa_hybrid_retrieval_recall_candidate_design.md
```

The current PrimeQA/TechQA hybrid query-view ablation is recorded in:

```text
docs/primeqa_hybrid_query_view_ablation.md
```

The current PrimeQA/TechQA hybrid fielded BM25 fusion experiment is recorded in:

```text
docs/primeqa_hybrid_fielded_bm25_fusion.md
```

The current PrimeQA/TechQA hybrid section BM25 doc-rollup experiment is recorded
in:

```text
docs/primeqa_hybrid_section_bm25_doc_rollup.md
```

The current PrimeQA/TechQA hybrid dense+sparse RRF feasibility check is recorded
in:

```text
docs/primeqa_hybrid_dense_sparse_rrf_feasibility.md
```

The current PrimeQA/TechQA hybrid dense+sparse RRF train/dev comparison is
recorded in:

```text
docs/primeqa_hybrid_dense_sparse_rrf_comparison.md
```

The current PrimeQA/TechQA hybrid BM25 k1/b grid train/dev experiment is
recorded in:

```text
docs/primeqa_hybrid_bm25_k1_b_grid.md
```

The current PrimeQA/TechQA hybrid retrieval-recall exhaustion summary is
recorded in:

```text
docs/primeqa_hybrid_retrieval_recall_exhaustion_summary.md
```

The current PrimeQA/TechQA hybrid second-wave retrieval candidate design is
recorded in:

```text
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
```

The current PrimeQA/TechQA hybrid lexical cluster diversity protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_lexical_cluster_diversity_protocol.md
```

The current PrimeQA/TechQA hybrid lexical cluster diversity train/dev
comparison is recorded in:

```text
docs/primeqa_hybrid_lexical_cluster_diversity_comparison.md
```

The current PrimeQA/TechQA hybrid lexical cluster diversity stop decision is
recorded in:

```text
docs/primeqa_hybrid_lexical_cluster_diversity_stop_decision.md
```

The current PrimeQA/TechQA hybrid structured query protocol freeze is recorded
in:

```text
docs/primeqa_hybrid_structured_query_protocol.md
```

The current PrimeQA/TechQA hybrid structured query train/dev comparison is
recorded in:

```text
docs/primeqa_hybrid_structured_query_comparison.md
```

The current PrimeQA/TechQA hybrid structured query stop decision is recorded in:

```text
docs/primeqa_hybrid_structured_query_stop_decision.md
```

The current PrimeQA/TechQA hybrid section signal protocol freeze is recorded in:

```text
docs/primeqa_hybrid_section_signal_protocol.md
```

The current PrimeQA/TechQA hybrid section signal train/dev comparison is
recorded in:

```text
docs/primeqa_hybrid_section_signal_comparison.md
```

The current PrimeQA/TechQA hybrid section signal stop decision is recorded in:

```text
docs/primeqa_hybrid_section_signal_stop_decision.md
```

The current PrimeQA/TechQA hybrid score-margin BM25 protocol freeze is recorded in:

```text
docs/primeqa_hybrid_score_margin_bm25_protocol.md
```

The current PrimeQA/TechQA hybrid score-margin BM25 train/dev comparison is
recorded in:

```text
docs/primeqa_hybrid_score_margin_bm25_comparison.md
```

The current PrimeQA/TechQA hybrid score-margin BM25 stop decision is recorded in:

```text
docs/primeqa_hybrid_score_margin_bm25_stop_decision.md
```

The current PrimeQA/TechQA hybrid selective dense+sparse protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_selective_dense_sparse_protocol.md
```

The current PrimeQA/TechQA hybrid selective dense+sparse train/dev comparison
is recorded in:

```text
docs/primeqa_hybrid_selective_dense_sparse_comparison.md
```

The current PrimeQA/TechQA hybrid selective dense+sparse stop decision is
recorded in:

```text
docs/primeqa_hybrid_selective_dense_sparse_stop_decision.md
```

The current PrimeQA/TechQA hybrid second-wave route exhaustion summary is
recorded in:

```text
docs/primeqa_hybrid_second_wave_route_exhaustion_summary.md
```

The current PrimeQA/TechQA hybrid answer-pipeline error decomposition protocol
freeze is recorded in:

```text
docs/primeqa_hybrid_answer_pipeline_error_decomposition_protocol.md
```

The current PrimeQA/TechQA hybrid answer-pipeline error decomposition analysis
is recorded in:

```text
docs/primeqa_hybrid_answer_pipeline_error_decomposition.md
```

The current PrimeQA/TechQA hybrid evidence/answerability candidate protocol
freeze is recorded in:

```text
docs/primeqa_hybrid_evidence_answerability_candidate_protocol.md
```

The current PrimeQA/TechQA hybrid evidence/answerability comparison-grid
protocol freeze is recorded in:

```text
docs/primeqa_hybrid_evidence_answerability_comparison_protocol.md
```

The current PrimeQA/TechQA hybrid evidence/answerability train/dev comparison
is recorded in:

```text
docs/primeqa_hybrid_evidence_answerability_comparison.md
```

The current PrimeQA/TechQA hybrid evidence/answerability stop decision is
recorded in:

```text
docs/primeqa_hybrid_evidence_answerability_stop_decision.md
```

The current PrimeQA/TechQA hybrid validation-failure pattern analysis is
recorded in:

```text
docs/primeqa_hybrid_validation_failure_pattern_analysis.md
```

The current PrimeQA/TechQA hybrid failure-pattern redesign protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_failure_pattern_redesign_protocol.md
```

The current PrimeQA/TechQA hybrid failure-pattern redesign train-CV/dev
comparison is recorded in:

```text
docs/primeqa_hybrid_failure_pattern_redesign_comparison.md
```

The current PrimeQA/TechQA hybrid failure-pattern redesign stop decision is
recorded in:

```text
docs/primeqa_hybrid_failure_pattern_redesign_stop_decision.md
```

The current PrimeQA/TechQA hybrid retrieval-context-miss root-cause audit
protocol freeze is recorded in:

```text
docs/primeqa_hybrid_retrieval_context_miss_audit_protocol.md
```

The current PrimeQA/TechQA hybrid retrieval-context-miss root-cause audit is
recorded in:

```text
docs/primeqa_hybrid_retrieval_context_miss_root_cause_audit.md
```

The current PrimeQA/TechQA hybrid retrieval/index redesign protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_retrieval_index_redesign_protocol.md
```

The current PrimeQA/TechQA hybrid retrieval/index redesign train-CV/dev
comparison is recorded in:

```text
docs/primeqa_hybrid_retrieval_index_redesign_comparison.md
```

The current PrimeQA/TechQA hybrid retrieval/index redesign stop decision is
recorded in:

```text
docs/primeqa_hybrid_retrieval_index_redesign_stop_decision.md
```

The current PrimeQA/TechQA hybrid high-recall first-stage union candidate-pool
comparison is recorded in:

```text
docs/primeqa_hybrid_high_recall_union_comparison.md
```

The current PrimeQA/TechQA hybrid second-stage reranking protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_second_stage_reranking_protocol.md
```

The current PrimeQA/TechQA hybrid second-stage reranking train-CV/dev
validation is recorded in:

```text
docs/primeqa_hybrid_second_stage_reranking_validation.md
```

Stage118 rebuilt 74,000 train and 15,200 dev candidate records in memory only.
It did not write raw candidate rows, raw question text, raw answer text, raw
document text, or raw document IDs.

The current PrimeQA/TechQA hybrid second-stage reranking stop decision is
recorded in:

```text
docs/primeqa_hybrid_second_stage_reranking_stop_decision.md
```

Stage119 reads only the public-safe Stage118 report. It does not load split
files, corpus documents, or candidate rows.

The current PrimeQA/TechQA hybrid fast-filter plus alternate-screening protocol
freeze is recorded in:

```text
docs/primeqa_hybrid_fast_filter_screening_protocol.md
```

Stage120 reads only the public-safe Stage119 stop decision report. It does not
load split files, corpus documents, or candidate rows, and it does not write raw
question text, answer text, document text, document IDs, or candidate rows.

The current PrimeQA/TechQA hybrid fast-filter plus alternate-screening
train-CV/dev validation is recorded in:

```text
docs/primeqa_hybrid_fast_filter_screening_validation.md
```

Stage121 loads only train/dev split files, the public Stage120 protocol, local
corpus documents, and existing local dense cache metadata. It rebuilds candidate
records in memory only. It does not load the test split, does not run final test
metrics, and does not write raw candidate rows or raw document/question/answer
text into the public report.

The current PrimeQA/TechQA hybrid fast-filter screening changed-case review is
recorded in:

```text
docs/primeqa_hybrid_fast_filter_screening_changed_case_review.md
```

Stage122 loads only train/dev split files, the public Stage120 protocol, the
public Stage121 validation report, local corpus documents, and existing local
dense cache metadata. It rebuilds candidate records in memory only. It does not
load the test split, does not run final test metrics, and writes only
public-safe aggregate summaries plus anonymized changed-case hashes.

The current PrimeQA/TechQA hybrid first-stage recall expansion protocol freeze
is recorded in:

```text
docs/primeqa_hybrid_first_stage_recall_expansion_protocol.md
```

Stage123 reads only the public-safe Stage122 report. It does not load split
files, corpus documents, candidate rows, model outputs, or test data. It freezes
a train/dev-only protocol and writes only public-safe configuration summaries.

The current PrimeQA/TechQA hybrid first-stage recall expansion train-CV/dev
validation is recorded in:

```text
docs/primeqa_hybrid_first_stage_recall_expansion_validation.md
```

Stage124 loads only train/dev split files, the public Stage123 protocol, local
corpus documents, and existing local dense cache metadata. It does not load the
test split, does not run final test metrics, and does not write raw candidate
rows, raw question text, raw answer text, raw document text, raw document IDs,
or raw sample IDs.

The current PrimeQA/TechQA hybrid Stage116 prefix-preserving recall expansion
protocol freeze is recorded in:

```text
docs/primeqa_hybrid_prefix_preserving_recall_expansion_protocol.md
```

Stage125 reads only the public-safe Stage124 validation report. It does not load
split files, corpus documents, candidate rows, model outputs, or test data. It
freezes a train/dev-only append-only protocol and writes only public-safe
configuration summaries.

The current PrimeQA/TechQA hybrid Stage116 prefix-preserving recall expansion
train-CV/dev validation is recorded in:

```text
docs/primeqa_hybrid_prefix_preserving_recall_expansion_validation.md
```

Stage126 loads only train/dev split files, the public Stage125 protocol, local
corpus documents, and existing local dense cache metadata. It does not load the
test split, does not run final test metrics, and does not write raw candidate
rows, raw question text, raw answer text, raw document text, raw document IDs,
or raw sample IDs.

The current PrimeQA/TechQA hybrid Stage116 prefix-preserving recall expansion
selected-config review is recorded in:

```text
docs/primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review.md
```

Stage127 reads only the public-safe Stage126 validation report. It does not load
split files, corpus documents, candidate rows, model outputs, or test data. It
keeps runtime defaults unchanged and writes only public-safe selected-config and
agent-integration summaries.

The current PrimeQA/TechQA hybrid Stage116 prefix-preserving recall expansion
agent retrieval integration protocol freeze is recorded in:

```text
docs/primeqa_hybrid_agent_retrieval_integration_protocol.md
```

Stage128 reads only the public-safe Stage127 selected-config review. It does
not load split files, corpus documents, candidate rows, model outputs, or test
data. It freezes only a public-safe agent candidate-pool contract: ranks 1-200
remain the immutable Stage116 prefix, ranks 201-400 are append-only recall
candidates for Stage129 validation, runtime defaults stay unchanged, and
fallback strategies stay disabled.

The current PrimeQA/TechQA hybrid Stage129 agent retrieval integration
train-CV/dev validation is recorded in:

```text
docs/primeqa_hybrid_agent_retrieval_integration_validation.md
```

Stage129 loads only train/dev split files, the public-safe Stage128 protocol,
the public-safe Stage125 executable append config, local corpus documents, and
existing local dense cache metadata. It does not load the test split, does not
run final test metrics, and writes only aggregate public-safe profile,
candidate-pool, guard, and visualization summaries. It does not write raw
candidate rows, raw question text, raw answer text, raw document text, raw
document IDs, or raw sample IDs.

The current PrimeQA/TechQA hybrid Stage130 Stage129 agent-integration
failure-pattern review is recorded in:

```text
docs/primeqa_hybrid_agent_integration_failure_review.md
```

Stage130 reads only the public-safe Stage129 aggregate report. It does not load
split files, corpus documents, candidate rows, model outputs, or test data. It
writes only aggregate public-safe failure-pattern, action-boundary, guard, and
visualization summaries.

The current PrimeQA/TechQA hybrid Stage131 append-candidate evidence shortlist
redesign protocol freeze is recorded in:

```text
docs/primeqa_hybrid_append_candidate_evidence_shortlist_protocol.md
```

Stage131 reads only the public-safe Stage130 aggregate failure review. It does
not load split files, corpus documents, candidate rows, model outputs, or test
data. It freezes three validation-only shortlist configs that keep Stage116
prefix evidence protected and treat Stage128 append candidates as supplemental
evidence candidates. The configs are not runtime defaults and are not fallback
strategies.

The current PrimeQA/TechQA hybrid Stage132 append-candidate evidence shortlist
train-CV/dev validation is recorded in:

```text
docs/primeqa_hybrid_append_candidate_evidence_shortlist_validation.md
```

Stage132 loads only train/dev split files, local corpus documents, existing
local dense cache metadata, and public-safe Stage125/128/131 protocol
artifacts. It does not load the test split, does not run final test metrics,
and writes only aggregate public-safe profile, selection, guard, and
visualization summaries. It selected only the conservative sidecar config; this
is not runtime defaultization.

The current PrimeQA/TechQA hybrid Stage133 append-candidate evidence shortlist
selected-config review is recorded in:

```text
docs/primeqa_hybrid_append_candidate_evidence_shortlist_selected_config_review.md
```

Stage133 reads only the public-safe Stage132 aggregate validation report. It
classifies the selected sidecar config as safe but neutral, stops the replacement
append answer-context route, and allows only a future agent-protocol design step.
It does not load split files, corpus documents, candidate rows, model outputs,
or test data.

The current PrimeQA/TechQA hybrid Stage134 Stage116 answer-context plus
Stage128 sidecar-observation agent protocol freeze is recorded in:

```text
docs/primeqa_hybrid_stage116_answer_context_stage128_sidecar_protocol.md
```

Stage134 reads only saved public-safe aggregate Stage128, Stage129, and
Stage133 reports. It does not load split files, corpus documents, candidate
rows, model outputs, or test data. It freezes a two-channel agent protocol:
Stage116 remains the only primary answer-context source, while Stage128/Stage132
append candidates are exposed only as sidecar observations for future
train/dev validation.

The current PrimeQA/TechQA hybrid Stage135 Stage116 answer-context plus
Stage128 sidecar-observation train-CV/dev validation is recorded in:

```text
docs/primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation.md
```

Stage135 loads only the frozen train/dev split files, local training/dev corpus
documents, existing local dense caches, and public-safe Stage125/128/132/134
artifacts. Runtime content handles and observation records exist only in memory.
The committed report contains aggregate summaries only and does not contain raw
questions, answers, document text, document identifiers, sample identifiers, or
candidate rows. The test split remains unloaded.

The current PrimeQA/TechQA hybrid Stage136 Stage116-primary plus
Stage128-sidecar agent orchestrator and public-safe trace protocol freeze is
recorded in:

```text
docs/primeqa_hybrid_sidecar_agent_orchestrator_protocol.md
```

Stage136 reads only the public-safe Stage135 aggregate report. It does not load
train/dev split files, corpus documents, candidate rows, model outputs, runtime
content handles, gold labels, or test data. Its public artifact freezes channel
routing, trace fields, guards, and the Stage137 validation plan. No per-row
runtime trace is written in Stage136.

The current PrimeQA/TechQA hybrid Stage137 Stage116-control versus sidecar-agent
train-CV/dev validation is recorded in:

```text
docs/primeqa_hybrid_sidecar_agent_orchestrator_validation.md
```

Stage137 loads only frozen train/dev split files, local training/dev corpus
documents, existing local dense caches, and public-safe Stage125/128/135/136
artifacts. Per-row control and agent executions, answers, document handles, and
trace objects exist only in memory. The saved report contains aggregate metrics,
identity/isolation counts, offline train/dev opportunity diagnostics, guards,
and visualization metadata. Test remains unloaded.

## Leakage Checks

Before reporting evaluation results:

1. Normalize questions from both sources.
2. Compare exact normalized strings.
3. Compare near-duplicates with token overlap or embedding similarity.
4. Exclude overlapping samples from training or development.
5. Save a leakage report under `artifacts/`.

## Git Policy

Commit:

- scripts
- docs
- schema definitions
- small synthetic fixtures for tests

Do not commit:

- downloaded datasets
- extracted corpus documents
- indexes
- model outputs from evaluation
- model weights
- private notes or credentials

## Honest Reporting

Do not claim model training, retrieval improvement, or answer quality before the
corresponding experiment has actually been run and saved.
