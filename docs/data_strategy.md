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
retrieval recall is the next blocking issue.
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
