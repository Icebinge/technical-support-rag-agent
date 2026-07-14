# Evaluation Strategy

This document records the current evaluation strategy after Stage 53 blocked the
previously intended NVIDIA held-out path, Stage 55 completed external dataset
discovery, Stage 56 probed MSQA locally, Stage 57 froze the MSQA evaluation
split, Stage 58 recorded the MSQA answer-source baseline, Stage 59 blocked
direct Stage 51 comparison on the current MSQA task, Stage 60 designed a
recommended MSQA source/citation protocol, Stage 61 completed the user-confirmed
adapter dry run, and Stage 62 blocked direct Stage 51 comparison because of
candidate-pool mismatch.

## Current Facts

- Stage 51 policy remains a non-default candidate:
  `candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker`.
- Stage 51 passed dev/train readiness checks, but dev/train are not final
  held-out evidence.
- NVIDIA TechQA-RAG-Eval `train.json` is blocked as an independent held-out
  source for the current development history.
- Stage 53 found:
  - NVIDIA rows: 910
  - exact overlap questions against PrimeQA train/dev: 910
  - exact overlap pairs: 974
  - unhandled overlap questions: 910
- The user confirmed the external-independent-evaluation path after Stage 54.
- Stage 55 recommends Microsoft Q&A (MSQA) only as the next schema-probe
  candidate, not as an already usable held-out test set.
- Stage 56 downloaded and parsed MSQA locally:
  - local rows parsed: 32,236
  - README row-count claim: 32,252
  - row-level Microsoft Learn Q&A URL coverage: 32,236 / 32,236
  - PrimeQA train/dev exact normalized question overlaps: 0
  - `test_id.txt` IDs found in CSV: 587 / 588
- Stage 57 defined the adapter contract and froze
  `msqa_stage57_project_eval_v1`:
  - answer field: `ProcessedAnswerText`
  - source URL field: `Url`
  - no answer-field fallback
  - near-duplicate leakage threshold: token Jaccard `0.9`
  - exact overlaps against PrimeQA train/dev: 0
  - near-duplicate overlaps against PrimeQA train/dev: 0
  - selected evaluation rows: 3,301
- Stage 58 ran MSQA answer-source BM25 baselines on the frozen split:
  - primary answer-only hit@1: 0.4147
  - primary answer-only hit@10: 0.6128
  - primary answer-only MRR: 0.4762
  - primary answer-only average top1 token F1: 0.5138
  - diagnostic question+answer page-text hit@1: 1.0
- Stage 59 reviewed Stage 58 failure modes and compatibility with Stage 51:
  - compatibility gate checks: 7
  - pass: 2
  - blocked: 5
  - blocker count: 5
  - primary answer-only gold-source misses at 10: 1278
  - primary answer-only wrong top1 sources: 1932
  - primary answer-only low-F1 top1 answers: 1758
- Stage 59 decision:
  - `can_run_stage51_candidate_now: false`
  - `can_defaultize_runtime_now: false`
  - diagnostic `question_answer_page_text` is rejected as a comparison target.
- Stage 60 protocol design:
  - recommended source/citation identity: `msqa_row_source_url`
  - recommended candidate construction: `processed_answer_sentence_candidates`
  - `can_run_stage51_candidate_now: false`
- The user confirmed Stage 60 option A before Stage 61.
- Stage 61 adapter dry run:
  - candidate rows: 266,647
  - samples with candidates: 3,301 / 3,301
  - samples with gold-source candidate: 2,023 / 3,301
  - contract checks passed: 7 / 7
  - candidate JSONL rows with `question` field: 0
  - `can_run_stage51_candidate_now: false`
- Stage 62 distribution review:
  - Stage61 median candidates/query: 79
  - Stage61 p10 candidates/query: 51
  - Stage31 max candidates/question: 15
  - Stage61 average candidate count is 6.1134x Stage31 average
  - direct Stage 51 adapter comparison is blocked
- Default runtime remains unchanged.

## Rejected Path

Do not use `data/raw/nvidia_techqa_rag_eval/train.json` as the current held-out
defaultization test. It has complete normalized question overlap with PrimeQA
train/dev, so any quality metric reported as held-out would be misleading.

## Chosen Path

### External Independent Evaluation Set

Status: user-confirmed on 2026-07-14; Stage 55 discovery complete; Stage 56
local MSQA schema probe complete; Stage 57 project-owned MSQA split frozen;
Stage 58 top-k answer-source baseline recorded.

This is still the cleanest path to a real defaultization decision. It preserves
the Stage 51 candidate as frozen and looks for an evaluation source that was not
used in the PrimeQA train/dev development loop.

Stage 55 result:

- Recommended candidate: Microsoft Q&A (MSQA).
- Stage 55 fit score: 17, from a generated audit rubric, not a model metric.
- Reason: MSQA has the strongest external technical-support fit and a public
  dataset license, but it still needs local schema and leakage checks.

Stage 56 result:

- MSQA local CSV is parseable and has 29 fields.
- Required fields `QuestionId`, `AnswerId`, `QuestionText`, `AnswerText`,
  `ProcessedAnswerText`, `Url`, and `Split` have 0 missing values.
- `DoubleProcessedAnswerText` has 76 missing rows, so the future adapter must
  choose an answer field explicitly.
- Source-link coverage is strong at the row level because every row has a
  Microsoft Learn Q&A page URL.
- Processed-answer documentation-link coverage is partial, not complete.
- Exact normalized overlap with PrimeQA train/dev is 0, but near-duplicate
  leakage has not been run.

Stage 57 result:

- Adapter contract version: `msqa_eval_adapter_v1`.
- Frozen split: `msqa_stage57_project_eval_v1`.
- Selected question count: 3,301.
- Selected question ID checksum:
  `26cab0b636845cd321a48c12e8bcbeb5b563e5eb234e63383bbc9d0a9d8cb93b`.
- The split is frozen for the next top-k baseline step only.

Stage 58 result:

- Baseline task: answer-source retrieval over frozen MSQA Q&A rows.
- Primary variant: `answer_only`.
- Diagnostic variant: `question_answer_page_text`.
- The diagnostic variant reaches 1.0 because it indexes the question text, so
  it is not the primary evidence for defaultization.
- Stage 51 comparison remains blocked because Stage 51 is PrimeQA
  document-grounded verified RAG logic, while Stage 58 is an MSQA answer-source
  retrieval baseline.

Stage 59 result:

- Stage 51 cannot be fairly compared directly on the current Stage 58 MSQA
  answer-source task.
- The diagnostic question+answer page-text variant must not be used for
  candidate comparison because it indexes the question text and trivializes
  source-row retrieval.
- Before any candidate comparison, the project needs an MSQA-compatible
  source/citation identity contract and candidate construction protocol.

Stage 60 result:

- Recommended protocol:
  `msqa_row_source_url + processed_answer_sentence_candidates`.
- This uses `QuestionId + AnswerId + Url` as row-source citation identity and
  splits `ProcessedAnswerText` into answer-sentence candidates.
- It excludes `QuestionText`, `AnswerText` fallback, `DoubleProcessedAnswerText`
  fallback, processed-answer links as required citation ground truth, external
  page fetching, and runtime default changes.
- The user confirmed this protocol before Stage 61.

Stage 61 result:

- The MSQA row-source answer-sentence adapter dry run passed.
- Candidate JSONL rows: 266,647.
- All 3,301 evaluation samples have candidate rows.
- 2,023 samples have gold-source candidate rows under top10 answer-only source
  retrieval.
- All contract checks passed.
- No candidate rows contain a `question` field.
- Stage 51 was not run.

Stage 62 result:

- Stage 61 adapter contract checks passed, but candidate distribution is not
  aligned with the Stage 31 training candidate pool.
- Stage61 median candidates/query: 79.
- Stage31 max candidates/question: 15.
- Stage61 p10 candidates/query: 51, so the mismatch is broad rather than an
  outlier-only issue.
- Direct Stage 51 adapter comparison remains blocked.

Stage65 completed the Stage64 changed-case and source-citation tradeoff review.
It confirmed:

```text
consistency_checks_passed: true
changed_answer_count: 719
top3_regression_count: 57
top3_improvement_count: 20
citation_gained_count: 3
citation_lost_count: 0
decision: msqa_stage51_changed_case_review_blocks_defaultization
```

Required next step:

1. Choose the Stage66 evaluation route explicitly.
2. Treat Stage64 and Stage65 as external-adapter risk evidence, not a
   defaultization result.
3. Either find another external dataset, design one frozen MSQA-specific
   rank-4/5 leading-source guard experiment, or freeze Stage51 as non-default
   research evidence.

## Parked Paths

### Rebuild A Leak-Safe PrimeQA Split

Status: parked after user chose the external evaluation route.

This avoids needing a new dataset, but it invalidates current Stage 31-53
model-selection evidence. The current Stage 51 candidate cannot be defaultized
from old evidence under this path.

Required first steps:

1. Design grouped split rules by normalized question and document identity.
2. Rebuild the candidate-reranker dataset only from the new train split.
3. Rerun the dev readiness workflow from scratch on the new dev split.
4. Use the new test split once after a new protocol freeze.

### Freeze Without Defaultization

Status: parked after user chose the external evaluation route.

This keeps Stage 51 as a documented non-default research result and keeps top-k
as the default runtime. It is the lowest-effort and safest path if no independent
evaluation source is available now, but it cannot support a defaultization
decision.

## Current Decision Boundary

The external route is confirmed, but no final evaluation can be reported yet.
Until MSQA or another external source passes schema, citation, license, and
leakage checks:

- Do not change the default runtime.
- Do not run pseudo-held-out metrics.
- Do not tune the Stage 51 candidate.
- Do not use NVIDIA `train.json` as held-out evidence.
- Do not treat MSQA as a held-out test set.
- Do not run additional MSQA Stage51 comparisons without a new frozen protocol;
  the approved single capped Stage64 comparison has already been run.
- Do not reuse MSQA `test_id.txt` as this project's final split until the
  missing ID and upstream filtering assumptions are handled explicitly.
- Do not use an answer-field fallback for MSQA evaluation samples.
- Do not treat Stage 58 MSQA answer-source metrics as PrimeQA-style verified
  RAG document-citation metrics.
- Do not use the Stage 58 diagnostic `question_answer_page_text` variant as a
  candidate-comparison target.
- Do not defaultize from MSQA adapter work without a separate final evaluation
  decision.

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
```
