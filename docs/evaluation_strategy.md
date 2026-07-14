# Evaluation Strategy

This document records the current evaluation strategy after Stage 53 blocked the
previously intended NVIDIA held-out path, Stage 55 completed external dataset
discovery, Stage 56 probed MSQA locally, and Stage 57 froze the MSQA evaluation
split for baseline evaluation.

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
- Default runtime remains unchanged.

## Rejected Path

Do not use `data/raw/nvidia_techqa_rag_eval/train.json` as the current held-out
defaultization test. It has complete normalized question overlap with PrimeQA
train/dev, so any quality metric reported as held-out would be misleading.

## Chosen Path

### External Independent Evaluation Set

Status: user-confirmed on 2026-07-14; Stage 55 discovery complete; Stage 56
local MSQA schema probe complete; Stage 57 project-owned MSQA split frozen for
baseline evaluation.

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

Required next steps:

1. Run top-k baseline on `msqa_stage57_project_eval_v1`.
2. Record baseline quality and failure modes.
3. Only after that, run the Stage 51 candidate once against the same frozen
   split.

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
- Do not compare top-k against Stage 51 on MSQA before the frozen-split top-k
  baseline is recorded.
- Do not reuse MSQA `test_id.txt` as this project's final split until the
  missing ID and upstream filtering assumptions are handled explicitly.
- Do not use an answer-field fallback for MSQA evaluation samples.

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
```
