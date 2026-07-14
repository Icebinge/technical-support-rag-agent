# Evaluation Strategy

This document records the current evaluation strategy after Stage 53 blocked the
previously intended NVIDIA held-out path and Stage 55 completed external dataset
discovery.

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
- Default runtime remains unchanged.

## Rejected Path

Do not use `data/raw/nvidia_techqa_rag_eval/train.json` as the current held-out
defaultization test. It has complete normalized question overlap with PrimeQA
train/dev, so any quality metric reported as held-out would be misleading.

## Chosen Path

### External Independent Evaluation Set

Status: user-confirmed on 2026-07-14; Stage 55 discovery complete.

This is still the cleanest path to a real defaultization decision. It preserves
the Stage 51 candidate as frozen and looks for an evaluation source that was not
used in the PrimeQA train/dev development loop.

Stage 55 result:

- Recommended candidate: Microsoft Q&A (MSQA).
- Stage 55 fit score: 17, from a generated audit rubric, not a model metric.
- Reason: MSQA has the strongest external technical-support fit and a public
  dataset license, but it still needs local schema and leakage checks.
- Blocking limitations:
  - no native unanswerable rows, because MSQA filters to accepted-answer rows;
  - source-link and citation coverage are not yet measured locally;
  - the CSV has not been downloaded or parsed in this repository;
  - no PrimeQA train/dev leakage audit has been run on MSQA rows.

Required next steps:

1. Download or sample MSQA only after recording URL, size, and checksum.
2. Probe CSV headers and parse a small local sample.
3. Measure source-link and `learn.microsoft.com` documentation-link coverage.
4. Run exact and near-duplicate leakage audit against PrimeQA train/dev.
5. Freeze the MSQA evaluation split before comparing top-k and Stage 51.

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
- Do not compare top-k against Stage 51 on MSQA.

## Artifacts

```text
artifacts/evaluation_strategy_stage54_review.json
artifacts/evaluation_strategy_stage54_visuals/
artifacts/nvidia_heldout_leakage_stage53.json
artifacts/nvidia_heldout_leakage_stage53_visuals/
artifacts/external_eval_dataset_discovery_stage55.json
artifacts/external_eval_dataset_discovery_stage55_visuals/
```
