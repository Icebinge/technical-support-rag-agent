# Evaluation Strategy

This document records the current evaluation strategy after Stage 53 blocked the
previously intended NVIDIA held-out path.

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
- Default runtime remains unchanged.

## Rejected Path

Do not use `data/raw/nvidia_techqa_rag_eval/train.json` as the current held-out
defaultization test. It has complete normalized question overlap with PrimeQA
train/dev, so any quality metric reported as held-out would be misleading.

## Available Paths

### 1. External Independent Evaluation Set

Status: available only after user confirmation.

This is the cleanest path to a real defaultization decision. It preserves the
Stage 51 candidate as frozen and looks for or constructs an evaluation source
that was not used in the PrimeQA train/dev development loop.

Required first steps:

1. List candidate external sources and license constraints.
2. Check whether each source has questions, answers, answerability, and
   evidence.
3. Run leakage audit against PrimeQA train/dev and Stage 51 artifacts.
4. Evaluate the frozen candidate only after the source passes leakage checks.

### 2. Rebuild A Leak-Safe PrimeQA Split

Status: available only after user confirmation.

This avoids needing a new dataset, but it invalidates current Stage 31-53
model-selection evidence. The current Stage 51 candidate cannot be defaultized
from old evidence under this path.

Required first steps:

1. Design grouped split rules by normalized question and document identity.
2. Rebuild the candidate-reranker dataset only from the new train split.
3. Rerun the dev readiness workflow from scratch on the new dev split.
4. Use the new test split once after a new protocol freeze.

### 3. Freeze Without Defaultization

Status: available only after user confirmation.

This keeps Stage 51 as a documented non-default research result and keeps top-k
as the default runtime. It is the lowest-effort and safest path if no independent
evaluation source is available now, but it cannot support a defaultization
decision.

## Decision Boundary

No further evaluation-design action should be taken without user confirmation
of one of the available paths. Until then:

- Do not change the default runtime.
- Do not run pseudo-held-out metrics.
- Do not tune the Stage 51 candidate.
- Do not use NVIDIA `train.json` as held-out evidence.

## Artifacts

```text
artifacts/evaluation_strategy_stage54_review.json
artifacts/evaluation_strategy_stage54_visuals/
artifacts/nvidia_heldout_leakage_stage53.json
artifacts/nvidia_heldout_leakage_stage53_visuals/
```
