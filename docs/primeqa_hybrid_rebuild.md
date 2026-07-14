# PrimeQA Hybrid Rebuild

This document records Stage 69: rebuilding local loaders and train/dev candidate
artifacts from the frozen PrimeQA/TechQA hybrid split.

Stage 69 uses `primeqa_hybrid_stage68_v1`. It does not use the frozen test split
for training or tuning, does not run final metrics, and does not change the
default runtime.

## Input Split

```text
split_name: primeqa_hybrid_stage68_v1
protocol_version: primeqa_hybrid_split_v1
```

Question ID policy:

```text
PrimeQAQuestion.id = source_split:QUESTION_ID
```

This avoids collisions because validation rows may reuse original dev question
IDs.

## Command

```powershell
python scripts\rebuild_primeqa_hybrid_artifacts.py `
  --output artifacts\primeqa_hybrid_rebuild_stage69.json `
  --question-output-dir artifacts\primeqa_hybrid_rebuild_stage69_questions `
  --candidate-output artifacts\primeqa_hybrid_rebuild_stage69_candidates.jsonl `
  --candidate-summary-output artifacts\primeqa_hybrid_rebuild_stage69_candidates.summary.json `
  --visualization-dir artifacts\primeqa_hybrid_rebuild_stage69_visuals `
  --candidate-splits train,dev `
  --retrieval-top-k 5 `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --candidate-limit 25 `
  --min-candidate-score 2.0
```

## Loaded Split Summary

| Split | Rows | Answerable | Unanswerable | Unique answer docs | Unique candidate docs |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 562 | 370 | 192 | 309 | 19,602 |
| dev | 121 | 76 | 45 | 71 | 5,373 |
| test | 247 | 175 | 72 | 145 | 8,954 |

The test split is loaded only so the loader contract and question artifact can
be verified. It is not used for candidate training artifacts.

## Candidate Artifact

Candidate artifact splits:

```text
train, dev
```

Forbidden tuning split:

```text
test
```

Candidate build configuration:

```text
retrieval_top_k: 5
evidence_selector: hybrid_routing_answer_aware_mcpd3_section_span_mcpd1
max_candidates_per_document: 3
candidate_limit: 25
min_candidate_score: 2.0
```

Candidate build result:

```text
total answerable questions used: 446
total candidate rows: 5993
train questions: 370
dev questions: 76
train candidate rows: 5006
dev candidate rows: 987
average rows per question: 13.4372
average top candidate token F1: 0.2343
average best candidate token F1: 0.4263
average oracle gain vs top candidate: 0.1920
questions with gold-document candidate: 274
gold-document candidate rows: 716
```

Rows by route:

| Route | Candidate rows |
| --- | ---: |
| other | 2,658 |
| error_or_log | 1,300 |
| install_upgrade_config | 1,087 |
| how_to_or_lookup | 592 |
| security_bulletin_vulnerability_detail | 236 |
| limitation_or_restriction | 60 |
| security_bulletin_post_fix_behavior | 30 |
| security_bulletin_remediation | 30 |

## Guard Checks

```text
test_split_not_used_for_candidate_training_artifact: passed
candidate_build_splits_match_allowed_train_dev: passed
all_loaded_splits_have_rows: passed
candidate_rows_have_no_test_split: passed
```

## Artifacts

These are local ignored artifacts and are not committed by git policy.

Report:

```text
artifacts/primeqa_hybrid_rebuild_stage69.json
sha256: 7e92de033c58fe5623e0d4bff422f7e5f3acd1bb5673f7f919af9315ea5ec633
```

PrimeQA-compatible question JSON files:

```text
artifacts/primeqa_hybrid_rebuild_stage69_questions/primeqa_hybrid_stage69_train_Q_A.json
rows: 562
sha256: 1db9f81f5ad0aec1f79eb42d90d201117e84f1f29331f31e63dc31a8a6c0105c

artifacts/primeqa_hybrid_rebuild_stage69_questions/primeqa_hybrid_stage69_dev_Q_A.json
rows: 121
sha256: 7b84c09f8dd1e30aacf10e075a8c1da8f732cf84643fa789fefda42ca4b43081

artifacts/primeqa_hybrid_rebuild_stage69_questions/primeqa_hybrid_stage69_test_Q_A.json
rows: 247
sha256: 1ad3c643131c8fddbed36aefb11141cf616022374f2774a248577d62bd04a021
```

Candidate artifacts:

```text
artifacts/primeqa_hybrid_rebuild_stage69_candidates.jsonl
rows: 5993
sha256: d379d59f5172394a40bcd1852aa8188f2dec18d4abcae20d08acd992a802da4d

artifacts/primeqa_hybrid_rebuild_stage69_candidates.summary.json
sha256: a753848fe2f6c111e2a376c53522ce5ca67536d0203d5addd135f86beaa6332d
```

Visualizations:

```text
artifacts/primeqa_hybrid_rebuild_stage69_visuals/stage69_primeqa_loaded_split_rows.svg
sha256: da91eb8de3e7364442562f80d10e12b4f8d8baa3bfa3772f48a48b39ba60e321

artifacts/primeqa_hybrid_rebuild_stage69_visuals/stage69_primeqa_loaded_answerable_rows.svg
sha256: 2685b19cd457c30a98e9083dc8b474fc4052603b4ac5a23a9444a0bcd4f4841d

artifacts/primeqa_hybrid_rebuild_stage69_visuals/stage69_primeqa_candidate_rows_by_split.svg
sha256: a029164ed4f93138383ff989965c3a03ca5783608abea434115caee3897a3bf0

artifacts/primeqa_hybrid_rebuild_stage69_visuals/stage69_primeqa_candidate_questions_by_split.svg
sha256: c6fe1617b294af3faf6d0c0b4f9fb1910951cdbdd4d59153b93f7e7b8179cc78
```

## Runtime Boundary

The candidate JSONL contains runtime features and offline gold labels. Gold
labels are allowed only for offline train/dev candidate development. They must
not become runtime features.

The frozen test split remains locked for future final evaluation.

## Decision

```text
status: primeqa_hybrid_train_dev_rebuild_ready
can_run_train_dev_metrics_next: true
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step

Stage 70 completed the PrimeQA train/dev baseline rerun and candidate artifact
development checks. The current follow-up is Stage 71: run train/dev
candidate-reranker policy development on `primeqa_hybrid_stage68_v1`, keeping
the frozen test split locked.

Stage 70 is recorded in:

```text
docs/primeqa_hybrid_development_checks.md
```
