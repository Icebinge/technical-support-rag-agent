# MSQA Stage 51 Compatibility Review

This document records the Stage 59 compatibility gate between the Stage 58 MSQA
answer-source baseline and the existing Stage 51 PrimeQA document-grounded
composition policy.

Stage 59 does not run Stage 51, does not tune policies, does not run PrimeQA
verified RAG, and does not change the default runtime.

## Inputs

Stage 58 baseline report:

```text
artifacts/msqa_topk_baseline_stage58.json
```

Current Stage 58 report checksum after the Stage 59 preflight rerun:

```text
f34f1d749d94ff08e2a62f3a22b58ec9804cddea4535d971c4618666b65a4dd8
```

Frozen split:

```text
msqa_stage57_project_eval_v1
```

Adapter contract:

```text
msqa_eval_adapter_v1
```

Evaluated rows:

```text
3301
```

## Failure Mode Review

Primary `answer_only` baseline failure counts:

| Failure mode | Count | Rate |
| --- | ---: | ---: |
| `gold_source_missing_at_10` | 1278 | 0.3872 |
| `top1_wrong_source` | 1932 | 0.5853 |
| `top1_token_f1_below_0_3` | 1758 | 0.5326 |

Primary versus diagnostic gap:

| Metric | Diagnostic minus primary |
| --- | ---: |
| `hit@1` | 0.5853 |
| `hit@10` | 0.3872 |
| `MRR` | 0.5238 |
| `average_top1_token_f1` | 0.4862 |

Interpretation:

- The primary answer-only task still has substantial source-row retrieval
  misses and wrong top1 sources.
- The diagnostic `question_answer_page_text` variant reaches 1.0 because it
  indexes the question text and is not a fair Stage 51 comparison target.
- These are answer-source row retrieval failure modes, not PrimeQA
  document-span citation failure modes.

## Compatibility Gate

| Check | Status | Decision effect |
| --- | --- | --- |
| `frozen_msqa_split_available` | pass | Allows a compatibility review |
| `stage58_primary_baseline_recorded` | pass | Provides the current MSQA baseline reference |
| `stage51_task_semantics_match_msqa` | blocked | Blocks direct Stage 51 comparison |
| `citation_identity_contract_match` | blocked | Requires an MSQA source/citation contract |
| `candidate_feature_contract_available` | blocked | Requires MSQA-compatible candidate construction |
| `diagnostic_variant_usable_for_comparison` | blocked | Rejects diagnostic metrics as comparison target |
| `failure_modes_are_policy_test_ready` | blocked | Requires retrieval/candidate protocol review |

Gate summary:

```text
total_checks: 7
pass: 2
blocked: 5
blocker_count: 5
```

## Decision

Current status:

```text
stage51_msqa_compatibility_blocked
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
rejected_comparison_variant: question_answer_page_text
```

Direct Stage 51 comparison is blocked because the current MSQA baseline is an
answer-source row retrieval task, while Stage 51 is a PrimeQA
document-grounded evidence composition policy.

Before any Stage 51 comparison on MSQA, the project must define:

1. an MSQA-compatible source/citation identity contract;
2. MSQA evidence candidates with sentence, score, retrieval rank, and source
   identity;
3. a frozen comparison protocol that does not use the diagnostic question-text
   index;
4. a baseline and candidate run under the same MSQA-compatible contract.

## Artifacts

```text
artifacts/msqa_stage51_compatibility_stage59.json
artifacts/msqa_stage51_compatibility_stage59_visuals/stage59_msqa_stage51_gate_checks.svg
artifacts/msqa_stage51_compatibility_stage59_visuals/stage59_msqa_answer_only_failure_modes.svg
artifacts/msqa_stage51_compatibility_stage59_visuals/stage59_msqa_variant_metric_comparison.svg
```

Stage 59 report checksum:

```text
9f72f74262ee4c2da0613e2482043366049051aae3a6ea647b617f2bfd6d79b2
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Stage 60 should design the MSQA source/citation adapter and comparison protocol
before any Stage 51 candidate run.
