# MSQA Top-K Baseline

This document records the Stage 58 MSQA frozen-split answer-source top-k
baseline.

Stage 58 does not run PrimeQA-style verified RAG metrics, does not compare
Stage 51, does not tune policies, and does not change the default runtime.

## Input

Frozen split:

```text
msqa_stage57_project_eval_v1
```

Frozen split JSONL:

```text
artifacts/msqa_evaluation_split_stage57.jsonl
```

Current JSONL checksum after Stage 58 normalization:

```text
a60db5be5b1a6bfbf24d32ffc99c5482f57ad3462c39cd7b8510cc3c8d569bb3
```

Stage 58 found that the original Stage 57 JSONL contained Unicode line
separators inside JSON strings. The JSON objects were valid, but some tools
treated those characters as physical line breaks. Stage 58 updated the writer to
escape `U+2028` and `U+2029`, then regenerated the Stage 57 ignored artifact.
The selected question ID checksum did not change.

## Boundary

MSQA provides Q&A rows and row-level source URLs, not a separate technote-style
documentation corpus. Therefore this stage evaluates an answer-source retrieval
baseline:

```text
query: frozen MSQA question
gold source: same frozen MSQA Q&A row
gold answer: ProcessedAnswerText
```

Metrics:

- hit@k and MRR: whether BM25 retrieves the gold MSQA Q&A source row;
- top1 token F1: retrieved top1 answer text against the frozen gold answer;
- oracle@k token F1: best answer text among retrieved top-k rows against the
  frozen gold answer.

These are not document-span citation metrics.

## Baseline Variants

Primary variant:

```text
answer_only
```

The BM25 document text is only `ProcessedAnswerText`. The question text is not
indexed.

Diagnostic variant:

```text
question_answer_page_text
```

The BM25 document text is `QuestionText + ProcessedAnswerText`, approximating a
Q&A page where the question itself is searchable. This is expected to be much
easier and is not the primary baseline.

Corpus scope:

```text
frozen_split_only
```

The earlier attempt to run against all 32,236 MSQA contract rows timed out after
about 304 seconds and did not produce a valid Stage 58 artifact. The committed
Stage 58 result uses the frozen split corpus, matching the Stage 57 decision to
run the next baseline on `msqa_stage57_project_eval_v1`.

## Results

Primary `answer_only` baseline:

```text
evaluated_questions: 3301
hit@1: 0.4147
hit@3: 0.5159
hit@5: 0.5604
hit@10: 0.6128
MRR: 0.4762
gold_source_missing_at_10: 1278
top1_wrong_source: 1932
average_top1_token_f1: 0.5138
oracle@10 token_f1: 0.7024
top1_token_f1_below_0.3: 1758
```

Diagnostic `question_answer_page_text` baseline:

```text
evaluated_questions: 3301
hit@1: 1.0
hit@3: 1.0
hit@5: 1.0
hit@10: 1.0
MRR: 1.0
average_top1_token_f1: 1.0
```

The diagnostic result confirms that indexing the original question text makes
the task almost trivial. The primary answer-only result is the meaningful
baseline for deciding what kind of MSQA comparison is fair.

## Decision

Current status:

```text
msqa_topk_baseline_recorded
primary_baseline_variant: answer_only
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
```

Stage 51 comparison remains blocked because Stage 51 was designed around
PrimeQA document-grounded verified RAG evidence candidates, while Stage 58 is an
MSQA answer-source retrieval task. The next step is a compatibility review, not
an automatic candidate run.

## Artifacts

```text
artifacts/msqa_topk_baseline_stage58.json
artifacts/msqa_topk_baseline_stage58_visuals/stage58_msqa_hit_at_1.svg
artifacts/msqa_topk_baseline_stage58_visuals/stage58_msqa_hit_at_10.svg
artifacts/msqa_topk_baseline_stage58_visuals/stage58_msqa_mrr.svg
artifacts/msqa_topk_baseline_stage58_visuals/stage58_msqa_top1_answer_f1.svg
```

Report checksum:

```text
f34f1d749d94ff08e2a62f3a22b58ec9804cddea4535d971c4618666b65a4dd8
```

Stage 59 preflight correction:

During Stage 59 preflight on 2026-07-14, the local ignored Stage 58 JSON
artifact on disk was found inconsistent with the documented frozen-split result.
It contained an older all-contract-row attempt. The Stage 58 frozen-split command
was rerun during Stage 59 to restore the current artifact. The metrics above
remained the frozen-split result; the checksum above is the current local report
checksum after that Stage 59 rerun.

The previously recorded checksum was:

```text
d114f5f3ad1a4e9680a6296c59d66150770d94605c652fea4e2e8e9039897234
```

These artifacts are local ignored outputs and are not committed by git policy.

## Next Step

Stage 59 reviewed MSQA baseline failure modes and blocked direct Stage 51
comparison. Stage 60 designed the recommended MSQA row-source answer-sentence
protocol, but it still requires user confirmation before implementation or any
Stage 51 candidate run.
