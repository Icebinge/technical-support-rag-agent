# External Evaluation Dataset Discovery

This document records the Stage 55 discovery snapshot for finding a replacement
held-out source after NVIDIA TechQA-RAG-Eval `train.json` was blocked by leakage.

The scores below are generated audit-rubric scores, not model quality metrics.
No dataset metrics were run in this stage.

## Recommendation

Recommended next candidate:

```text
microsoft_msqa
```

Recommended next stage:

```text
Stage 56: MSQA local schema probe, source-link coverage audit, and PrimeQA leakage audit protocol
```

MSQA is recommended because it is the best external technical-support match found
in this stage. It is not yet approved as a held-out test set. It must first pass
local schema probing, source-link/citation coverage checks, and leakage audit
against PrimeQA train/dev.

## Candidate Ranking

| Candidate | Status | Fit score | Domain fit | Citation fit | Effort | Current decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `microsoft_msqa` | `recommended_for_stage56_schema_probe` | 17 | 3 | 2 | 2 | Probe next; do not run metrics yet |
| `multidoc2dial` | `secondary_document_grounded_reference` | 15 | 1 | 3 | 3 | Keep as schema reference |
| `natural_questions` | `control_benchmark_only` | 15 | 0 | 3 | 3 | Not a technical-support test set |
| `doc2dial` | `secondary_document_grounded_reference` | 14 | 1 | 3 | 3 | Keep as citation-schema reference |
| `stackexchange_dumps` | `manual_derivation_candidate_only` | 11 | 2 | 1 | 4 | Requires derivation and attribution plan |
| `msdialog` | `blocked_until_access_and_license_confirmation` | 10 | 3 | 0 | 4 | Blocked until access/license are confirmed |

Fit score formula:

```text
domain_fit * 2
+ schema_fit
+ citation_fit
+ answerability_fit
+ license_fit
+ independence_fit
```

Domain fit is weighted twice because the current objective is a
technical-support RAG defaultization decision, not a generic QA benchmark.

## MSQA Readiness Boundary

Stage 55 found MSQA promising, but not ready for metrics.

Known strengths:

- external to the PrimeQA/TechQA and NVIDIA development loop;
- Microsoft product and IT support domain;
- 32,252 QA rows according to the public README;
- human-generated accepted answers;
- public data files include `msqa-32k.csv` and `test_id.txt`;
- dataset license is listed as CDLA-Permissive-2.0;
- README shows link standardization and Azure documentation processing code.

Known risks:

- no native unanswerable rows, because unanswered rows are filtered out;
- source-link and citation coverage is not measured locally yet;
- `msqa-32k.csv` has not been downloaded or parsed by this repository;
- no exact or near-duplicate leakage audit has been run against PrimeQA
  train/dev;
- the GitHub repository is archived, so future maintainer fixes are unlikely.

Therefore:

- do not run Stage 51 metrics on MSQA in Stage 55;
- do not compare MSQA top-k baseline with Stage 51 yet;
- do not treat answer-only rows as citation-ready;
- do not change default runtime.

## Source Snapshot

Sources checked on 2026-07-14:

- Microsoft MSQA GitHub:
  `https://github.com/microsoft/Microsoft-Q-A-MSQA-`
- MSQA paper:
  `https://aclanthology.org/2023.emnlp-industry.29/`
- CDLA-Permissive-2.0 license:
  `https://cdla.dev/permissive-2-0/`
- Doc2Dial dataset card:
  `https://huggingface.co/datasets/IBM/doc2dial`
- MultiDoc2Dial dataset card:
  `https://huggingface.co/datasets/IBM/multidoc2dial`
- MSDialog dataset page:
  `https://ciir.cs.umass.edu/downloads/msdialog/`
- Stack Exchange data dump:
  `https://archive.org/download/stackexchange`
- Stack Exchange license and attribution post:
  `https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/`
- Natural Questions repository:
  `https://github.com/google-research-datasets/natural-questions`

## Artifacts

```text
artifacts/external_eval_dataset_discovery_stage55.json
artifacts/external_eval_dataset_discovery_stage55_visuals/stage55_candidate_fit_score.svg
artifacts/external_eval_dataset_discovery_stage55_visuals/stage55_candidate_domain_fit.svg
artifacts/external_eval_dataset_discovery_stage55_visuals/stage55_candidate_citation_fit.svg
artifacts/external_eval_dataset_discovery_stage55_visuals/stage55_candidate_effort_score.svg
```

These artifacts are local outputs under `artifacts/` and are not committed by
git policy.
