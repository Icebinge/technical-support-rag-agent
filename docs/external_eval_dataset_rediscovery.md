# External Evaluation Dataset Rediscovery

This document records the Stage66 source-backed search for another external
dataset after Stage65 blocked Stage51 defaultization from MSQA adapter evidence.

Stage66 does not download datasets, does not run metrics, does not tune Stage51,
and does not change the default runtime.

Post-Stage66 update:

The user later chose to rebuild a project-owned PrimeQA/TechQA hybrid split for
document-style RAG instead of continuing with HQA download and schema probing.
Therefore, this document is now a historical external-dataset rediscovery
snapshot. HQA remains a possible parked route, but it is no longer the active
next stage.

## Why This Stage Exists

Stage64 and Stage65 showed that MSQA is useful external-adapter risk evidence,
but it does not support defaultizing Stage51:

```text
Stage65 decision: msqa_stage51_changed_case_review_blocks_defaultization
changed_answer_count: 719
top3_regression_count: 57
top3_improvement_count: 20
citation_gained_count: 3
citation_lost_count: 0
```

The user chose the route to find another external dataset.

## Sources Checked

Stage66 checked public dataset-owner, hosting, or access-instruction pages:

```text
https://data.mendeley.com/datasets/p85z3v45xk/1
https://www.sciencedirect.com/science/article/pii/S2352340923003645
https://huggingface.co/datasets/sedthh/ubuntu_dialogue_qa
https://ciir.cs.umass.edu/downloads/msdialog/
https://archive.org/download/stackexchange
https://stackoverflow.com/help/data-dumps
```

## Command

```powershell
python scripts\rediscover_external_eval_datasets.py `
  --output artifacts\external_eval_dataset_rediscovery_stage66.json `
  --visualization-dir artifacts\external_eval_dataset_rediscovery_stage66_visuals
```

## Recommendation

Recommended next candidate:

```text
hqa_data_ubuntu_dialogue
```

Recommended next stage:

```text
Stage 67: HQA-Data local schema probe, file checksum capture, context-span coverage audit, and PrimeQA/MSQA leakage protocol
```

HQA-Data is recommended for the next probe because it is:

- derived from Ubuntu Dialogue Corpus technical-support conversations;
- public on Mendeley Data;
- available in CSV and JSON formats;
- listed with train/test files;
- context/span based;
- licensed as CC BY 4.0 on the Mendeley page.

Important boundary:

HQA-Data is not approved as a final evaluation set. Its questions and answers
are generated from dialogue contexts, not natural user questions paired with
human accepted support answers. It can only move to Stage67 schema probing.

## Candidate Ranking

| Candidate | Status | Fit score | Domain fit | Citation fit | License fit | Effort | Current decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hqa_data_ubuntu_dialogue` | `recommended_for_stage67_schema_probe` | 15 | 2 | 3 | 2 | 2 | Probe next; do not run metrics yet |
| `multidoc2dial` | `strong_document_grounding_reference_not_new_primary` | 15 | 1 | 3 | 3 | 3 | Keep as fallback citation-schema reference |
| `msdialog` | `blocked_until_access_and_license_boundary_confirmation` | 14 | 3 | 1 | 0 | 4 | Blocked by access and non-redistribution terms |
| `askubuntu_stackexchange_dump` | `derivation_candidate_blocked_by_size_access_and_attribution_plan` | 11 | 2 | 1 | 1 | 4 | Parked unless derived-dataset plan is approved |
| `hf_ubuntu_dialogue_qa` | `blocked_by_license_metadata_mismatch` | 10 | 2 | 1 | 0 | 2 | Blocked until license mismatch is resolved |

Fit score formula:

```text
domain_fit * 2
+ schema_fit
+ citation_fit
+ answerability_fit
+ license_fit
+ independence_fit
```

These are discovery audit scores, not answer quality metrics.

## Source-Backed Findings

### HQA-Data

The Mendeley page lists:

```text
published date: 2022-12-15
DOI: 10.17632/p85z3v45xk.1
source: Ubuntu Dialogue Corpus conversations by dialogueID
formats: CSV and JSON
train QA pairs: 29,150
test QA pairs: 7,288
total contexts: 9,364
total QA pairs: 36,438
license: CC BY 4.0
```

The ScienceDirect article says the QA pairs are contained within the context and
points to Mendeley Data as the original data source.

### MSDialog

MSDialog remains the strongest domain match after MSQA because it comes from
Microsoft Community support dialogues. However:

```text
access: contact CIIR for password
use boundary: internal research only
sharing: dataset sharing is forbidden
```

This makes it unsuitable for the next repo artifact unless the user explicitly
approves the access and non-redistribution constraints.

### Ask Ubuntu StackExchange Dump

Ask Ubuntu is domain-relevant, but this path is a dataset-construction project:

```text
historical archive: askubuntu.com.7z listed at about 1,022.0 MB
current access: latest dumps require profile settings access
current access condition: affirm not using the file for LLM training
license family: CC BY-SA
```

This route requires a separate attribution-preserving extraction protocol.

### Hugging Face ubuntu_dialogue_qa

The Hugging Face page is not recommended now because the public page is
internally inconsistent:

```text
metadata license: MIT
dataset card text: Apache License 2.0
dataset viewer: not available
```

Until the licensing ambiguity is resolved, this mirror should not be used.

## Decision

```text
recommended_candidate: hqa_data_ubuntu_dialogue
recommended_next_stage: Stage 67 HQA-Data local schema probe
can_run_final_metrics_now: false
can_download_without_user_confirmation: false
default_runtime_policy: unchanged
```

## Artifacts

```text
artifacts/external_eval_dataset_rediscovery_stage66.json
artifacts/external_eval_dataset_rediscovery_stage66_visuals/stage66_candidate_fit_score.svg
artifacts/external_eval_dataset_rediscovery_stage66_visuals/stage66_candidate_domain_fit.svg
artifacts/external_eval_dataset_rediscovery_stage66_visuals/stage66_candidate_citation_fit.svg
artifacts/external_eval_dataset_rediscovery_stage66_visuals/stage66_candidate_effort_score.svg
```

Stage66 report checksum:

```text
a357e2b466d102c1b30a374e87aca3b895f906ef8d5de6b7ea386741f5f6ace3
```

These artifacts are local ignored outputs and are not committed by git policy.

## Original Next Step

Stage67 should download HQA-Data only after user confirmation, record file URLs,
sizes, and checksums, then run:

1. local CSV/JSON schema probe;
2. context and answer-span coverage audit;
3. exact and near-duplicate leakage checks against PrimeQA and MSQA;
4. decision on whether HQA can be frozen into a project-owned evaluation split.

No final metrics should be run before those checks pass.

## Superseding Next Step

Stage67 was redirected to the user-confirmed PrimeQA/TechQA hybrid split dry
run. The active next stage after that dry run is Stage68: review whether to
freeze the hybrid split and rebuild train/dev/test artifacts from the new split
boundary.
