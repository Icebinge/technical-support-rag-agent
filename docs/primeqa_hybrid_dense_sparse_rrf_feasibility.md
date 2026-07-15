# PrimeQA Hybrid Dense Sparse RRF Feasibility

This document records Stage 80.

## Scope

Stage 80 checks local feasibility for the fourth allowed Stage76
retrieval-recall candidate: `dense_sparse_rrf_train_dev_probe`.

This is a feasibility stage only. It inspects installed packages, existing
project dense/hybrid retrieval code, local dense embedding caches, and local
Hugging Face model cache directories. It does not run train/dev retrieval
metrics, does not load the frozen test split, does not run final test metrics,
does not use source `DOC_IDS` as runtime retrieval evidence, does not download
models, and does not choose a dense model silently.

The report stores dependency versions, cache metadata, model-cache identity,
candidate options, guard checks, and visualization paths. It does not output raw
question text, answer text, document titles, or document body text.

## Command

```text
python scripts\check_primeqa_hybrid_dense_sparse_rrf_feasibility.py ^
  --output artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json ^
  --visualization-dir artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals
```

The actual run completed in `1.214s`.

## Local Dependencies

```text
numpy: available, 2.2.6
sentence-transformers: available, 5.6.0
transformers: available, 5.13.1
torch: available, 2.13.0
scikit-learn: available, 1.7.2
scipy: available, 1.15.3
huggingface-hub: available, 1.23.0
faiss-cpu: not available, not required for the existing NumPy RRF path
```

Existing project code is present:

```text
src\ts_rag_agent\infrastructure\dense_retriever.py
src\ts_rag_agent\infrastructure\dense_embedding_cache.py
src\ts_rag_agent\infrastructure\hybrid_retriever.py
scripts\evaluate_hybrid.py
```

The current project path uses NumPy matrix similarity plus existing
`HybridRetriever` reciprocal-rank fusion. It does not require FAISS.

## Local Dense Caches

Two local dense caches are compatible with the current corpus document IDs.

```text
model: intfloat/e5-small-v2
cache: data\indexes\dense\intfloat__e5-small-v2_512_passage.npz
cache sha256: 91edc7db00363769d6ae4e6a4b1b5a9c23e8682f2d8fefff42f98d0015fb9d63
document_text_max_chars: 512
document_prefix: "passage: "
embedding_shape: 28482 x 384
document_ids_match_current_corpus: true
can_run_without_reencoding_documents: true
can_run_without_model_download: true
huggingface snapshot: ffb93f3bd4047442299a41ebb6fa998a38507c52
```

```text
model: sentence-transformers/all-MiniLM-L6-v2
cache: data\indexes\dense\sentence-transformers__all-MiniLM-L6-v2_1600.npz
cache sha256: 5f2d2ad64ecd5902e6859f0932f69d05c54309dd7216598b708d1fab52975008
document_text_max_chars: 1600
document_prefix: ""
embedding_shape: 28482 x 384
document_ids_match_current_corpus: true
can_run_without_reencoding_documents: true
can_run_without_model_download: true
huggingface snapshot: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

The cache files are ignored by git:

```text
.gitignore:18:data/indexes/*
```

## Historical Metrics Boundary

Stage80 found four older dense/hybrid metric artifacts:

```text
artifacts\dense_dev_metrics.json
artifacts\hybrid_dev_metrics.json
artifacts\dense_e5_small_v2_512_dev_metrics.json
artifacts\hybrid_e5_small_v2_512_dev_metrics.json
```

These are useful as historical evidence that the local caches and scripts have
worked before, but they predate the frozen Stage68 split boundary. They must not
be treated as current train/dev or final-test evidence for
`primeqa_hybrid_stage68_v1`.

## Candidate Options

Stage80 found three eligible next-step options. All require user confirmation
before a train/dev run because they define the dense model/cache protocol.

```text
recommended option:
  compare_existing_cached_dense_models

meaning:
  Run a fixed train/dev-only dense+sparse RRF comparison across both eligible
  local dense caches, select by train, and validate on dev.

download_required:
  false
```

Other eligible options:

```text
single_cached_model::intfloat/e5-small-v2
single_cached_model::sentence-transformers/all-MiniLM-L6-v2
```

## Guard Checks

```text
stage76_source_report_is_stage76: passed
stage76_dense_sparse_candidate_is_allowed: passed
stage79_source_report_is_stage79: passed
stage79_did_not_open_final_test_gate: passed
required_cached_rrf_packages_available: passed
existing_dense_hybrid_code_available: passed
compatible_local_dense_cache_available: passed
no_model_download_attempted: passed
train_dev_metrics_not_run: passed
final_test_metrics_not_run: passed
source_doc_ids_not_used_as_runtime_evidence: passed
default_runtime_policy_unchanged: passed
```

Additional local check:

```text
Exact raw-field Select-String over the Stage80 JSON for question_title,
question_text, gold_answer, candidate_sentence, document_title, document_text,
and known raw text snippets returned no matches.

A broader substring check does match document_text_max_chars, which is a config
field and not raw document text.
```

Stage80 JSON SHA256:

```text
2441BB1CB1E7888299D3F57962B18CD59DF84E2086AC281105ABCACFC144880F
```

## Visualizations

```text
artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals\stage80_dense_cache_readiness.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals\stage80_dependency_availability.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals\stage80_candidate_options.svg
```

Visualization SHA256:

```text
stage80_dense_cache_readiness.svg: B53ABF4528674DC3BA84FA1EED0B1BCD482B9DAF53056F3D486CCECF6C53127A
stage80_dependency_availability.svg: 177D8E21784A9139F258F88499E16C19806E15D87238ACF3CB71D373D7893EC2
stage80_candidate_options.svg: 395FD5B05B7D94F0CB879BE2671FD6DA26A81AE54E863895F62E6C6CF0A467B0
```

## Decision

```text
status: primeqa_hybrid_dense_sparse_rrf_feasibility_completed
compatible_local_dense_cache_count: 2
can_continue_train_dev_development: true
can_run_dense_sparse_rrf_without_download: true
requires_user_confirmation_before_train_dev_run: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

## Next Step Status

Stage81 was run after the user confirmed the recommended Stage80 option:
`compare_existing_cached_dense_models`.

Stage81 compared both compatible local dense caches on train/dev only, selected
on train, and validated on dev. It kept the frozen test split locked, avoided
source `DOC_IDS` as runtime retrieval evidence, did not run final test metrics,
did not download models, and did not change runtime defaults.

The Stage81 record is:

```text
docs\primeqa_hybrid_dense_sparse_rrf_comparison.md
```
