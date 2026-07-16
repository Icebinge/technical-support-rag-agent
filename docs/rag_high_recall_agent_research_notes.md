# RAG Agent High-Recall Retrieval Notes

Checked on 2026-07-16. This note uses public papers and official framework
documentation only. It does not make claims about private product internals.

## What High-Recall RAG Systems Tend To Do

High recall usually comes from a broad first-stage candidate generator, not from
the final reranker alone.

Common patterns:

- use multiple query views or generated query variants;
- combine sparse keyword search and dense semantic search;
- fuse independent result lists with rank-based fusion such as RRF;
- retrieve a wider candidate window, then rerank or screen;
- use learned sparse or late-interaction retrieval when available;
- retrieve at multiple document granularities, including chunk, document, and
  summary/tree levels;
- allow agentic/self-reflective retrieval loops to retry when evidence is weak.

## Public Source Notes

LlamaIndex's reciprocal rerank fusion example combines multiple queries and
multiple indexes, fuses BM25 and vector retrieval, and generates extra queries:

```text
source: https://developers.llamaindex.ai/python/framework/integrations/retrievers/reciprocal_rerank_fusion/
observed lines: combine retrieval results from multiple queries and multiple indexes;
BM25 plus vector fusion; num_queries=4 with original query plus generated queries.
```

Haystack's hybrid retrieval tutorial frames hybrid retrieval as combining
keyword-based and embedding-based retrieval, then ranking the combined results
with a cross-encoder:

```text
source: https://haystack.deepset.ai/tutorials/33_hybrid_retrieval
observed lines: hybrid retrieval combines keyword and embedding retrieval;
BM25 can outperform dense retrieval in specific domains; pipeline uses BM25,
embedding retriever, and a similarity ranker.
```

Elasticsearch documents RRF as combining multiple result sets with different
relevance indicators. Its `rank_window_size` controls the size of each result
set, with larger windows trading performance for relevance:

```text
source: https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion
observed lines: RRF combines multiple result sets; does not require related
scores; rank_window_size controls individual result-set size.
```

ColBERT uses late interaction: query and documents are encoded independently,
then a cheap fine-grained interaction step scores similarity. The paper reports
competitive effectiveness with much lower query-time cost than full cross
encoding:

```text
source: https://arxiv.org/abs/2004.12832
observed lines: late interaction independently encodes query/document, keeps
fine-grained similarity, and supports efficient retrieval.
```

SPLADE-style learned sparse retrieval keeps sparse/inverted-index advantages
while adding neural expansion:

```text
source: https://arxiv.org/abs/2109.10086
observed lines: sparse representations inherit exact-match and inverted-index
properties; SPLADE reports strong dense/sparse benchmark results.
```

HyDE improves zero-shot dense retrieval by generating a hypothetical document,
embedding it, then retrieving real documents near that embedding:

```text
source: https://aclanthology.org/2023.acl-long.99/
observed lines: hypothetical document captures relevance patterns; dense
bottleneck grounds retrieval back to the real corpus.
```

RAPTOR addresses long-document retrieval by embedding, clustering, and
summarizing chunks into a tree, then retrieving across abstraction levels:

```text
source: https://arxiv.org/abs/2401.18059
observed lines: recursive embedding/clustering/summarization creates a tree;
retrieval can use multiple abstraction levels.
```

Self-RAG is not just higher recall. It adapts whether to retrieve and can
retrieve on demand with self-reflection:

```text
source: https://arxiv.org/abs/2310.11511
observed lines: fixed indiscriminate retrieval can hurt; Self-RAG retrieves
adaptively and reflects on retrieved passages.
```

## Implications For This Project

Already aligned:

- Stage116 already follows the broad first-stage idea: multi-route sparse/dense
  union with RRF into top200.
- Stage121 follows the second-stage idea: keep the broad pool, then screen
  without reordering the whole pool.

Current limitation:

```text
train top200 recall: 0.9324
dev top200 recall: 0.9079
```

No second-stage screening method can recover examples whose gold document is not
inside the fixed top200 candidate pool.

Most relevant future directions:

1. Multi-query first-stage expansion, evaluated as a candidate-pool experiment,
   not as an answer/runtime change.
2. Learned sparse retrieval, such as SPLADE-style expansion, if local model and
   index constraints are acceptable.
3. Late-interaction retrieval, such as ColBERT-style retrieval, if index size
   and build time are acceptable.
4. HyDE-style query-to-hypothetical-document retrieval, but only if generated
   queries are logged as public-safe metadata and evaluated train/dev-only.
5. Hierarchical document retrieval for long technotes, especially when gold
   evidence is not well represented by short chunks.

Immediate recommendation after Stage121:

```text
review changed cases first
then design a Stage123 first-stage recall expansion protocol
```

The reason is simple: Stage121 improved screening safety but did not raise
fixed-pool recall. To make recall materially higher, the next major research
direction should return to the first-stage candidate generator.
