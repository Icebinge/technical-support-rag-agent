from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalConfig,
    DerivedCandidatePoolSearchChannel,
    IndependentCandidatePoolSearchChannel,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_online_candidate_pool_reuses_derived_source_and_profiles_channels() -> None:
    calls = {"base": 0, "dense": 0, "derived": 0}
    documents = _documents()

    def base_search(query: str, top_k: int) -> Sequence[RetrievalResult]:
        calls["base"] += 1
        assert query == "adapter\n\ntoken failure"
        return _results(documents, ("a", "b", "c"))[:top_k]

    def dense_search(query: str, top_k: int) -> Sequence[RetrievalResult]:
        calls["dense"] += 1
        return _results(documents, ("c", "d", "b"))[:top_k]

    def derived_search(
        query: str,
        source_results: Sequence[RetrievalResult],
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        calls["derived"] += 1
        assert [result.document.id for result in source_results] in (
            ["a", "b", "c"],
            ["a", "b"],
        )
        return _results(documents, ("b", "a", "c"))[:top_k]

    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=(
            IndependentCandidatePoolSearchChannel(
                channel_id="base",
                family="lexical",
                weight=1.0,
                searcher=base_search,
            ),
            DerivedCandidatePoolSearchChannel(
                channel_id="derived",
                family="exact_token",
                weight=1.0,
                source_channel_id="base",
                searcher=derived_search,
            ),
            IndependentCandidatePoolSearchChannel(
                channel_id="dense",
                family="dense",
                weight=1.0,
                searcher=dense_search,
            ),
        ),
        config=CandidatePoolRetrievalConfig(
            channel_top_k=3,
            prefix_depth=2,
            target_pool_depth=3,
            rrf_k=60,
        ),
    )

    run = retriever.retrieve_profiled(_question())

    assert calls == {"base": 1, "dense": 1, "derived": 2}
    assert [result.document.id for result in run.results] == ["a", "b", "c"]
    assert [result.rank for result in run.results] == [1, 2, 3]
    assert len(run.profile.channel_timings) == 3
    assert run.profile.channel_timings[1].derived_from_existing_results is True
    assert run.profile.channel_timings[0].derived_from_existing_results is False
    assert run.profile.total_seconds >= 0


def test_derived_prefix_is_recomputed_from_source_prefix() -> None:
    documents = _documents()
    observed_source_ids = []

    def derived_search(
        query: str,
        source_results: Sequence[RetrievalResult],
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        _ = query
        source_ids = tuple(result.document.id for result in source_results)
        observed_source_ids.append(source_ids)
        order = ("d", "c", "b", "a") if len(source_ids) == 4 else ("b", "a")
        return _results(documents, order)[:top_k]

    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=(
            IndependentCandidatePoolSearchChannel(
                channel_id="base",
                family="lexical",
                weight=1.0,
                searcher=lambda query, top_k: _results(
                    documents,
                    ("a", "b", "c", "d"),
                )[:top_k],
            ),
            DerivedCandidatePoolSearchChannel(
                channel_id="derived",
                family="exact_token",
                weight=2.0,
                source_channel_id="base",
                searcher=derived_search,
            ),
        ),
        config=CandidatePoolRetrievalConfig(
            channel_top_k=4,
            prefix_depth=2,
            target_pool_depth=4,
            rrf_k=60,
        ),
    )

    run = retriever.retrieve_profiled(_question())

    assert observed_source_ids == [("a", "b", "c", "d"), ("a", "b")]
    assert [result.document.id for result in run.results[:2]] == ["b", "a"]


def test_derived_channel_rejects_unresolved_dependency() -> None:
    channel = DerivedCandidatePoolSearchChannel(
        channel_id="derived",
        family="exact_token",
        weight=1.0,
        source_channel_id="base",
        searcher=lambda query, results, top_k: results,
    )

    with pytest.raises(RuntimeError, match="source channel must run first"):
        channel.search("query", top_k=3, resolved_results={})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channel_top_k": 0}, "channel_top_k"),
        ({"prefix_depth": 0}, "prefix_depth"),
        ({"target_pool_depth": 1}, "target_pool_depth"),
        ({"channel_top_k": 2}, "channel_top_k must cover"),
        ({"rrf_k": 0}, "rrf_k"),
    ],
)
def test_candidate_pool_config_rejects_invalid_dimensions(
    kwargs: Mapping[str, int],
    message: str,
) -> None:
    values = {
        "channel_top_k": 3,
        "prefix_depth": 2,
        "target_pool_depth": 3,
        "rrf_k": 60,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        CandidatePoolRetrievalConfig(**values)


def test_online_candidate_pool_rejects_duplicate_channel_ids() -> None:
    channel = IndependentCandidatePoolSearchChannel(
        channel_id="same",
        family="lexical",
        weight=1.0,
        searcher=lambda query, top_k: [],
    )

    with pytest.raises(ValueError, match="unique"):
        PrimeQAHybridOnlineCandidatePoolRetriever(
            channels=(channel, channel),
            config=CandidatePoolRetrievalConfig(
                channel_top_k=3,
                prefix_depth=2,
                target_pool_depth=3,
                rrf_k=60,
            ),
        )


def test_online_candidate_pool_rejects_derived_channel_before_source() -> None:
    derived = DerivedCandidatePoolSearchChannel(
        channel_id="derived",
        family="exact_token",
        weight=1.0,
        source_channel_id="base",
        searcher=lambda query, results, top_k: results,
    )

    with pytest.raises(ValueError, match="must follow"):
        PrimeQAHybridOnlineCandidatePoolRetriever(
            channels=(derived,),
            config=CandidatePoolRetrievalConfig(
                channel_top_k=3,
                prefix_depth=2,
                target_pool_depth=3,
                rrf_k=60,
            ),
        )


def _question() -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="private",
        title="adapter",
        text="token failure",
        answer="",
        answerable=False,
        answer_doc_id=None,
    )


def _documents() -> dict[str, PrimeQADocument]:
    return {
        doc_id: PrimeQADocument(id=doc_id, title=doc_id, text=f"document {doc_id}")
        for doc_id in ("a", "b", "c", "d")
    }


def _results(
    documents: Mapping[str, PrimeQADocument],
    doc_ids: Sequence[str],
) -> list[RetrievalResult]:
    return [
        RetrievalResult(document=documents[doc_id], score=1.0 / rank, rank=rank)
        for rank, doc_id in enumerate(doc_ids, start=1)
    ]
