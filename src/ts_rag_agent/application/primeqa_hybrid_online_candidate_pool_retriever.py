from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint import (
    CandidatePoolRetrieverPort,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult

ChannelSearcher = Callable[[str, int], Sequence[RetrievalResult]]
DerivedChannelSearcher = Callable[
    [str, Sequence[RetrievalResult], int],
    Sequence[RetrievalResult],
]


class CandidatePoolSearchChannel(Protocol):
    """One long-lived retrieval route in the frozen candidate-pool graph."""

    channel_id: str
    family: str
    weight: float

    def search(
        self,
        query: str,
        *,
        top_k: int,
        resolved_results: Mapping[str, Sequence[RetrievalResult]],
    ) -> Sequence[RetrievalResult]: ...


@dataclass(frozen=True)
class IndependentCandidatePoolSearchChannel:
    """A route backed directly by one initialized long-lived retriever."""

    channel_id: str
    family: str
    weight: float
    searcher: ChannelSearcher

    def search(
        self,
        query: str,
        *,
        top_k: int,
        resolved_results: Mapping[str, Sequence[RetrievalResult]],
    ) -> Sequence[RetrievalResult]:
        _ = resolved_results
        return self.searcher(query, top_k)


@dataclass(frozen=True)
class DerivedCandidatePoolSearchChannel:
    """A route derived from an earlier route without repeating its base search."""

    channel_id: str
    family: str
    weight: float
    source_channel_id: str
    searcher: DerivedChannelSearcher

    def search(
        self,
        query: str,
        *,
        top_k: int,
        resolved_results: Mapping[str, Sequence[RetrievalResult]],
    ) -> Sequence[RetrievalResult]:
        try:
            source_results = resolved_results[self.source_channel_id]
        except KeyError as error:
            raise RuntimeError(
                f"source channel must run first: {self.source_channel_id}"
            ) from error
        return self.searcher(query, source_results, top_k)


@dataclass(frozen=True)
class CandidatePoolRetrievalConfig:
    """Frozen Stage128 candidate-pool dimensions and RRF settings."""

    channel_top_k: int
    prefix_depth: int
    target_pool_depth: int
    rrf_k: int

    def __post_init__(self) -> None:
        if self.channel_top_k <= 0:
            raise ValueError("channel_top_k must be positive")
        if self.prefix_depth <= 0:
            raise ValueError("prefix_depth must be positive")
        if self.target_pool_depth < self.prefix_depth:
            raise ValueError("target_pool_depth must be at least prefix_depth")
        if self.channel_top_k < self.target_pool_depth:
            raise ValueError("channel_top_k must cover target_pool_depth")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")


@dataclass(frozen=True)
class CandidatePoolChannelTiming:
    channel_id: str
    family: str
    derived_from_existing_results: bool
    duration_seconds: float
    result_count: int


@dataclass(frozen=True)
class CandidatePoolRetrievalProfile:
    channel_timings: tuple[CandidatePoolChannelTiming, ...]
    fusion_seconds: float
    materialization_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class CandidatePoolRetrievalRun:
    results: tuple[RetrievalResult, ...]
    profile: CandidatePoolRetrievalProfile


class PrimeQAHybridOnlineCandidatePoolRetriever(CandidatePoolRetrieverPort):
    """Build one query-specific pool while reusing initialized channel indexes."""

    def __init__(
        self,
        *,
        channels: Sequence[CandidatePoolSearchChannel],
        config: CandidatePoolRetrievalConfig,
    ) -> None:
        if not channels:
            raise ValueError("at least one candidate-pool channel is required")
        channel_ids = [channel.channel_id for channel in channels]
        if len(channel_ids) != len(set(channel_ids)):
            raise ValueError("candidate-pool channel ids must be unique")
        resolved_channel_ids = set()
        for channel in channels:
            if (
                isinstance(channel, DerivedCandidatePoolSearchChannel)
                and channel.source_channel_id not in resolved_channel_ids
            ):
                raise ValueError(
                    "derived candidate-pool channels must follow their source channel: "
                    f"{channel.channel_id} -> {channel.source_channel_id}"
                )
            resolved_channel_ids.add(channel.channel_id)
        self._channels = tuple(channels)
        self._config = config

    def retrieve(self, question: PrimeQAQuestion) -> Sequence[RetrievalResult]:
        return self.retrieve_profiled(question).results

    def retrieve_profiled(self, question: PrimeQAQuestion) -> CandidatePoolRetrievalRun:
        started_at = time.perf_counter()
        query = question.full_question
        results_by_channel: dict[str, Sequence[RetrievalResult]] = {}
        channel_timings = []

        for channel in self._channels:
            channel_started_at = time.perf_counter()
            results = tuple(
                channel.search(
                    query,
                    top_k=self._config.channel_top_k,
                    resolved_results=results_by_channel,
                )
            )
            channel_finished_at = time.perf_counter()
            results_by_channel[channel.channel_id] = results
            channel_timings.append(
                CandidatePoolChannelTiming(
                    channel_id=channel.channel_id,
                    family=channel.family,
                    derived_from_existing_results=isinstance(
                        channel,
                        DerivedCandidatePoolSearchChannel,
                    ),
                    duration_seconds=channel_finished_at - channel_started_at,
                    result_count=len(results),
                )
            )

        fusion_started_at = time.perf_counter()
        prefix_results_by_channel = {
            channel.channel_id: results_by_channel[channel.channel_id][: self._config.prefix_depth]
            for channel in self._channels
        }
        prefix = _weighted_rrf_rank(
            channels=self._channels,
            results_by_channel=prefix_results_by_channel,
            rrf_k=self._config.rrf_k,
        )[: self._config.prefix_depth]
        append_source_ranked = _weighted_rrf_rank(
            channels=self._channels,
            results_by_channel=results_by_channel,
            rrf_k=self._config.rrf_k,
        )[: self._config.target_pool_depth]
        ranked_doc_ids = _prefix_preserving_pool(
            prefix=prefix,
            append_source_ranked=append_source_ranked,
            target_pool_depth=self._config.target_pool_depth,
        )
        fused_at = time.perf_counter()
        documents_by_id = _documents_by_id(results_by_channel)
        candidate_pool = tuple(
            RetrievalResult(
                document=documents_by_id[doc_id],
                score=_rank_score(rank),
                rank=rank,
            )
            for rank, doc_id in enumerate(ranked_doc_ids, start=1)
        )
        finished_at = time.perf_counter()
        return CandidatePoolRetrievalRun(
            results=candidate_pool,
            profile=CandidatePoolRetrievalProfile(
                channel_timings=tuple(channel_timings),
                fusion_seconds=fused_at - fusion_started_at,
                materialization_seconds=finished_at - fused_at,
                total_seconds=finished_at - started_at,
            ),
        )


def _weighted_rrf_rank(
    *,
    channels: Sequence[CandidatePoolSearchChannel],
    results_by_channel: Mapping[str, Sequence[RetrievalResult]],
    rrf_k: int,
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for channel in channels:
        for result in results_by_channel[channel.channel_id]:
            scores[result.document.id] += channel.weight / (rrf_k + result.rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def _documents_by_id(
    results_by_channel: Mapping[str, Sequence[RetrievalResult]],
) -> dict[str, PrimeQADocument]:
    return {
        result.document.id: result.document
        for results in results_by_channel.values()
        for result in results
    }


def _prefix_preserving_pool(
    *,
    prefix: Sequence[str],
    append_source_ranked: Sequence[str],
    target_pool_depth: int,
) -> list[str]:
    ranked = list(prefix)
    seen = set(ranked)
    for doc_id in append_source_ranked:
        if doc_id in seen:
            continue
        ranked.append(doc_id)
        seen.add(doc_id)
        if len(ranked) >= target_pool_depth:
            break
    return ranked[:target_pool_depth]


def _rank_score(rank: int) -> float:
    return round(1.0 / math.log2(rank + 1), 8)
