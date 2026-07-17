from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Condition

import numpy as np
import pytest

from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
    PrimeQAHybridConcurrentCapacityExceededError,
    concurrent_sidecar_runtime_contract,
    create_primeqa_hybrid_concurrent_sidecar_agent_runtime,
)
from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_retriever import (
    CandidatePoolRetrievalConfig,
    IndependentCandidatePoolSearchChannel,
    PrimeQAHybridOnlineCandidatePoolRetriever,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridRuntimeResourceSummary,
    PrimeQAHybridSharedRuntimeResources,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


class FourRequestSearchBarrier:
    def __init__(self) -> None:
        self._barrier = Barrier(4)

    def __call__(self, query: str, top_k: int):
        self._barrier.wait()
        prefix = query.split()[0].lower()
        return _candidate_results(prefix)[:top_k]


class HoldFourAdmissions:
    def __init__(self) -> None:
        self._condition = Condition()
        self._admitted = 0
        self._released = False

    def on_admitted(self, in_flight_at_admission: int) -> None:
        assert 1 <= in_flight_at_admission <= 4
        with self._condition:
            self._admitted += 1
            self._condition.notify_all()
            while not self._released:
                self._condition.wait()

    def wait_until_full(self) -> None:
        with self._condition:
            while self._admitted < 4:
                self._condition.wait()

    def release(self) -> None:
        with self._condition:
            self._released = True
            self._condition.notify_all()


def test_four_requests_run_concurrently_with_request_local_profiles() -> None:
    runtime = _runtime(searcher=FourRequestSearchBarrier())
    tokens = ("alpha", "beta", "gamma", "delta")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                runtime.run,
                _question(token),
                arrival_pattern=ConcurrentArrivalPattern.SYNCHRONIZED,
            )
            for token in tokens
        ]
        runs = [future.result() for future in futures]

    for token, run in zip(tokens, runs, strict=True):
        assert len(run.candidate_pool_results) == 400
        assert all(
            result.document.id.startswith(f"{token}-") for result in run.candidate_pool_results
        )
        trace = run.public_safe_trace.to_public_dict()
        assert set(trace) == set(
            concurrent_sidecar_runtime_contract()["request_trace_allowed_fields"]
        )
        assert trace["arrival_pattern"] == "synchronized_four_request_burst"
        assert trace["admission_state"] == "admitted"
        assert trace["candidate_pool_depth"] == 400
        assert trace["end_to_end_latency_ms"] >= trace["retrieval_latency_ms"]

    counters = runtime.counters()
    assert counters.admission_attempt_count == 4
    assert counters.admitted_request_count == 4
    assert counters.capacity_rejected_request_count == 0
    assert counters.downstream_request_count == 4
    assert counters.completed_request_count == 4
    assert counters.failed_request_count == 0
    assert counters.current_in_flight == 0
    assert counters.max_observed_in_flight == 4


def test_fifth_request_is_typed_rejection_before_downstream() -> None:
    runtime = _runtime(searcher=lambda query, top_k: _candidate_results("shared")[:top_k])
    probe = HoldFourAdmissions()

    with ThreadPoolExecutor(max_workers=4) as executor:
        admitted = [
            executor.submit(
                runtime.run,
                _question(f"held{index}"),
                arrival_pattern=ConcurrentArrivalPattern.OVERLOAD_PROBE,
                admission_probe=probe,
            )
            for index in range(4)
        ]
        probe.wait_until_full()
        before = runtime.counters()
        with pytest.raises(PrimeQAHybridConcurrentCapacityExceededError) as captured:
            runtime.run(
                _question("rejected"),
                arrival_pattern=ConcurrentArrivalPattern.OVERLOAD_PROBE,
            )
        after_rejection = runtime.counters()
        probe.release()
        runs = [future.result() for future in admitted]

    rejection = captured.value.public_safe_trace.to_public_dict()
    assert rejection["admission_state"] == "rejected_capacity"
    assert rejection["terminal_state"] == "capacity_rejected"
    assert rejection["in_flight_at_admission"] == 4
    assert rejection["candidate_pool_depth"] == 0
    assert rejection["retrieval_latency_ms"] == 0.0
    assert before.downstream_request_count == 0
    assert after_rejection.downstream_request_count == 0
    assert len(runs) == 4

    counters = runtime.counters()
    assert counters.admission_attempt_count == 5
    assert counters.admitted_request_count == 4
    assert counters.capacity_rejected_request_count == 1
    assert counters.downstream_request_count == 4
    assert counters.completed_request_count == 4
    assert counters.failed_request_count == 0
    assert counters.current_in_flight == 0
    assert counters.max_observed_in_flight == 4


def test_downstream_error_propagates_and_releases_admission_permit() -> None:
    def searcher(query: str, top_k: int):
        if query.startswith("fail"):
            raise RuntimeError("retrieval failed")
        return _candidate_results("recovered")[:top_k]

    runtime = _runtime(searcher=searcher)

    with pytest.raises(RuntimeError, match="retrieval failed"):
        runtime.run(
            _question("fail"),
            arrival_pattern=ConcurrentArrivalPattern.SYNCHRONIZED,
        )

    recovered = runtime.run(
        _question("success"),
        arrival_pattern=ConcurrentArrivalPattern.SYNCHRONIZED,
    )

    assert len(recovered.candidate_pool_results) == 400
    counters = runtime.counters()
    assert counters.admitted_request_count == 2
    assert counters.failed_request_count == 1
    assert counters.completed_request_count == 1
    assert counters.current_in_flight == 0


def test_concurrent_runtime_contract_keeps_closed_boundaries() -> None:
    contract = concurrent_sidecar_runtime_contract()

    assert contract["max_in_flight"] == 4
    assert contract["admission_mode"] == "nonblocking_bounded_semaphore"
    assert contract["request_local_retrieval_profile"] is True
    assert contract["shared_pending_retrieval_profile_allowed"] is False
    assert contract["registered_as_runtime_default"] is False
    assert contract["test_access_allowed"] is False
    assert contract["queue_actions_allowed"] is False
    assert contract["retry_actions_allowed"] is False
    assert contract["fallback_strategies_allowed"] is False


def _runtime(*, searcher):
    retriever = PrimeQAHybridOnlineCandidatePoolRetriever(
        channels=(
            IndependentCandidatePoolSearchChannel(
                channel_id="test",
                family="test",
                weight=1.0,
                searcher=searcher,
            ),
        ),
        config=CandidatePoolRetrievalConfig(
            channel_top_k=400,
            prefix_depth=200,
            target_pool_depth=400,
            rrf_k=60,
        ),
    )
    return create_primeqa_hybrid_concurrent_sidecar_agent_runtime(
        shared_resources=PrimeQAHybridSharedRuntimeResources(
            candidate_pool_retriever=retriever,
            summary=PrimeQAHybridRuntimeResourceSummary(
                dense_model_count=2,
                dense_embedding_cache_count=2,
                lexical_index_count=4,
                derived_route_count=1,
                candidate_pool_retriever_instance_count=1,
                optional_entrypoint_instance_count=1,
            ),
        )
    )


def _question(token: str) -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id=f"private-{token}",
        title=f"{token} adapter token installation",
        text="How do I repair the adapter token installation failure?",
        answer="Apply the adapter token procedure.",
        answerable=True,
        answer_doc_id=f"{token}-001",
    )


def _candidate_results(prefix: str) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"{prefix}-{index:03d}",
                title=f"{prefix} adapter token installation",
                text=f"Apply the adapter token procedure for configuration {index}.",
            ),
            score=float(np.float64(1 / index)),
            rank=index,
        )
        for index in range(1, 401)
    )
