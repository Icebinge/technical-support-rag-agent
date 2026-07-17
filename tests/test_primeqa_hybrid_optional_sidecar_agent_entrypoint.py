from __future__ import annotations

from collections.abc import Sequence

import pytest

from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint import (
    FrozenSidecarAgentOrchestratorExecutionFactory,
    PrimeQAHybridOptionalSidecarAgentEntrypoint,
    _forbidden_keys_found,
    optional_sidecar_agent_entrypoint_contract,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


class RecordingCandidatePoolRetriever:
    def __init__(self, results: Sequence[RetrievalResult]) -> None:
        self._results = tuple(results)
        self.call_count = 0

    def retrieve(self, question: PrimeQAQuestion) -> Sequence[RetrievalResult]:
        self.call_count += 1
        return self._results


class FailingCandidatePoolRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    def retrieve(self, question: PrimeQAQuestion) -> Sequence[RetrievalResult]:
        self.call_count += 1
        raise RuntimeError("retrieval failed")


def test_entrypoint_executes_one_accepted_action_path() -> None:
    retriever = RecordingCandidatePoolRetriever(_candidate_results())
    entrypoint = PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=retriever,
        orchestrator_factory=FrozenSidecarAgentOrchestratorExecutionFactory(min_evidence_score=0.0),
    )

    run = entrypoint.run(_question())
    trace = run.public_safe_trace.to_public_dict()

    assert retriever.call_count == 1
    assert trace["action_count"] == 5
    assert trace["terminal_state"] == "complete"
    assert trace["terminal"] is True
    assert trace["retriever_call_count"] == 1
    assert trace["orchestrator_call_count"] == 1
    assert trace["answer_generator_call_count"] == 1
    assert trace["answer_verifier_call_count"] == 1
    assert [row["action"] for row in trace["action_trace"]] == [
        "retrieve",
        "answer",
        "verify",
        "observe",
        "complete",
    ]
    assert trace["retry_action_count"] == 0
    assert trace["fallback_action_count"] == 0


def test_entrypoint_records_real_generator_and_verifier_contexts() -> None:
    results = _candidate_results()
    entrypoint = PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=RecordingCandidatePoolRetriever(results),
    )

    run = entrypoint.run(_question())

    assert run.generation_context_results
    assert max(result.rank for result in run.generation_context_results) <= 200
    assert len(run.verification_context_results) == 200
    assert max(result.rank for result in run.verification_context_results) == 200
    sidecar_handles = {
        record.runtime_content_handle
        for record in run.agent_run.observation_bundle.sidecar_observations
    }
    assert not sidecar_handles & {result.document.id for result in run.generation_context_results}
    assert not sidecar_handles & {result.document.id for result in run.verification_context_results}
    assert run.public_safe_trace.terminal_state == (
        "refuse" if run.verified_answer.refused else "complete"
    )


def test_entrypoint_uses_a_fresh_state_machine_per_request() -> None:
    retriever = RecordingCandidatePoolRetriever(_candidate_results())
    entrypoint = PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=retriever,
    )

    first = entrypoint.run(_question())
    second = entrypoint.run(_question())

    assert retriever.call_count == 2
    assert first.public_safe_trace.action_trace[0]["sequence_number"] == 1
    assert second.public_safe_trace.action_trace[0]["sequence_number"] == 1
    assert first.public_safe_trace.action_trace == second.public_safe_trace.action_trace


def test_retrieval_error_propagates_without_retry_or_fallback() -> None:
    retriever = FailingCandidatePoolRetriever()
    entrypoint = PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=retriever,
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        entrypoint.run(_question())

    assert retriever.call_count == 1


def test_invalid_candidate_pool_propagates_without_retry() -> None:
    retriever = RecordingCandidatePoolRetriever(
        [
            RetrievalResult(
                document=PrimeQADocument(id="bad", title="bad", text="bad"),
                score=1.0,
                rank=401,
            )
        ]
    )
    entrypoint = PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=retriever,
        orchestrator_factory=FrozenSidecarAgentOrchestratorExecutionFactory(),
    )

    with pytest.raises(ValueError, match="depth 400"):
        entrypoint.run(_question())

    assert retriever.call_count == 1


def test_public_trace_contains_no_private_runtime_values() -> None:
    question = _question()
    results = _candidate_results()
    run = PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=RecordingCandidatePoolRetriever(results),
    ).run(question)
    payload = run.public_safe_trace.to_public_dict()
    serialized = str(payload)

    assert _forbidden_keys_found(payload) == set()
    assert question.id not in serialized
    assert question.text not in serialized
    assert all(result.document.id not in serialized for result in results)
    assert payload["runtime_gold_labels_read"] is False
    assert payload["test_membership_read"] is False


def test_entrypoint_contract_keeps_runtime_test_retry_and_fallback_closed() -> None:
    contract = optional_sidecar_agent_entrypoint_contract()
    permissions = contract["permissions"]

    assert contract["action_state_protocol_id"] == (
        "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_v1"
    )
    assert permissions["optional_entrypoint"] is True
    assert permissions["registered_as_runtime_default"] is False
    assert permissions["sidecar_answer_generation_allowed"] is False
    assert permissions["sidecar_verification_context_allowed"] is False
    assert permissions["retry_actions_allowed"] is False
    assert permissions["fallback_strategies_allowed"] is False
    assert permissions["test_access_allowed"] is False


def _question() -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="q-private",
        title="adapter installation",
        text="How do I fix the adapter installation token failure?",
        answer="Apply the adapter technote procedure.",
        answerable=True,
        answer_doc_id="append-special",
    )


def _candidate_results() -> list[RetrievalResult]:
    results = [
        RetrievalResult(
            document=PrimeQADocument(
                id=f"prefix-{index:03d}",
                title="adapter installation reference",
                text=(
                    "The adapter installation token procedure describes the fix "
                    f"for configuration {index}."
                ),
            ),
            score=round(1.0 / (index + 1), 8),
            rank=index,
        )
        for index in range(1, 201)
    ]
    append_documents = [
        PrimeQADocument(
            id="append-special",
            title="special adapter token failure",
            text="Apply the dedicated adapter technote procedure before installation.",
        ),
        PrimeQADocument(
            id="append-novel",
            title="adapter installation failure",
            text="The token workaround resets the adapter before installation.",
        ),
        PrimeQADocument(
            id="append-ordinary",
            title="adapter guide",
            text="The adapter guide contains ordinary setup guidance.",
        ),
    ]
    results.extend(
        RetrievalResult(
            document=document,
            score=round(1.0 / (rank + 1), 8),
            rank=rank,
        )
        for rank, document in enumerate(append_documents, start=201)
    )
    return results
