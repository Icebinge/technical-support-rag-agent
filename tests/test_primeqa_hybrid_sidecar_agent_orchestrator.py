from __future__ import annotations

import json

import pytest

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator import (
    SidecarAgentConsumerPolicy,
    _forbidden_keys_found,
    create_primeqa_hybrid_sidecar_agent_orchestrator,
    sidecar_agent_orchestrator_contract,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_orchestrator_keeps_sidecar_out_of_answer_and_verification_paths() -> None:
    orchestrator = create_primeqa_hybrid_sidecar_agent_orchestrator()
    run = orchestrator.run(
        question=_question(answer_doc_id="append-special"),
        candidate_pool_results=_candidate_results(),
    )

    sidecar_handles = {
        row.runtime_content_handle for row in run.observation_bundle.sidecar_observations
    }
    cited_handles = {citation.document_id for citation in run.original_answer.citations}
    trace = run.public_safe_trace

    assert sidecar_handles
    assert not sidecar_handles & cited_handles
    assert trace.sidecar_used_for_answer_generation is False
    assert trace.sidecar_used_for_answer_verification is False
    assert trace.sidecar_replaced_primary_context is False
    assert all(
        row.selected_for_answer_generation is False
        and row.selected_for_answer_verification is False
        for row in trace.sidecar_selection_trace
    )


def test_public_trace_contains_metadata_without_private_runtime_values() -> None:
    question = _question(answer_doc_id="append-special")
    results = _candidate_results()
    run = create_primeqa_hybrid_sidecar_agent_orchestrator().run(
        question=question,
        candidate_pool_results=results,
    )

    payload = run.public_safe_trace.to_public_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert _forbidden_keys_found(payload) == set()
    assert question.id not in serialized
    assert question.text not in serialized
    assert all(result.document.id not in serialized for result in results)
    assert payload["sidecar_effectiveness_status"] == "diagnostic_only_unproven"
    assert payload["runtime_gold_labels_read"] is False
    assert payload["test_membership_read"] is False


def test_orchestrator_trace_does_not_depend_on_gold_label() -> None:
    orchestrator = create_primeqa_hybrid_sidecar_agent_orchestrator()
    results = _candidate_results()
    first = orchestrator.run(
        question=_question(answer_doc_id="append-special"),
        candidate_pool_results=results,
    )
    second = orchestrator.run(
        question=_question(answer_doc_id="different-gold"),
        candidate_pool_results=results,
    )

    assert first.public_safe_trace == second.public_safe_trace
    assert first.original_answer == second.original_answer
    assert first.verified_answer == second.verified_answer


def test_consumer_policy_rejects_answer_path_or_effectiveness_expansion() -> None:
    with pytest.raises(ValueError, match="answer generation"):
        SidecarAgentConsumerPolicy(answer_generation_allowed=True)
    with pytest.raises(ValueError, match="answer verification"):
        SidecarAgentConsumerPolicy(answer_verification_context_allowed=True)
    with pytest.raises(ValueError, match="diagnostic_only_unproven"):
        SidecarAgentConsumerPolicy(citation_verification_probe_mode="enabled")


def test_orchestrator_contract_keeps_test_runtime_and_fallback_closed() -> None:
    contract = sidecar_agent_orchestrator_contract()
    policy = contract["consumer_policy"]
    trace = contract["public_trace"]

    assert contract["channel_routing"]["answer_generation"] == ("stage116_primary_answer_context")
    assert contract["channel_routing"]["answer_verification"] == (
        "stage116_prefix_verification_context"
    )
    assert policy["answer_generation_allowed"] is False
    assert policy["answer_verification_context_allowed"] is False
    assert policy["runtime_defaultization_allowed"] is False
    assert policy["fallback_strategy_allowed"] is False
    assert trace["contains_gold_labels"] is False
    assert trace["contains_test_membership"] is False


def test_orchestrator_rejects_duplicate_or_out_of_protocol_ranks() -> None:
    orchestrator = create_primeqa_hybrid_sidecar_agent_orchestrator()
    results = _candidate_results()
    duplicate = [*results, results[-1]]
    out_of_protocol = [
        *results,
        RetrievalResult(
            document=PrimeQADocument(id="outside", title="outside", text="outside"),
            score=0.1,
            rank=401,
        ),
    ]

    with pytest.raises(ValueError, match="unique"):
        orchestrator.run(
            question=_question(answer_doc_id="append-special"),
            candidate_pool_results=duplicate,
        )
    with pytest.raises(ValueError, match="depth 400"):
        orchestrator.run(
            question=_question(answer_doc_id="append-special"),
            candidate_pool_results=out_of_protocol,
        )


def _question(*, answer_doc_id: str) -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="q1",
        title="adapter installation",
        text="How do I fix special adapter token failure?",
        answer="Use the adapter technote.",
        answerable=True,
        answer_doc_id=answer_doc_id,
    )


def _candidate_results() -> list[RetrievalResult]:
    results = [
        RetrievalResult(
            document=PrimeQADocument(
                id=f"prefix-{index:03d}",
                title="adapter reference",
                text=(
                    "The adapter installation reference describes ordinary setup "
                    f"steps for configuration {index}."
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
            text=(
                "The special adapter token failure is fixed by applying the "
                "dedicated technote procedure before installation."
            ),
        ),
        PrimeQADocument(
            id="append-novel",
            title="adapter installation failure",
            text="The special token installation workaround resets the adapter.",
        ),
        PrimeQADocument(
            id="append-ordinary",
            title="adapter guide",
            text="The adapter setup reference contains ordinary setup guidance.",
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
