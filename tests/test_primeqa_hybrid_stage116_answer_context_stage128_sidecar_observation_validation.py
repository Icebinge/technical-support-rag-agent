from __future__ import annotations

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _DocumentEvidenceShortlister,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
    _check,
    _decision,
    _forbidden_keys_found,
    _result_signature,
    _summarize_observation_traces,
    _trace_observation,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
)


def test_adapter_preserves_stage116_primary_context_and_isolates_sidecar() -> None:
    question = _question(answer_doc_id="append-special")
    results = _candidate_results()
    control = _DocumentEvidenceShortlister().shortlist(
        question=question,
        candidates=results[:200],
        top_k=10,
    )

    bundle = PrimeQAHybridSidecarObservationAdapter().observe(
        question=question,
        candidate_pool_results=results,
    )

    assert _result_signature(bundle.answer_context_for_generation()) == _result_signature(control)
    assert len(bundle.primary_context_records) == 10
    assert len(bundle.sidecar_observations) == 3
    assert all(
        record.sidecar_score_summary.retrieval_rank > 200 for record in bundle.sidecar_observations
    )
    assert not (
        {result.document.id for result in bundle.answer_context_for_generation()}
        & {record.runtime_content_handle for record in bundle.sidecar_observations}
    )


def test_adapter_signals_do_not_depend_on_gold_labels() -> None:
    results = _candidate_results()
    first = PrimeQAHybridSidecarObservationAdapter().observe(
        question=_question(answer_doc_id="append-special"),
        candidate_pool_results=results,
    )
    second = PrimeQAHybridSidecarObservationAdapter().observe(
        question=_question(answer_doc_id="different-gold"),
        candidate_pool_results=results,
    )

    assert first.primary_context_records == second.primary_context_records
    assert first.sidecar_observations == second.sidecar_observations


def test_trace_reports_real_gold_opportunity_without_sidecar_leak() -> None:
    documents = {result.document.id: result.document for result in _candidate_results()}
    sample = _sample(answer_doc_id="append-special")
    trace = _trace_observation(
        sample=sample,
        pool={
            "prefix_pool": [f"prefix-{index:03d}" for index in range(1, 201)],
            "stage128_pool": [
                *[f"prefix-{index:03d}" for index in range(1, 201)],
                "append-special",
                "append-novel",
                "append-ordinary",
            ],
        },
        documents_by_id=documents,
        adapter=PrimeQAHybridSidecarObservationAdapter(),
        stage116_shortlister=_DocumentEvidenceShortlister(),
    )

    summary = _summarize_observation_traces([trace])

    assert summary["primary_context_identity_violation_count"] == 0
    assert summary["sidecar_answer_context_leak_count"] == 0
    assert summary["append_pool_incremental_gold_hit_count"] == 1
    assert summary["sidecar_incremental_gold_hit_count"] == 1


def test_decision_opens_only_train_dev_agent_implementation() -> None:
    decision = _decision([_check(name="integrity", passed=True, observed=0, expected=0)])

    assert decision["sidecar_observation_protocol_validated"] is True
    assert decision["can_implement_train_dev_agent_orchestrator_now"] is True
    assert decision["can_run_final_test_metrics_now"] is False
    assert decision["runtime_defaultization_allowed_now"] is False
    assert decision["fallback_strategies_enabled"] is False
    assert decision["recommended_next_direction"].startswith("implement_stage116")


def test_decision_blocks_agent_implementation_when_guard_fails() -> None:
    decision = _decision([_check(name="integrity", passed=False, observed=1, expected=0)])

    assert decision["sidecar_observation_protocol_validated"] is False
    assert decision["can_implement_train_dev_agent_orchestrator_now"] is False
    assert decision["failed_checks"] == ["integrity"]


def test_public_safe_detector_rejects_private_keys() -> None:
    assert _forbidden_keys_found({"metric": {"count": 1}}) == set()
    assert _forbidden_keys_found({"question_text": "private"}) == {"question_text"}


def _question(*, answer_doc_id: str) -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="q1",
        title="adapter installation",
        text="How do I fix special adapter token failure?",
        answer="Use the adapter technote.",
        answerable=True,
        answer_doc_id=answer_doc_id,
    )


def _sample(*, answer_doc_id: str) -> PrimeQAHybridSplitSample:
    return PrimeQAHybridSplitSample(
        split_name="primeqa_hybrid_stage68_v1",
        protocol_version="primeqa_hybrid_split_v1",
        assigned_split="train",
        split_subtype="grouped_random",
        source_split="train",
        sample_id="sample-1",
        question_id="source-1",
        question_title="adapter installation",
        question_text="How do I fix special adapter token failure?",
        answerable=True,
        answer="Use the adapter technote.",
        answer_doc_id=answer_doc_id,
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )


def _candidate_results() -> list[RetrievalResult]:
    results = [
        RetrievalResult(
            document=PrimeQADocument(
                id=f"prefix-{index:03d}",
                title="adapter reference",
                text=f"ordinary adapter installation reference {index}",
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
            text="fix special adapter token failure using the technote",
        ),
        PrimeQADocument(
            id="append-novel",
            title="adapter installation failure",
            text="special token installation workaround",
        ),
        PrimeQADocument(
            id="append-ordinary",
            title="adapter guide",
            text="adapter setup reference",
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
