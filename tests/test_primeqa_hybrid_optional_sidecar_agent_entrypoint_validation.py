from __future__ import annotations

from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _rank_score,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint import (
    PrimeQAHybridOptionalSidecarAgentEntrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_entrypoint_validation import (
    InMemoryFrozenCandidatePoolRetriever,
    _decision,
    _dependency_calls_are_exact,
    _entrypoint_trace_contract_violation_count,
    _entrypoint_train_cv_validation,
    _entrypoint_train_fold_reports,
    _source_fingerprint_comparison,
    _stage137_source_parity,
    _summarize_entrypoint_traces,
    _trace_entrypoint_validation,
    write_primeqa_hybrid_optional_sidecar_agent_entrypoint_validation_visualizations,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator_validation import (
    create_stage116_control_runner,
)
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


def test_validation_retriever_returns_one_frozen_pool() -> None:
    sample = _sample()
    question = sample.to_primeqa_question()
    results = _candidate_results(_documents())
    retriever = InMemoryFrozenCandidatePoolRetriever({question.id: results})

    first = retriever.retrieve(question)
    second = retriever.retrieve(question)

    assert tuple(first) == tuple(results)
    assert tuple(second) == tuple(results)
    assert retriever.call_count == 2
    assert retriever.missing_question_count == 0


def test_entrypoint_trace_contract_accepts_real_terminal_path() -> None:
    trace = _validation_trace()
    public_trace = trace.agent_trace.agent_verified_answer.refused
    expected_terminal = "refuse" if public_trace else "complete"

    assert trace.entrypoint_trace_contract_violation_count == 0
    assert trace.terminal_state == expected_terminal
    assert trace.terminal_state_mismatch is False
    assert trace.dependency_call_count_violation is False


def test_entrypoint_trace_contract_rejects_retry_and_wrong_order() -> None:
    payload = _valid_public_trace(terminal="complete", refused=False)
    payload["retry_action_count"] = 1
    payload["action_trace"][2]["action"] = "observe"

    violations = _entrypoint_trace_contract_violation_count(
        public_trace=payload,
        verified_refused=False,
    )

    assert violations == 2


def test_dependency_call_contract_requires_exactly_one_call() -> None:
    payload = _valid_public_trace(terminal="refuse", refused=True)
    assert _dependency_calls_are_exact(payload) is True

    payload["answer_verifier_call_count"] = 2
    assert _dependency_calls_are_exact(payload) is False


def test_split_summary_adds_action_trace_integrity() -> None:
    summary = _summarize_entrypoint_traces([_validation_trace()])

    assert summary["row_count"] == 1
    assert summary["candidate_pool_identity_violation_count"] == 0
    assert summary["entrypoint_trace_contract_violation_count"] == 0
    assert summary["dependency_call_count_violation_count"] == 0
    assert summary["terminal_state_mismatch_count"] == 0
    assert summary["exact_five_transition_trace_rate"] == 1.0
    assert summary["retry_action_count"] == 0
    assert summary["fallback_action_count"] == 0


def test_train_fold_validation_requires_action_trace_integrity() -> None:
    trace = _validation_trace()
    summary = _summarize_entrypoint_traces([trace])
    fold_reports = _entrypoint_train_fold_reports(
        traces=[trace],
        fold_assignments={trace.sample.sample_id: "fold_1"},
    )

    validation = _entrypoint_train_cv_validation(
        train_summary=summary,
        fold_reports=fold_reports,
    )

    assert validation["passed"] is True
    assert validation["checks"]["all_folds_entrypoint_traces_valid"] is True
    assert validation["checks"]["all_folds_have_exact_five_transition_traces"] is True


def test_stage137_source_parity_detects_mismatch() -> None:
    summary = _summarize_entrypoint_traces([_validation_trace()])
    source = {key: summary[key] for key in summary if not key.startswith("entrypoint_")}
    source.update(
        {
            key: summary[key]
            for key in (
                "candidate_pool_identity_violation_count",
                "dependency_call_count_violation_count",
                "terminal_state_mismatch_count",
                "retry_action_count",
                "fallback_action_count",
                "complete_terminal_count",
                "refuse_terminal_count",
                "exact_five_transition_trace_count",
                "exact_five_transition_trace_rate",
            )
            if key in summary
        }
    )
    stage137_summary = {
        "candidate_pool_summary": {"same": True},
        "split_agent_reports": {"train": summary, "dev": summary},
    }

    parity = _stage137_source_parity(
        pool_summary={"same": True},
        split_reports={"train": summary, "dev": summary},
        stage137_summary=stage137_summary,
    )
    assert parity["passed"] is True

    stage137_summary["split_agent_reports"]["dev"] = {**source, "row_count": 2}
    mismatch = _stage137_source_parity(
        pool_summary={"same": True},
        split_reports={"train": summary, "dev": summary},
        stage137_summary=stage137_summary,
    )
    assert mismatch["passed"] is False
    assert "row_count" in mismatch["split_checks"]["dev"]["mismatched_keys"]


def test_source_fingerprint_comparison_uses_sha256() -> None:
    current = {
        key: {"sha256": f"sha-{key}"}
        for key in (
            "stage128_protocol",
            "stage125_protocol",
            "train_split",
            "dev_split",
            "corpus_documents",
            "stage80_dense_cache_report",
        )
    }
    source = {key: dict(value) for key, value in current.items()}
    source["dev_split"] = {"sha256": "different"}

    comparison = _source_fingerprint_comparison(
        current_sources=current,
        stage137_sources=source,
    )

    assert comparison["train_split"]["matches"] is True
    assert comparison["dev_split"]["matches"] is False


def test_decision_opens_only_explicit_optional_runtime_route() -> None:
    decision = _decision(
        [{"name": "all", "passed": True}],
        split_reports={
            "train": {"sidecar_incremental_gold_hit_count": 0},
            "dev": {"sidecar_incremental_gold_hit_count": 0},
        },
    )

    assert decision["optional_entrypoint_implementation_validated"] is True
    assert decision["runtime_action_order_validated"] is True
    assert decision["can_expose_optional_runtime_entrypoint_now"] is True
    assert decision["entrypoint_registered_as_runtime_default"] is False
    assert decision["can_open_final_test_gate_now"] is False
    assert decision["fallback_strategies_enabled"] is False


def test_visualizations_write_seven_public_safe_svgs(tmp_path: Path) -> None:
    trace = _validation_trace()
    summary = _summarize_entrypoint_traces([trace])
    folds = _entrypoint_train_fold_reports(
        traces=[trace],
        fold_assignments={trace.sample.sample_id: "fold_1"},
    )
    report = {
        "train_fold_reports": folds,
        "split_entrypoint_reports": {"train": summary, "dev": summary},
        "stage137_source_parity": {
            "candidate_pool_summary_matches_stage137": True,
            "split_checks": {"train": {"passed": True}, "dev": {"passed": True}},
            "passed": True,
        },
        "decision": {
            "optional_entrypoint_implementation_validated": True,
            "runtime_action_order_validated": True,
            "answer_path_invariance_validated": True,
            "can_expose_optional_runtime_entrypoint_now": True,
        },
        "guard_checks": [{"name": "all", "passed": True}],
    }

    artifacts = write_primeqa_hybrid_optional_sidecar_agent_entrypoint_validation_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(artifacts) == 7
    assert all(
        Path(artifact.path).read_text(encoding="utf-8").startswith("<svg") for artifact in artifacts
    )


def _validation_trace():
    sample = _sample()
    documents = _documents()
    results = _candidate_results(documents)
    retriever = InMemoryFrozenCandidatePoolRetriever({sample.to_primeqa_question().id: results})
    entrypoint = PrimeQAHybridOptionalSidecarAgentEntrypoint(
        candidate_pool_retriever=retriever,
    )
    return _trace_entrypoint_validation(
        sample=sample,
        pool={
            "prefix_pool": [f"prefix-{index:03d}" for index in range(1, 201)],
            "stage128_pool": [result.document.id for result in results],
        },
        documents_by_id=documents,
        control_runner=create_stage116_control_runner(),
        entrypoint=entrypoint,
    )


def _valid_public_trace(*, terminal: str, refused: bool) -> dict[str, object]:
    actions = ["retrieve", "answer", "verify", "observe", terminal]
    previous = ["ready", "retrieve", "answer", "verify", "observe"]
    next_states = ["retrieve", "answer", "verify", "observe", terminal]
    return {
        "entrypoint_id": "stage138_optional_sidecar_agent_entrypoint_v1",
        "action_state_protocol_id": (
            "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_v1"
        ),
        "orchestrator_id": "stage116_primary_plus_stage128_sidecar_agent_orchestrator_v1",
        "action_count": 5,
        "terminal_state": terminal,
        "terminal": True,
        "verified_refused": refused,
        "retriever_call_count": 1,
        "orchestrator_call_count": 1,
        "answer_generator_call_count": 1,
        "answer_verifier_call_count": 1,
        "sidecar_used_for_answer_generation": False,
        "sidecar_used_for_answer_verification": False,
        "sidecar_replaced_primary_context": False,
        "runtime_gold_labels_read": False,
        "test_membership_read": False,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "action_trace": [
            {
                "sequence_number": index,
                "previous_state": previous[index - 1],
                "action": actions[index - 1],
                "next_state": next_states[index - 1],
            }
            for index in range(1, 6)
        ],
    }


def _sample() -> PrimeQAHybridSplitSample:
    return PrimeQAHybridSplitSample(
        split_name="primeqa_hybrid_stage68_v1",
        protocol_version="primeqa_hybrid_split_v1",
        assigned_split="train",
        split_subtype="grouped_random",
        source_split="train",
        sample_id="sample-entrypoint",
        question_id="question-entrypoint",
        question_title="adapter installation",
        question_text="How do I fix special adapter token failure?",
        answerable=True,
        answer="Use the adapter technote.",
        answer_doc_id="append-special",
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )


def _documents() -> dict[str, PrimeQADocument]:
    documents = {
        f"prefix-{index:03d}": PrimeQADocument(
            id=f"prefix-{index:03d}",
            title="adapter reference",
            text=f"ordinary adapter installation reference {index}",
        )
        for index in range(1, 201)
    }
    documents.update(
        {
            "append-special": PrimeQADocument(
                id="append-special",
                title="special adapter token failure",
                text="fix special adapter token failure using the technote",
            ),
            "append-novel": PrimeQADocument(
                id="append-novel",
                title="adapter installation failure",
                text="special token installation workaround",
            ),
            "append-ordinary": PrimeQADocument(
                id="append-ordinary",
                title="adapter guide",
                text="adapter setup reference",
            ),
        }
    )
    return documents


def _candidate_results(
    documents: dict[str, PrimeQADocument],
) -> list[RetrievalResult]:
    identifiers = [
        *[f"prefix-{index:03d}" for index in range(1, 201)],
        "append-special",
        "append-novel",
        "append-ordinary",
    ]
    return [
        RetrievalResult(
            document=documents[document_id],
            score=_rank_score(rank),
            rank=rank,
        )
        for rank, document_id in enumerate(identifiers, start=1)
    ]
