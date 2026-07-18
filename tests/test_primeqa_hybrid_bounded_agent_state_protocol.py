from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    BoundedDynamicToolSelectionPolicy,
    CompletedThreadTurn,
    DynamicDecisionAction,
    DynamicToolSelectionEvidence,
    DynamicToolSelectionState,
    ThreadStateLimits,
    ThreadStatePolicyViolationError,
    VolatileThreadStateLedger,
    bounded_decision_contract,
    freeze_primeqa_hybrid_bounded_agent_state_protocol,
    volatile_thread_state_contract,
    write_primeqa_hybrid_bounded_agent_state_visualizations,
)

_ROOT = Path(__file__).resolve().parents[1]
_STAGE155 = _ROOT / "artifacts" / "primeqa_hybrid_agent_runtime_observability_stage155.json"


def test_bounded_policy_accepts_exact_compose_and_refuse_actions() -> None:
    policy = BoundedDynamicToolSelectionPolicy()

    compose = policy.evaluate(DynamicToolSelectionEvidence())
    refuse = policy.evaluate(
        DynamicToolSelectionEvidence(
            selected_action=DynamicDecisionAction.REFUSE_INSUFFICIENT_EVIDENCE.value
        )
    )

    assert compose.state is DynamicToolSelectionState.ELIGIBLE
    assert refuse.state is DynamicToolSelectionState.ELIGIBLE
    assert compose.rejection_reasons == refuse.rejection_reasons == ()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"model_decision_count": 2}, "model_decision_count_not_one"),
        ({"selected_action": "retrieve_candidate_pool"}, "selected_action_not_allowed"),
        ({"unauthorized_tool_ids": ("unknown",)}, "unauthorized_tool_requested"),
        ({"model_can_call_retrieval": True}, "model_retrieval_authority_not_allowed"),
        ({"second_retrieval_requested": True}, "second_retrieval_not_allowed"),
        ({"decision_loop_available": True}, "decision_loop_not_allowed"),
        ({"model_owns_final_answer_authority": True}, "model_final_answer_authority_not_allowed"),
        ({"retry_action_count": 1}, "retry_actions_not_allowed"),
        ({"fallback_action_count": 1}, "fallback_actions_not_allowed"),
        ({"test_gate_opened": True}, "test_gate_not_allowed"),
        ({"model_loaded_or_called": True}, "model_execution_out_of_scope"),
    ],
)
def test_bounded_policy_rejects_unauthorized_behavior(
    overrides: dict[str, object],
    reason: str,
) -> None:
    values = DynamicToolSelectionEvidence().__dict__ | overrides
    result = BoundedDynamicToolSelectionPolicy().evaluate(DynamicToolSelectionEvidence(**values))

    assert result.state is DynamicToolSelectionState.REJECTED
    assert reason in result.rejection_reasons


def test_decision_contract_has_one_non_looping_structured_choice() -> None:
    contract = bounded_decision_contract()

    assert contract["model_selectable_actions"] == [
        "compose_grounded_answer",
        "refuse_insufficient_evidence",
    ]
    assert contract["model_decision_count_per_turn"] == 1
    assert contract["retrieval_call_count_per_turn"] == 1
    assert contract["terminal_authority_by_action"] == {
        "compose_grounded_answer": "system_verifier",
        "refuse_insufficient_evidence": "fixed_system_refusal_constructor",
    }
    assert contract["diagnostic_observation_count_by_action"] == {
        "compose_grounded_answer": 1,
        "refuse_insufficient_evidence": 0,
    }
    assert contract["transition_loops_available"] is False
    assert contract["query_rewrite_enabled"] is False
    assert contract["second_retrieval_enabled"] is False


def test_thread_state_contract_is_volatile_strict_and_parameterized() -> None:
    contract = volatile_thread_state_contract()

    assert contract["storage_scope"] == "process_local_volatile_memory_only"
    assert contract["checkpointer_selected"] is False
    assert contract["persistent_store_selected"] is False
    assert contract["cross_thread_read_allowed"] is False
    assert contract["runtime_limit_values_frozen_in_stage156"] is False
    assert contract["runtime_limit_values_require_explicit_configuration"] is True
    assert contract["overflow_behavior"] == "reject_before_mutation"
    assert contract["silent_truncation_allowed"] is False
    assert contract["silent_eviction_allowed"] is False


def test_volatile_ledger_isolates_threads_and_clears_on_close() -> None:
    ledger = VolatileThreadStateLedger(
        limits=ThreadStateLimits(max_completed_turns=2, max_retained_bytes=1024)
    )
    ledger.open_thread("thread-a")
    ledger.open_thread("thread-b")
    ledger.append_completed_turn(
        "thread-a", CompletedThreadTurn(1, "private-a", "verified-a", "complete")
    )

    assert ledger.private_history("thread-a")[0].user_turn_input == "private-a"
    assert ledger.private_history("thread-b") == ()
    assert ledger.public_summary("thread-a").to_public_dict() == {
        "completed_turn_count": 1,
        "retained_state_bytes": len(b"private-a") + len(b"verified-a"),
        "opened": True,
    }

    closed = ledger.close_thread("thread-a")
    assert closed.opened is False
    with pytest.raises(ThreadStatePolicyViolationError):
        ledger.private_history("thread-a")


@pytest.mark.parametrize(
    "turn",
    [
        CompletedThreadTurn(2, "out-of-order", "response", "complete"),
        CompletedThreadTurn(1, "", "response", "complete"),
        CompletedThreadTurn(1, "input", "response", "answered"),
    ],
)
def test_volatile_ledger_rejects_invalid_turn_without_mutation(
    turn: CompletedThreadTurn,
) -> None:
    ledger = VolatileThreadStateLedger(
        limits=ThreadStateLimits(max_completed_turns=2, max_retained_bytes=1024)
    )
    ledger.open_thread("thread-a")

    with pytest.raises(ThreadStatePolicyViolationError):
        ledger.append_completed_turn("thread-a", turn)

    assert ledger.private_history("thread-a") == ()


def test_volatile_ledger_rejects_count_and_byte_overflow_without_eviction() -> None:
    ledger = VolatileThreadStateLedger(
        limits=ThreadStateLimits(max_completed_turns=1, max_retained_bytes=16)
    )
    ledger.open_thread("thread-a")
    ledger.append_completed_turn("thread-a", CompletedThreadTurn(1, "1234", "5678", "refuse"))
    before = ledger.private_history("thread-a")

    with pytest.raises(ThreadStatePolicyViolationError, match="turn limit"):
        ledger.append_completed_turn("thread-a", CompletedThreadTurn(2, "1", "2", "complete"))
    assert ledger.private_history("thread-a") == before

    byte_ledger = VolatileThreadStateLedger(
        limits=ThreadStateLimits(max_completed_turns=2, max_retained_bytes=4)
    )
    byte_ledger.open_thread("thread-b")
    with pytest.raises(ThreadStatePolicyViolationError, match="byte limit"):
        byte_ledger.append_completed_turn(
            "thread-b", CompletedThreadTurn(1, "1234", "5", "complete")
        )
    assert byte_ledger.private_history("thread-b") == ()


def test_volatile_ledger_never_implicitly_creates_thread() -> None:
    ledger = VolatileThreadStateLedger(
        limits=ThreadStateLimits(max_completed_turns=2, max_retained_bytes=1024)
    )

    with pytest.raises(ThreadStatePolicyViolationError, match="not open"):
        ledger.append_completed_turn(
            "missing", CompletedThreadTurn(1, "input", "response", "complete")
        )


def test_stage156_preflight_fails_only_confirmation_guard() -> None:
    report = _freeze(confirmed=False, note="synthetic preflight")

    failed = [check["name"] for check in report["guard_checks"] if not check["passed"]]
    assert failed == ["stage156_user_confirmed"]
    assert report["decision"]["protocol_frozen"] is False
    assert report["public_safe_contract"]["test_split_loaded"] is False
    assert report["public_safe_contract"]["models_loaded_or_called"] is False


def test_stage156_formal_freeze_passes_all_guards_without_private_content() -> None:
    sentinel = "private-stage156-sentinel"
    ledger = VolatileThreadStateLedger(
        limits=ThreadStateLimits(max_completed_turns=1, max_retained_bytes=1024)
    )
    ledger.open_thread("private-thread")
    ledger.append_completed_turn(
        "private-thread", CompletedThreadTurn(1, sentinel, sentinel, "complete")
    )

    report = _freeze(confirmed=True, note="user approved next big step")
    serialized = json.dumps(report, ensure_ascii=False)

    assert len(report["guard_checks"]) == 43
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["status"] == (
        "primeqa_hybrid_bounded_agent_tool_selection_state_protocol_frozen"
    )
    assert report["decision"]["model_loaded_or_called"] is False
    assert report["decision"]["runtime_changed"] is False
    assert report["decision"]["test_gate_opened"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert sentinel not in serialized


def test_tampered_stage155_source_is_rejected(tmp_path: Path) -> None:
    tampered = json.loads(_STAGE155.read_text(encoding="utf-8"))
    tampered["decision"]["status"] = "tampered"
    path = tmp_path / "stage155.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    report = freeze_primeqa_hybrid_bounded_agent_state_protocol(
        stage155_validation_path=path,
        user_confirmed_protocol=True,
        confirmation_note="synthetic tamper rejection",
    )

    failed = [check["name"] for check in report["guard_checks"] if not check["passed"]]
    assert "stage155_identity_exact" in failed
    assert report["decision"]["protocol_frozen"] is False


def test_stage156_visualizations_are_ten_parseable_svgs(tmp_path: Path) -> None:
    report = _freeze(confirmed=True, note="synthetic visualization validation")

    visualizations = write_primeqa_hybrid_bounded_agent_state_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        path = Path(visualization.path)
        assert path.exists()
        assert path.stat().st_size > 100
        assert ET.parse(path).getroot().tag.endswith("svg")


def _freeze(*, confirmed: bool, note: str) -> dict[str, object]:
    return freeze_primeqa_hybrid_bounded_agent_state_protocol(
        stage155_validation_path=_STAGE155,
        user_confirmed_protocol=confirmed,
        confirmation_note=note,
    )
