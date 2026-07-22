from __future__ import annotations

import json

import pytest

from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    ClarificationKind,
    IterativeDecisionPhase,
    IterativeRouterPromptBuilder,
    StrictIterativeStructuredDecisionRouter,
    iterative_decision_router_contract,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    GeneratedRouterText,
    StructuredDecisionSchemaError,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


class StaticBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, *, prompt: str, max_input_tokens: int, max_new_tokens: int):
        assert max_input_tokens == 12_288
        assert max_new_tokens == 48
        self.prompts.append(prompt)
        return GeneratedRouterText(
            text=self.outputs.pop(0),
            input_token_count=100,
            output_token_count=12,
            generation_latency_ms=5.0,
        )


@pytest.mark.parametrize(
    "action",
    [
        "compose_grounded_answer",
        "inspect_alternate_evidence",
        "refuse_insufficient_evidence",
    ],
)
def test_initial_router_accepts_exact_non_clarification_actions(action: str) -> None:
    router = StrictIterativeStructuredDecisionRouter(
        backend=StaticBackend([json.dumps({"action": action}, separators=(",", ":"))])
    )

    decision = _decide(router, phase=IterativeDecisionPhase.INITIAL)

    assert decision.action == action
    assert decision.clarification_kind is None
    assert router.last_metrics is not None and router.last_metrics.schema_valid is True


def test_router_accepts_system_mapped_clarification_kind() -> None:
    kind = ClarificationKind.ERROR_CODE_OR_LOG.value
    router = StrictIterativeStructuredDecisionRouter(
        backend=StaticBackend(
            [json.dumps({"action": "request_clarification", "clarification_kind": kind})]
        )
    )

    decision = _decide(router, phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION)

    assert decision.clarification_kind == kind


@pytest.mark.parametrize(
    "payload,phase",
    [
        ({"action": "inspect_alternate_evidence"}, IterativeDecisionPhase.FINAL_AFTER_INSPECTION),
        ({"action": "request_clarification"}, IterativeDecisionPhase.INITIAL),
        (
            {"action": "compose_grounded_answer", "clarification_kind": "version_or_build"},
            IterativeDecisionPhase.INITIAL,
        ),
        ({"action": "arbitrary_tool"}, IterativeDecisionPhase.INITIAL),
        (
            {"action": "compose_grounded_answer", "reason": "private"},
            IterativeDecisionPhase.INITIAL,
        ),
    ],
)
def test_router_rejects_phase_loop_missing_kind_and_extra_fields(payload: dict, phase) -> None:
    router = StrictIterativeStructuredDecisionRouter(
        backend=StaticBackend([json.dumps(payload, separators=(",", ":"))])
    )

    with pytest.raises(StructuredDecisionSchemaError):
        _decide(router, phase=phase)

    assert router.last_metrics is not None and router.last_metrics.schema_valid is False


def test_final_prompt_contains_both_bounded_views_and_no_inspect_action() -> None:
    builder = IterativeRouterPromptBuilder()

    prompt = builder.build(
        phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
        question=_question(),
        initial_evidence_results=_results("initial"),
        alternate_evidence_results=_results("alternate"),
        completed_turns=(),
    )
    payload = json.loads(
        prompt.split("PRIVATE_RUNTIME_DATA_BEGIN\n", 1)[1].split("\nPRIVATE_RUNTIME_DATA_END", 1)[0]
    )

    assert len(payload["initial_evidence"]) == 10
    assert len(payload["alternate_evidence"]) == 10
    instruction = prompt.split("PRIVATE_RUNTIME_DATA_BEGIN", 1)[0]
    assert '"inspect_alternate_evidence"' not in instruction.split("Authorized clarification")[0]


def test_router_contract_records_user_authorized_fallback_and_closed_loop() -> None:
    contract = iterative_decision_router_contract()

    assert contract["maximum_model_decisions_per_turn"] == 2
    assert "inspect_alternate_evidence" in contract["initial_actions"]
    assert "inspect_alternate_evidence" not in contract["final_actions"]
    assert contract["model_generated_clarification_text_allowed"] is False
    assert contract["user_authorized_clarification_fallback"] is True
    assert contract["retry_actions_allowed"] is False


def _decide(router, *, phase: IterativeDecisionPhase):
    return router.decide(
        phase=phase,
        question=_question(),
        initial_evidence_results=_results("initial"),
        alternate_evidence_results=(
            () if phase is IterativeDecisionPhase.INITIAL else _results("alternate")
        ),
        completed_turns=(),
    )


def _question() -> PrimeQARuntimeQuery:
    return PrimeQARuntimeQuery(id="question", title="Adapter", text="How do I configure it?")


def _results(prefix: str) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"{prefix}-{rank}",
                title=f"{prefix} document {rank}",
                text="Documented adapter configuration procedure.",
            ),
            score=1.0 / rank,
            rank=rank,
        )
        for rank in range(1, 11)
    )
