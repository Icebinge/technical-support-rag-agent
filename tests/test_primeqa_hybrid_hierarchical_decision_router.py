from __future__ import annotations

import json

import pytest

from ts_rag_agent.application.primeqa_hybrid_hierarchical_decision_router import (
    EvidenceDisposition,
    HierarchicalConstrainedDecisionRouter,
    HierarchicalDecisionLayer,
    HierarchicalLayerInvocationMetrics,
    HierarchicalRouterPromptBuilder,
    RequestDisposition,
    hierarchical_decision_router_contract,
)
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    ClarificationKind,
    IterativeDecisionPhase,
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
        assert max_input_tokens == 4_096
        assert max_new_tokens == 32
        self.prompts.append(prompt)
        return GeneratedRouterText(
            text=self.outputs.pop(0),
            input_token_count=100,
            output_token_count=10,
            generation_latency_ms=5.0,
        )


class RecordingObserver:
    def __init__(self) -> None:
        self.before: list[str] = []
        self.after: list[HierarchicalLayerInvocationMetrics] = []

    def before_generation(self, layer: HierarchicalDecisionLayer) -> None:
        self.before.append(layer.value)

    def after_generation(self, metrics: HierarchicalLayerInvocationMetrics) -> None:
        self.after.append(metrics)


def _request(disposition: str, clarification_kind: str | None = None) -> str:
    return json.dumps(
        {"disposition": disposition, "clarification_kind": clarification_kind},
        separators=(",", ":"),
    )


def _evidence(disposition: str) -> str:
    return json.dumps({"disposition": disposition}, separators=(",", ":"))


@pytest.mark.parametrize(
    "phase,request_json,evidence_json,expected_action",
    [
        (
            IterativeDecisionPhase.INITIAL,
            _request(RequestDisposition.COMPLETE.value),
            _evidence(EvidenceDisposition.SUFFICIENT.value),
            "compose_grounded_answer",
        ),
        (
            IterativeDecisionPhase.INITIAL,
            _request(RequestDisposition.COMPLETE.value),
            _evidence(EvidenceDisposition.INSUFFICIENT.value),
            "inspect_alternate_evidence",
        ),
        (
            IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
            _request(RequestDisposition.COMPLETE.value),
            _evidence(EvidenceDisposition.INSUFFICIENT.value),
            "refuse_insufficient_evidence",
        ),
        (
            IterativeDecisionPhase.INITIAL,
            _request(RequestDisposition.UNSUPPORTED.value),
            _evidence(EvidenceDisposition.SUFFICIENT.value),
            "refuse_insufficient_evidence",
        ),
        (
            IterativeDecisionPhase.INITIAL,
            _request(
                RequestDisposition.MISSING_FACT.value,
                ClarificationKind.VERSION_OR_BUILD.value,
            ),
            _evidence(EvidenceDisposition.SUFFICIENT.value),
            "request_clarification",
        ),
    ],
)
def test_hierarchy_maps_two_strict_layers_to_authorized_action(
    phase: IterativeDecisionPhase,
    request_json: str,
    evidence_json: str,
    expected_action: str,
) -> None:
    observer = RecordingObserver()
    router = HierarchicalConstrainedDecisionRouter(
        backend=StaticBackend([request_json, evidence_json]),
        observer=observer,
    )

    decision = _decide(router, phase)

    assert decision.action == expected_action
    assert observer.before == ["request", "evidence"]
    assert len(observer.after) == 2
    assert router.last_metrics is not None
    assert router.last_metrics.schema_valid is True
    assert router.last_metrics.input_token_count == 200


def test_invalid_request_schema_still_runs_fixed_evidence_layer_then_fails() -> None:
    backend = StaticBackend(
        [
            '{"disposition":"complete_technical_request","extra":true}',
            _evidence(EvidenceDisposition.INSUFFICIENT.value),
        ]
    )
    observer = RecordingObserver()
    router = HierarchicalConstrainedDecisionRouter(backend=backend, observer=observer)

    with pytest.raises(StructuredDecisionSchemaError):
        _decide(router, IterativeDecisionPhase.INITIAL)

    assert len(backend.prompts) == 2
    assert [metric.schema_valid for metric in observer.after] == [False, True]
    assert router.last_metrics is not None
    assert router.last_metrics.schema_valid is False
    assert router.last_trace is not None
    assert router.last_trace.selected_action is None


def test_request_prompt_excludes_evidence_and_evidence_prompt_uses_shared_bounds() -> None:
    builder = HierarchicalRouterPromptBuilder()
    question = PrimeQARuntimeQuery(id="q", text="How do I reset Acme?")
    evidence = _results("doc", "Reset Acme in Settings. " * 30)

    request_prompt = builder.build_request_prompt(question=question, completed_turns=())
    evidence_prompt = builder.build_evidence_prompt(
        phase=IterativeDecisionPhase.INITIAL,
        question=question,
        initial_evidence_results=evidence,
        alternate_evidence_results=(),
        completed_turns=(),
    )

    assert "initial_evidence" not in request_prompt
    assert "retrieval_score" not in request_prompt
    assert "initial_evidence" in evidence_prompt
    payload = json.loads(
        evidence_prompt.split("PRIVATE_RUNTIME_DATA_BEGIN\n", 1)[1].rsplit("\n", 1)[0]
    )
    assert len(payload["initial_evidence"]) == 10
    assert all(len(row["excerpt"]) <= 200 for row in payload["initial_evidence"])


def test_hierarchical_contract_has_no_retry_fallback_or_default_activation() -> None:
    contract = hierarchical_decision_router_contract()

    assert contract["model_calls_per_phase"] == 2
    assert contract["request_layer_receives_evidence"] is False
    assert contract["action_selected_by_deterministic_mapping"] is True
    assert contract["retry_actions_allowed"] is False
    assert contract["fallback_actions_allowed"] is False
    assert contract["runtime_registered_as_default"] is False


def _decide(router: HierarchicalConstrainedDecisionRouter, phase: IterativeDecisionPhase):
    return router.decide(
        phase=phase,
        question=PrimeQARuntimeQuery(id="q", text="How do I reset Acme?"),
        initial_evidence_results=_results("initial", "Reset Acme in Settings."),
        alternate_evidence_results=(
            _results("alternate", "Open Security and choose Reset.")
            if phase is IterativeDecisionPhase.FINAL_AFTER_INSPECTION
            else ()
        ),
        completed_turns=(),
    )


def _results(prefix: str, text: str) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(id=f"{prefix}-{rank}", title="Acme note", text=text),
            score=1.0 / rank,
            rank=rank,
        )
        for rank in range(1, 11)
    )
