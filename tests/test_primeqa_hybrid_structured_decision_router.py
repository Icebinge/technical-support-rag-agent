from __future__ import annotations

import json

import pytest

from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    CompletedThreadTurn,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    GeneratedRouterText,
    StrictStructuredDecisionRouter,
    StructuredDecisionSchemaError,
    StructuredRouterPromptBuilder,
    StructuredRouterPromptPolicy,
    structured_decision_router_contract,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


class StaticGenerationBackend:
    def __init__(self, text: str) -> None:
        self.text = text
        self.call_count = 0
        self.prompt = ""
        self.max_input_tokens = 0
        self.max_new_tokens = 0

    def generate(
        self,
        *,
        prompt: str,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> GeneratedRouterText:
        self.call_count += 1
        self.prompt = prompt
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        return GeneratedRouterText(
            text=self.text,
            input_token_count=321,
            output_token_count=9,
            generation_latency_ms=12.5,
        )


@pytest.mark.parametrize(
    "action",
    ["compose_grounded_answer", "refuse_insufficient_evidence"],
)
def test_router_accepts_each_exact_action_with_one_generation_call(action: str) -> None:
    backend = StaticGenerationBackend(json.dumps({"action": action}, separators=(",", ":")))
    router = StrictStructuredDecisionRouter(backend=backend)

    decision = router.decide(
        question=_question(),
        generation_context_results=_results(),
        completed_turns=(),
    )

    assert decision.action == action
    assert backend.call_count == 1
    assert backend.max_input_tokens == 12_288
    assert backend.max_new_tokens == 32
    assert router.last_metrics is not None
    assert router.last_metrics.schema_valid is True
    assert router.last_metrics.selected_action == action


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        '```json\n{"action":"compose_grounded_answer"}\n```',
        '{"action":"retrieve_candidate_pool"}',
        '{"action":"compose_grounded_answer","reason":"private"}',
        '{"action":"compose_grounded_answer"} trailing',
    ],
)
def test_router_rejects_malformed_or_unauthorized_output_without_retry(raw: str) -> None:
    backend = StaticGenerationBackend(raw)
    router = StrictStructuredDecisionRouter(backend=backend)

    with pytest.raises(StructuredDecisionSchemaError):
        router.decide(
            question=_question(),
            generation_context_results=_results(),
            completed_turns=(),
        )

    assert backend.call_count == 1
    assert router.last_metrics is not None
    assert router.last_metrics.schema_valid is False
    assert router.last_metrics.selected_action is None


def test_prompt_uses_confirmed_top10_600_character_policy_and_private_history() -> None:
    policy = StructuredRouterPromptPolicy()
    builder = StructuredRouterPromptBuilder(policy=policy)
    sentinel = "private-history-sentinel"

    prompt = builder.build(
        question=_question(),
        generation_context_results=_results(count=12, text="x" * 800),
        completed_turns=(CompletedThreadTurn(1, sentinel, "verified response", "complete"),),
    )

    payload_text = prompt.split("PRIVATE_RUNTIME_DATA_BEGIN\n", maxsplit=1)[1].split(
        "\nPRIVATE_RUNTIME_DATA_END", maxsplit=1
    )[0]
    payload = json.loads(payload_text)
    assert len(payload["retrieved_evidence"]) == 10
    assert all(len(row["excerpt"]) == 600 for row in payload["retrieved_evidence"])
    assert payload["retrieved_evidence"][-1]["rank"] == 10
    assert sentinel in prompt
    assert "document-11" not in prompt
    assert policy.max_input_tokens == 12_288
    assert policy.max_new_tokens == 32


def test_router_contract_freezes_gpu_schema_limits_and_closed_recovery() -> None:
    contract = structured_decision_router_contract()

    assert contract["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert contract["provider"] == "transformers_local_files_only"
    assert contract["device"] == "cuda:0"
    assert contract["dtype"] == "bfloat16"
    assert contract["allowed_actions"] == [
        "compose_grounded_answer",
        "refuse_insufficient_evidence",
    ]
    assert contract["prompt_policy"] == {
        "max_evidence_results": 10,
        "max_evidence_chars_per_result": 600,
        "max_input_tokens": 12_288,
        "max_new_tokens": 32,
    }
    assert contract["automatic_prompt_truncation"] is False
    assert contract["input_overflow_behavior"] == "reject_before_generation"
    assert contract["queue_actions_allowed"] is False
    assert contract["retry_actions_allowed"] is False
    assert contract["fallback_actions_allowed"] is False


def _question() -> PrimeQARuntimeQuery:
    return PrimeQARuntimeQuery(
        id="private-router-question",
        title="Adapter setup",
        text="How do I configure the adapter?",
    )


def _results(
    *,
    count: int = 10,
    text: str = "Configure the adapter and restart the service.",
) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"document-{rank}",
                title=f"Adapter document {rank}",
                text=text,
            ),
            score=1.0 / rank,
            rank=rank,
        )
        for rank in range(1, count + 1)
    )
