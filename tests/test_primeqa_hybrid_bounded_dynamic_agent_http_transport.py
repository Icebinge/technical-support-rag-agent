from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    PrimeQAHybridAgentToolset,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    ThreadStateLimits,
    VolatileThreadStateLedger,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_http_transport import (
    BoundedDynamicAgentServiceError,
    BoundedDynamicAgentServiceErrorCode,
    PublicSafeBoundedDynamicAgentHttpEvent,
    bounded_dynamic_agent_http_transport_contract,
    create_primeqa_hybrid_bounded_dynamic_agent_http_app,
    create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    PRODUCTION_MAX_COMPLETED_TURNS,
    PRODUCTION_MAX_RETAINED_BYTES,
    create_primeqa_hybrid_bounded_dynamic_agent_runtime_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    BoundedAnswerDecision,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.answer import (
    AnswerCitation,
    AnswerVerificationResult,
    GeneratedAnswer,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


@dataclass
class RecordingSink:
    events: list[PublicSafeBoundedDynamicAgentHttpEvent] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def emit(self, event: PublicSafeBoundedDynamicAgentHttpEvent) -> None:
        event.to_public_dict()
        with self.lock:
            self.events.append(event)


class SyntheticRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    def retrieve(self, question: PrimeQARuntimeQuery) -> tuple[RetrievalResult, ...]:
        self.call_count += 1
        return tuple(
            RetrievalResult(
                document=PrimeQADocument(
                    id=f"{question.id}-{rank}",
                    title=f"Procedure {rank}",
                    text="Apply the verified service procedure.",
                ),
                score=10.0 / rank,
                rank=rank,
            )
            for rank in range(1, 401)
        )


class SyntheticRouter:
    def __init__(self, *, entered: Event | None = None, release: Event | None = None) -> None:
        self.call_count = 0
        self.entered = entered
        self.release = release

    @property
    def last_metrics(self):
        return None

    def decide(self, **kwargs):
        _ = kwargs
        self.call_count += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait()
        return BoundedAnswerDecision(action="compose_grounded_answer")


class SyntheticGenerator:
    def generate(
        self,
        question: PrimeQARuntimeQuery,
        retrieval_results: tuple[RetrievalResult, ...],
    ) -> GeneratedAnswer:
        first = retrieval_results[0]
        return GeneratedAnswer(
            question_id=question.id,
            answer="Apply the verified service procedure.",
            citations=[
                AnswerCitation(
                    document_id=first.document.id,
                    title=first.document.title,
                    retrieval_rank=first.rank,
                    evidence_score=10.0,
                )
            ],
            refused=False,
        )


class SyntheticGeneratorFactory:
    def create(self) -> SyntheticGenerator:
        return SyntheticGenerator()


class SyntheticVerifier:
    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: tuple[RetrievalResult, ...],
    ) -> AnswerVerificationResult:
        _ = retrieval_results
        return AnswerVerificationResult(
            original_answer=answer,
            verified_answer=answer,
            citation_context_valid=True,
            reasons=["verified"],
        )


class SyntheticVerifierFactory:
    def create(self) -> SyntheticVerifier:
        return SyntheticVerifier()


def test_stage158_settings_default_closed_and_require_exact_activation(monkeypatch) -> None:
    monkeypatch.delenv("TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_RUNTIME", raising=False)
    monkeypatch.delenv("TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_HTTP_TRANSPORT", raising=False)
    settings = ProjectSettings(_env_file=None)

    assert settings.enable_bounded_dynamic_agent_runtime is False
    assert settings.enable_bounded_dynamic_agent_http_transport is False
    assert settings.bounded_dynamic_agent_model_snapshot is None

    with pytest.raises(ValidationError, match="requires its bounded dynamic runtime"):
        ProjectSettings(
            _env_file=None,
            enable_bounded_dynamic_agent_http_transport=True,
        )
    with pytest.raises(ValidationError, match="requires an explicit model snapshot"):
        ProjectSettings(
            _env_file=None,
            enable_bounded_dynamic_agent_runtime=True,
        )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ProjectSettings(
            _env_file=None,
            enable_concurrent_sidecar_agent=True,
            enable_bounded_dynamic_agent_runtime=True,
            bounded_dynamic_agent_model_snapshot=Path("snapshot"),
        )


@pytest.mark.parametrize("raw", ["1", "yes", "on", "enabled", ""])
def test_stage158_settings_reject_ambiguous_activation_values(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_RUNTIME", raw)
    with pytest.raises(ValidationError, match="must be explicit true or false"):
        ProjectSettings(_env_file=None)


def test_explicit_open_turn_close_returns_private_payload_and_public_safe_events() -> None:
    runtime, retriever, router = _runtime()
    sink = RecordingSink()
    app = create_primeqa_hybrid_bounded_dynamic_agent_http_app(
        settings=_active_settings(),
        runtime=runtime,
        log_sink=sink,
    )

    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").json() == {
            "status": "ready",
            "coordinator_state": "accepting",
        }
        opened = client.post(
            "/v1/bounded-agent/threads/open",
            json={"thread_handle": "thread.alpha"},
        )
        turn = client.post(
            "/v1/bounded-agent/threads/turn",
            json={
                "thread_handle": "thread.alpha",
                "title": "Service verification",
                "text": "How do I verify the service?",
            },
        )
        closed = client.post(
            "/v1/bounded-agent/threads/close",
            json={"thread_handle": "thread.alpha"},
        )

    assert opened.status_code == 201
    assert opened.json()["opened"] is True
    assert turn.status_code == 200
    assert turn.json()["thread_handle"] == "thread.alpha"
    assert turn.json()["terminal_state"] == "complete"
    assert turn.json()["completed_turn_count"] == 1
    assert turn.json()["citations"][0]["rank"] == 1
    assert closed.status_code == 200
    assert closed.json()["opened"] is False
    assert retriever.call_count == router.call_count == 1
    serialized_events = json.dumps([event.to_public_dict() for event in sink.events])
    assert "thread.alpha" not in serialized_events
    assert "How do I verify" not in serialized_events
    assert "Apply the verified" not in serialized_events
    assert all(event.queue_action_count == 0 for event in sink.events)
    assert all(event.retry_action_count == 0 for event in sink.events)
    assert all(event.fallback_action_count == 0 for event in sink.events)


def test_thread_conflicts_and_unknown_thread_have_exact_status_codes() -> None:
    runtime, retriever, router = _runtime()
    app = create_primeqa_hybrid_bounded_dynamic_agent_http_app(
        settings=_active_settings(),
        runtime=runtime,
    )
    with TestClient(app) as client:
        first = client.post(
            "/v1/bounded-agent/threads/open",
            json={"thread_handle": "thread-a"},
        )
        duplicate = client.post(
            "/v1/bounded-agent/threads/open",
            json={"thread_handle": "thread-a"},
        )
        missing_turn = client.post(
            "/v1/bounded-agent/threads/turn",
            json={"thread_handle": "missing", "text": "Question"},
        )
        missing_close = client.post(
            "/v1/bounded-agent/threads/close",
            json={"thread_handle": "missing"},
        )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "thread_already_open"
    assert missing_turn.status_code == missing_close.status_code == 404
    assert retriever.call_count == router.call_count == 0


def test_second_turn_rejects_before_retrieval_without_queue_or_retry() -> None:
    entered = Event()
    release = Event()
    runtime, retriever, router = _runtime(entered=entered, release=release)
    sink = RecordingSink()
    app = create_primeqa_hybrid_bounded_dynamic_agent_http_app(
        settings=_active_settings(),
        runtime=runtime,
        log_sink=sink,
    )
    with TestClient(app) as client:
        for handle in ("thread-a", "thread-b"):
            assert (
                client.post(
                    "/v1/bounded-agent/threads/open",
                    json={"thread_handle": handle},
                ).status_code
                == 201
            )

        coordinator = app.state.bounded_dynamic_agent_http_transport.coordinator
        admission = coordinator.admit_turn("thread-a")
        first_result: dict[str, object] = {}

        def run_first() -> None:
            first_result["run"] = coordinator.run_admitted_turn(
                admission=admission,
                question=PrimeQARuntimeQuery(
                    id="first-turn",
                    text="First question",
                ),
            )

        worker = Thread(target=run_first)
        worker.start()
        entered.wait()
        with pytest.raises(BoundedDynamicAgentServiceError) as capacity:
            coordinator.admit_turn("thread-b")
        with pytest.raises(BoundedDynamicAgentServiceError) as same_thread:
            coordinator.admit_turn("thread-a")
        with pytest.raises(BoundedDynamicAgentServiceError) as busy_close:
            coordinator.close_thread("thread-a")
        release.set()
        worker.join()
        assert (
            client.post(
                "/v1/bounded-agent/threads/close",
                json={"thread_handle": "thread-a"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/bounded-agent/threads/close",
                json={"thread_handle": "thread-b"},
            ).status_code
            == 200
        )

    assert capacity.value.code is BoundedDynamicAgentServiceErrorCode.GPU_CAPACITY_EXCEEDED
    assert same_thread.value.code is BoundedDynamicAgentServiceErrorCode.THREAD_BUSY
    assert busy_close.value.code is BoundedDynamicAgentServiceErrorCode.THREAD_BUSY
    assert first_result["run"].runtime_run.verified_answer.refused is False
    assert retriever.call_count == router.call_count == 1
    counters = app.state.bounded_dynamic_agent_http_transport.coordinator.counters()
    assert counters.max_observed_in_flight_turns == 1
    assert counters.capacity_rejected_turn_count == 1
    assert counters.queue_action_count == counters.retry_action_count == 0
    assert counters.fallback_action_count == 0


def test_http_maps_nonblocking_gpu_capacity_rejection_to_503(monkeypatch) -> None:
    runtime, retriever, router = _runtime()
    app = create_primeqa_hybrid_bounded_dynamic_agent_http_app(
        settings=_active_settings(),
        runtime=runtime,
    )
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/bounded-agent/threads/open",
                json={"thread_handle": "thread-a"},
            ).status_code
            == 201
        )
        coordinator = app.state.bounded_dynamic_agent_http_transport.coordinator

        def reject_capacity(thread_handle: str):
            _ = thread_handle
            raise BoundedDynamicAgentServiceError(
                BoundedDynamicAgentServiceErrorCode.GPU_CAPACITY_EXCEEDED
            )

        monkeypatch.setattr(coordinator, "admit_turn", reject_capacity)
        response = client.post(
            "/v1/bounded-agent/threads/turn",
            json={"thread_handle": "thread-a", "text": "Question"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "gpu_capacity_exceeded"
    assert retriever.call_count == router.call_count == 0


@pytest.mark.parametrize(
    ("body", "expected_status"),
    [
        (b"not-json", 400),
        (b'{"thread_handle":"a","thread_handle":"b"}', 400),
        (b'{"thread_handle":"bad handle"}', 422),
        (b'{"thread_handle":"valid","extra":1}', 422),
    ],
)
def test_transport_rejects_malformed_or_nonexact_payloads(
    body: bytes,
    expected_status: int,
) -> None:
    runtime, retriever, router = _runtime()
    app = create_primeqa_hybrid_bounded_dynamic_agent_http_app(
        settings=_active_settings(),
        runtime=runtime,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/bounded-agent/threads/open",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == expected_status
    assert retriever.call_count == router.call_count == 0


def test_contract_and_uvicorn_config_are_nondefault_loopback_only() -> None:
    contract = bounded_dynamic_agent_http_transport_contract()
    runtime, _, _ = _runtime()
    app = create_primeqa_hybrid_bounded_dynamic_agent_http_app(
        settings=_active_settings(),
        runtime=runtime,
    )
    config = create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config(
        app=app,
        port=18158,
    )

    assert contract["max_in_flight_turns"] == 1
    assert contract["application_waiting_queue"] is False
    assert contract["existing_answer_route_changed"] is False
    assert contract["runtime_registered_as_default"] is False
    assert contract["test_access_allowed"] is False
    assert config.host == "127.0.0.1"
    assert config.workers == 1
    assert config.timeout_graceful_shutdown is None


def _active_settings() -> ProjectSettings:
    return ProjectSettings(
        _env_file=None,
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=Path("synthetic-model-snapshot"),
    )


def _runtime(*, entered: Event | None = None, release: Event | None = None):
    retriever = SyntheticRetriever()
    router = SyntheticRouter(entered=entered, release=release)
    toolset = PrimeQAHybridAgentToolset(
        candidate_pool_retriever=retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator_factory=SyntheticGeneratorFactory(),
        answer_verifier_factory=SyntheticVerifierFactory(),
    )
    runtime = create_primeqa_hybrid_bounded_dynamic_agent_runtime_from_toolset(
        toolset=toolset,
        decision_router=router,
        thread_ledger=VolatileThreadStateLedger(
            limits=ThreadStateLimits(
                max_completed_turns=PRODUCTION_MAX_COMPLETED_TURNS,
                max_retained_bytes=PRODUCTION_MAX_RETAINED_BYTES,
            )
        ),
    )
    return runtime, retriever, router
