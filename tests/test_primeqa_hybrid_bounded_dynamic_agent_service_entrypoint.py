from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    ThreadStatePolicyViolationError,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    BoundedDynamicAgentServiceExitCode,
    BoundedDynamicAgentServiceSourceFingerprint,
    BoundedDynamicAgentStartupError,
    CanonicalBoundedDynamicAgentServicePaths,
    PrimeQAHybridBoundedDynamicAgentServiceEntrypoint,
    bounded_dynamic_agent_service_entrypoint_contract,
    parse_exact_bounded_dynamic_agent_service_cli,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    GeneratedRouterText,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


class SyntheticSourceAuthorizer:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.call_count = 0

    def authorize(self, paths: CanonicalBoundedDynamicAgentServicePaths):
        _ = paths
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return (BoundedDynamicAgentServiceSourceFingerprint("stage157", 1, "a" * 64),)


class SyntheticRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    def retrieve(self, question: PrimeQARuntimeQuery) -> tuple[RetrievalResult, ...]:
        self.call_count += 1
        return tuple(
            RetrievalResult(
                document=PrimeQADocument(
                    id=f"{question.id}-{rank}",
                    title=f"Synthetic {rank}",
                    text="Generated service verification evidence.",
                ),
                score=10.0 / rank,
                rank=rank,
            )
            for rank in range(1, 401)
        )


class SyntheticResourceFactory:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.build_count = 0
        self.retriever = SyntheticRetriever()

    def build_shared(self):
        self.build_count += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(candidate_pool_retriever=self.retriever)


class SyntheticResourceProvider:
    def __init__(self, factory: SyntheticResourceFactory) -> None:
        self.factory = factory
        self.call_count = 0

    def create(self, paths: CanonicalBoundedDynamicAgentServicePaths):
        _ = paths
        self.call_count += 1
        return self.factory


class SyntheticBackend:
    def __init__(self) -> None:
        self.generation_call_count = 0
        self.snapshot_path = Path("synthetic-snapshot")

    def generate(self, **kwargs) -> GeneratedRouterText:
        _ = kwargs
        self.generation_call_count += 1
        return GeneratedRouterText(
            text='{"action":"refuse_insufficient_evidence"}',
            input_token_count=700,
            output_token_count=9,
            generation_latency_ms=2.0,
        )


class SyntheticBackendLoader:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.call_count = 0
        self.backend = SyntheticBackend()

    def load(self, snapshot_path: Path):
        _ = snapshot_path
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.backend


class SyntheticServer:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.started = False
        self.run_count = 0

    def run(self) -> None:
        self.run_count += 1
        if self.error is not None:
            raise self.error
        self.started = True


class SyntheticServerFactory:
    def __init__(self, server: SyntheticServer) -> None:
        self.server = server

    def create(self, config):
        _ = config
        return self.server


class RecordingEventSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


def test_exact_cli_shape_and_port_range() -> None:
    assert parse_exact_bounded_dynamic_agent_service_cli(["--port", "18158"]) == 18158
    invalid = (
        [],
        ["--port"],
        ["--port", "0"],
        ["--port", "1023"],
        ["--port", "65536"],
        ["--port=18158"],
        ["--host", "127.0.0.1", "--port", "18158"],
        ["--port", "+18158"],
        ["--port", "１８１５８"],
    )
    for arguments in invalid:
        with pytest.raises(ValueError):
            parse_exact_bounded_dynamic_agent_service_cli(arguments)


def test_prepare_orders_source_resource_model_warmup_and_closes_warmup_thread() -> None:
    order: list[str] = []
    authorizer = SyntheticSourceAuthorizer()
    original_authorize = authorizer.authorize

    def authorize(paths):
        order.append("source")
        return original_authorize(paths)

    authorizer.authorize = authorize
    factory = SyntheticResourceFactory()
    provider = SyntheticResourceProvider(factory)
    original_create = provider.create

    def create(paths):
        order.append("resource")
        return original_create(paths)

    provider.create = create
    loader = SyntheticBackendLoader()
    original_load = loader.load

    def load(path):
        order.append("model")
        return original_load(path)

    loader.load = load

    def app_factory(**kwargs):
        _ = kwargs
        order.append("app")
        return SimpleNamespace(state=SimpleNamespace())

    entrypoint = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=_active_settings(),
        paths=_paths(),
        source_authorizer=authorizer,
        resource_factory_provider=provider,
        backend_loader=loader,
        app_factory=app_factory,
    )

    prepared = entrypoint.prepare()

    assert order == ["source", "resource", "model", "app"]
    assert factory.build_count == 1
    assert factory.retriever.call_count == 1
    assert loader.backend.generation_call_count == 1
    assert prepared.warmup.retrieval_call_count == 1
    assert prepared.warmup.model_decision_count == 1
    assert prepared.warmup.terminal_state == "refuse"
    assert prepared.warmup.thread_opened_after_close is False
    with pytest.raises(ThreadStatePolicyViolationError, match="not open"):
        prepared.runtime.thread_summary("stage158-warmup-thread")


@pytest.mark.parametrize(
    ("failure_stage", "expected_code", "resource_calls", "model_calls"),
    [
        (
            "source",
            BoundedDynamicAgentServiceExitCode.STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED,
            0,
            0,
        ),
        ("resource", BoundedDynamicAgentServiceExitCode.RESOURCE_INITIALIZATION_FAILED, 1, 0),
        ("model", BoundedDynamicAgentServiceExitCode.MODEL_INITIALIZATION_FAILED, 1, 1),
    ],
)
def test_prepare_failure_stops_before_later_startup_stages(
    failure_stage: str,
    expected_code: BoundedDynamicAgentServiceExitCode,
    resource_calls: int,
    model_calls: int,
) -> None:
    source_error = (
        BoundedDynamicAgentStartupError(expected_code) if failure_stage == "source" else None
    )
    factory = SyntheticResourceFactory(
        error=RuntimeError("resource failure") if failure_stage == "resource" else None
    )
    provider = SyntheticResourceProvider(factory)
    loader = SyntheticBackendLoader(
        error=RuntimeError("model failure") if failure_stage == "model" else None
    )
    entrypoint = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=_active_settings(),
        paths=_paths(),
        source_authorizer=SyntheticSourceAuthorizer(error=source_error),
        resource_factory_provider=provider,
        backend_loader=loader,
        app_factory=lambda **kwargs: SimpleNamespace(),
    )

    with pytest.raises(BoundedDynamicAgentStartupError) as captured:
        entrypoint.prepare()

    assert captured.value.exit_code is expected_code
    assert provider.call_count == resource_calls
    assert loader.call_count == model_calls


def test_run_uses_separate_server_and_emits_one_public_terminal_event() -> None:
    server = SyntheticServer()
    sink = RecordingEventSink()
    entrypoint = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=_active_settings(),
        paths=_paths(),
        source_authorizer=SyntheticSourceAuthorizer(),
        resource_factory_provider=SyntheticResourceProvider(SyntheticResourceFactory()),
        backend_loader=SyntheticBackendLoader(),
        app_factory=lambda **kwargs: SimpleNamespace(),
        config_factory=lambda **kwargs: SimpleNamespace(),
        server_factory=SyntheticServerFactory(server),
        event_sink=sink,
    )

    result = entrypoint.run(port=18158)

    assert result.exit_code is BoundedDynamicAgentServiceExitCode.CLEAN_SERVER_RETURN
    assert server.run_count == 1
    assert len(sink.events) == 1
    assert sink.events[0].binding_host == "127.0.0.1"
    assert sink.events[0].server_started is True
    assert sink.events[0].queue_action_count == 0
    assert sink.events[0].retry_action_count == 0
    assert sink.events[0].fallback_action_count == 0


@pytest.mark.parametrize(
    ("failure_stage", "expected_progress"),
    [
        (
            "source",
            {
                "activation_authorized": True,
                "source_authorized": False,
                "resources_initialized": False,
                "model_initialized": False,
                "warmup_completed": False,
                "app_composed": False,
            },
        ),
        (
            "resource",
            {
                "activation_authorized": True,
                "source_authorized": True,
                "resources_initialized": False,
                "model_initialized": False,
                "warmup_completed": False,
                "app_composed": False,
            },
        ),
        (
            "model",
            {
                "activation_authorized": True,
                "source_authorized": True,
                "resources_initialized": True,
                "model_initialized": False,
                "warmup_completed": False,
                "app_composed": False,
            },
        ),
    ],
)
def test_terminal_event_reports_exact_completed_startup_stages(
    failure_stage: str,
    expected_progress: dict[str, bool],
) -> None:
    source_code = BoundedDynamicAgentServiceExitCode.STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED
    authorizer = SyntheticSourceAuthorizer(
        error=BoundedDynamicAgentStartupError(source_code) if failure_stage == "source" else None
    )
    factory = SyntheticResourceFactory(
        error=RuntimeError("resource failure") if failure_stage == "resource" else None
    )
    loader = SyntheticBackendLoader(
        error=RuntimeError("model failure") if failure_stage == "model" else None
    )
    sink = RecordingEventSink()
    entrypoint = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=_active_settings(),
        paths=_paths(),
        source_authorizer=authorizer,
        resource_factory_provider=SyntheticResourceProvider(factory),
        backend_loader=loader,
        app_factory=lambda **kwargs: SimpleNamespace(),
        event_sink=sink,
    )

    result = entrypoint.run(port=18158)

    assert (
        result.exit_code
        is {
            "source": source_code,
            "resource": BoundedDynamicAgentServiceExitCode.RESOURCE_INITIALIZATION_FAILED,
            "model": BoundedDynamicAgentServiceExitCode.MODEL_INITIALIZATION_FAILED,
        }[failure_stage]
    )
    event = sink.events[0]
    assert {
        key: getattr(event, key)
        for key in (
            "activation_authorized",
            "source_authorized",
            "resources_initialized",
            "model_initialized",
            "warmup_completed",
            "app_composed",
        )
    } == expected_progress


def test_contract_keeps_old_service_test_remote_and_recovery_closed() -> None:
    contract = bounded_dynamic_agent_service_entrypoint_contract()

    assert len(contract["required_activation_flags"]) == 2
    assert contract["source_gate_before_resource_build"] is True
    assert contract["resource_build_before_model_load"] is True
    assert contract["warmup_thread_closed_before_listener"] is True
    assert contract["existing_service_changed"] is False
    assert contract["runtime_registered_as_default"] is False
    assert contract["remote_exposure_authorized"] is False
    assert contract["persistent_state_enabled"] is False
    assert contract["test_access_allowed"] is False
    assert contract["queue_actions_allowed"] is False
    assert contract["retry_actions_allowed"] is False
    assert contract["fallback_strategies_allowed"] is False


def _active_settings() -> ProjectSettings:
    return ProjectSettings(
        _env_file=None,
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=Path("synthetic-snapshot"),
    )


def _paths() -> CanonicalBoundedDynamicAgentServicePaths:
    root = Path("synthetic")
    return CanonicalBoundedDynamicAgentServicePaths(
        stage157_validation=root / "stage157.json",
        router_source=root / "router.py",
        runtime_source=root / "runtime.py",
        stage128_protocol=root / "stage128.json",
        stage125_protocol=root / "stage125.json",
        stage80_report=root / "stage80.json",
        documents=root / "documents.json",
        model_snapshot=root / "model",
    )
