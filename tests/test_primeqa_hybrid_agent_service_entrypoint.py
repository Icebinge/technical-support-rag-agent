from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint import (
    AgentServiceExitCode,
    AgentServiceSourceFingerprint,
    CanonicalAgentServiceSourcePaths,
    LoadedJsonSource,
    PrimeQAHybridLocalAgentServiceEntrypoint,
    PublicSafeAgentServiceTerminalEvent,
    builtin_label_free_agent_service_warmup_query,
    parse_exact_agent_service_cli,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery
from ts_rag_agent.local_agent_service import main

_ROOT = Path(__file__).resolve().parents[1]
_STAGE150_PATH = (
    _ROOT / "artifacts" / "primeqa_hybrid_agent_http_transport_validation_stage150.json"
)


class _EventSink:
    def __init__(self) -> None:
        self.events: list[PublicSafeAgentServiceTerminalEvent] = []

    def emit(self, event: PublicSafeAgentServiceTerminalEvent) -> None:
        self.events.append(event)


class _SourceRepository:
    def __init__(self, *, stage150: dict[str, Any], fail_key: str | None = None) -> None:
        self._reports = {
            "stage150_http_transport_validation": stage150,
            "stage145_concurrent_runtime_validation": {"stage": "Stage 145"},
        }
        self._fail_key = fail_key
        self.calls: list[str] = []

    def load_json(self, source_key: str, path: Path) -> LoadedJsonSource:
        _ = path
        self.calls.append(f"load:{source_key}")
        if source_key == self._fail_key:
            raise OSError("synthetic source read failure")
        return LoadedJsonSource(
            report=self._reports[source_key],
            fingerprint=_fingerprint(source_key),
        )

    def fingerprint(self, source_key: str, path: Path) -> AgentServiceSourceFingerprint:
        _ = path
        self.calls.append(f"fingerprint:{source_key}")
        if source_key == self._fail_key:
            raise OSError("synthetic fingerprint failure")
        return _fingerprint(source_key)


class _ResourceFactory:
    def __init__(self) -> None:
        self.build_count = 0


class _ResourceFactoryProvider:
    def __init__(self) -> None:
        self.create_count = 0
        self.factory = _ResourceFactory()

    def create(self, paths: CanonicalAgentServiceSourcePaths) -> _ResourceFactory:
        _ = paths
        self.create_count += 1
        return self.factory


class _Bootstrap:
    def __init__(
        self,
        *,
        eligible: bool = True,
        failure_after_build: bool | None = None,
    ) -> None:
        self._eligible = eligible
        self._failure_after_build = failure_after_build
        self.received_warmup: PrimeQARuntimeQuery | None = None

    def start(self, **kwargs: Any) -> Any:
        factory = kwargs["resource_factory"]
        self.received_warmup = kwargs["warmup_question"]
        if self._failure_after_build is not None:
            factory.build_count = 1 if self._failure_after_build else 0
            raise RuntimeError("synthetic bootstrap failure")
        if not self._eligible:
            return _bootstrap_result(eligible=False)
        factory.build_count = 1
        return _bootstrap_result(eligible=True)


class _Listener:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _ListenerFactory:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.create_count = 0
        self.listener = _Listener()

    def create(self, *, port: int, backlog: int) -> _Listener:
        assert port == 18080
        assert backlog == 2048
        self.create_count += 1
        if self.fail:
            raise OSError("synthetic bind failure")
        return self.listener


class _Server:
    def __init__(self, *, fail: bool = False, interrupt: bool = False) -> None:
        self.fail = fail
        self.interrupt = interrupt
        self.started = False
        self.run_count = 0

    def run(self, sockets: list[Any] | None = None) -> None:
        assert sockets is not None and len(sockets) == 1
        self.run_count += 1
        if self.interrupt:
            self.started = True
            raise KeyboardInterrupt
        if self.fail:
            raise RuntimeError("synthetic server failure")
        self.started = True


class _ServerFactory:
    def __init__(self, server: _Server) -> None:
        self.server = server
        self.create_count = 0

    def create(self, config: Any) -> _Server:
        assert config.backlog == 2048
        self.create_count += 1
        return self.server


def test_parse_exact_cli_accepts_only_required_port_shape() -> None:
    assert parse_exact_agent_service_cli(["--port", "1024"]) == 1024
    assert parse_exact_agent_service_cli(["--port", "65535"]) == 65535

    invalid = (
        [],
        ["--port"],
        ["--port", "0"],
        ["--port", "1023"],
        ["--port", "65536"],
        ["--port=18080"],
        ["--host", "127.0.0.1", "--port", "18080"],
        ["--port", "18080", "--reload"],
        ["--help"],
        ["--port", "+18080"],
        ["--port", "１２３４"],
    )
    for argv in invalid:
        with pytest.raises(ValueError):
            parse_exact_agent_service_cli(argv)


def test_builtin_warmup_is_exact_label_free_runtime_query() -> None:
    query = builtin_label_free_agent_service_warmup_query()

    assert type(query) is PrimeQARuntimeQuery
    assert set(query.model_dump()) == {"id", "title", "text"}
    assert not hasattr(query, "answer")
    assert not hasattr(query, "answerable")
    assert not hasattr(query, "answer_doc_id")
    assert not hasattr(query, "doc_ids")


def test_stage150_rejection_stops_before_other_sources_and_resources() -> None:
    stage150 = _valid_stage150()
    stage150["decision"]["status"] = "rejected"
    harness = _harness(stage150=stage150)

    result = harness.entrypoint.run(port=18080)

    assert result.exit_code is AgentServiceExitCode.STAGE150_AUTHORIZATION_REJECTED
    assert harness.sources.calls == ["load:stage150_http_transport_validation"]
    assert harness.resources.create_count == 0
    assert harness.listener.create_count == 0
    _assert_one_event(harness, code=3)


def test_disabled_activation_stops_after_stage150_authorization() -> None:
    harness = _harness(concurrent=False, transport=False)

    result = harness.entrypoint.run(port=18080)

    assert result.exit_code is AgentServiceExitCode.ACTIVATION_CONFIGURATION_REJECTED
    assert harness.sources.calls == ["load:stage150_http_transport_validation"]
    assert harness.resources.create_count == 0
    assert harness.listener.create_count == 0
    _assert_one_event(harness, code=4)


def test_stage145_rejection_does_not_build_or_bind() -> None:
    bootstrap = _Bootstrap(eligible=False)
    harness = _harness(bootstrap=bootstrap)

    result = harness.entrypoint.run(port=18080)

    assert result.exit_code is AgentServiceExitCode.STAGE145_OR_RUNTIME_ACTIVATION_REJECTED
    assert harness.resources.create_count == 1
    assert harness.resources.factory.build_count == 0
    assert harness.listener.create_count == 0
    _assert_one_event(harness, code=5)


@pytest.mark.parametrize(
    ("failure_after_build", "expected_code"),
    [(False, 5), (True, 6)],
)
def test_bootstrap_failure_is_classified_by_resource_build_boundary(
    failure_after_build: bool,
    expected_code: int,
) -> None:
    harness = _harness(bootstrap=_Bootstrap(failure_after_build=failure_after_build))

    result = harness.entrypoint.run(port=18080)

    assert int(result.exit_code) == expected_code
    assert harness.listener.create_count == 0
    _assert_one_event(harness, code=expected_code)


def test_retrieval_source_failure_is_resource_failure_without_bind() -> None:
    harness = _harness(fail_source_key="primeqa_technote_documents")

    result = harness.entrypoint.run(port=18080)

    assert result.exit_code is AgentServiceExitCode.RESOURCE_OR_WARMUP_FAILURE
    assert harness.resources.create_count == 0
    assert harness.listener.create_count == 0
    _assert_one_event(harness, code=6)


def test_bind_failure_has_one_attempt_and_no_alternate_port() -> None:
    harness = _harness(listener_fail=True)

    result = harness.entrypoint.run(port=18080)

    assert result.exit_code is AgentServiceExitCode.SOCKET_BIND_OR_LISTEN_FAILURE
    assert harness.listener.create_count == 1
    assert harness.server.server.run_count == 0
    _assert_one_event(harness, code=7)


def test_app_composition_failure_maps_to_exit_one_without_binding() -> None:
    def failing_app_factory(**kwargs: Any) -> Any:
        _ = kwargs
        raise RuntimeError("synthetic app composition failure")

    harness = _harness(app_factory=failing_app_factory)

    result = harness.entrypoint.run(port=18080)

    assert result.exit_code is AgentServiceExitCode.UNEXPECTED_COMPOSITION_FAILURE
    assert harness.listener.create_count == 0
    assert harness.server.server.run_count == 0
    _assert_one_event(harness, code=1)


def test_server_failure_closes_listener_once_and_preserves_no_recovery_counts() -> None:
    harness = _harness(server=_Server(fail=True))

    result = harness.entrypoint.run(port=18080)

    assert result.exit_code is AgentServiceExitCode.SERVER_OR_LIFESPAN_FAILURE
    assert harness.listener.listener.close_count == 1
    event = _assert_one_event(harness, code=8)
    assert event.queue_action_count == 0
    assert event.retry_action_count == 0
    assert event.fallback_action_count == 0


def test_clean_lifecycle_uses_all_sources_warmup_bind_server_and_close_once() -> None:
    bootstrap = _Bootstrap()
    harness = _harness(bootstrap=bootstrap)

    result = harness.entrypoint.run(port=18080)

    assert result.exit_code is AgentServiceExitCode.CLEAN_SERVER_RETURN
    assert harness.sources.calls == [
        "load:stage150_http_transport_validation",
        "load:stage145_concurrent_runtime_validation",
        "fingerprint:stage128_agent_retrieval_protocol",
        "fingerprint:stage125_recall_expansion_protocol",
        "fingerprint:stage80_dense_sparse_report",
        "fingerprint:primeqa_technote_documents",
    ]
    assert [row.source_key for row in result.source_fingerprints] == [
        "stage150_http_transport_validation",
        "stage145_concurrent_runtime_validation",
        "stage128_agent_retrieval_protocol",
        "stage125_recall_expansion_protocol",
        "stage80_dense_sparse_report",
        "primeqa_technote_documents",
    ]
    assert bootstrap.received_warmup == builtin_label_free_agent_service_warmup_query()
    assert harness.resources.create_count == 1
    assert harness.resources.factory.build_count == 1
    assert harness.listener.create_count == 1
    assert harness.server.server.run_count == 1
    assert harness.listener.listener.close_count == 1
    event = _assert_one_event(harness, code=0)
    assert event.source_validation_state == "stage145_authorized"
    assert event.runtime_activation_state == "eligible"
    assert event.resources_initialized is True
    assert event.warmup_completed is True
    assert event.listener_bound is True
    assert event.server_started is True
    assert event.transport_state == "closed"


def test_keyboard_interrupt_is_not_normalized_and_listener_is_closed_once() -> None:
    harness = _harness(server=_Server(interrupt=True))

    with pytest.raises(KeyboardInterrupt):
        harness.entrypoint.run(port=18080)

    assert harness.listener.listener.close_count == 1
    assert len(harness.sink.events) == 1
    event = harness.sink.events[0]
    assert event.exit_code is None
    assert event.outcome_code == "external_signal"
    assert event.shutdown_trigger == "external_signal"


def test_terminal_event_has_exact_public_allowlist() -> None:
    harness = _harness()
    result = harness.entrypoint.run(port=18080)

    payload = result.terminal_event.to_public_dict()

    assert len(payload) == 18
    assert set(payload) == {
        "entrypoint_id",
        "phase",
        "outcome_code",
        "exit_code",
        "binding_host",
        "binding_port",
        "source_validation_state",
        "runtime_activation_state",
        "resources_initialized",
        "warmup_completed",
        "listener_bound",
        "server_started",
        "shutdown_trigger",
        "transport_state",
        "runtime_registered_as_default",
        "queue_action_count",
        "retry_action_count",
        "fallback_action_count",
    }
    assert "exception_message" not in payload
    assert "source_path" not in payload


def test_cli_main_rejects_unknown_option_with_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--port", "18080", "--reload"])

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["outcome_code"] == "cli_contract_invalid"
    assert output["exit_code"] == 2
    assert len(output) == 18


def _valid_stage150() -> dict[str, Any]:
    return json.loads(_STAGE150_PATH.read_text(encoding="utf-8"))


def _fingerprint(source_key: str) -> AgentServiceSourceFingerprint:
    return AgentServiceSourceFingerprint(
        source_key=source_key,
        size_bytes=1,
        sha256="0" * 64,
    )


def _bootstrap_result(*, eligible: bool) -> Any:
    trace = SimpleNamespace(
        activation_state="eligible" if eligible else "rejected",
        source_validation_state="eligible" if eligible else "rejected",
        resource_factory_build_count=1 if eligible else 0,
        warmup_request_count=1 if eligible else 0,
        warmup_candidate_pool_depth=400 if eligible else 0,
        resources_initialized=eligible,
        runtime_activated=eligible,
        registered_as_runtime_default=False,
        test_access_allowed=False,
        queue_action_count=0,
        retry_action_count=0,
        fallback_action_count=0,
    )
    return SimpleNamespace(
        runtime=object() if eligible else None,
        resource_summary=object() if eligible else None,
        startup_trace=trace,
    )


def _app_factory(**kwargs: Any) -> Any:
    _ = kwargs
    transport = SimpleNamespace(state=SimpleNamespace(value="closed"))
    return SimpleNamespace(state=SimpleNamespace(agent_http_transport=transport))


def _config_factory(**kwargs: Any) -> Any:
    _ = kwargs
    return SimpleNamespace(backlog=2048)


class _Harness:
    def __init__(
        self,
        *,
        entrypoint: PrimeQAHybridLocalAgentServiceEntrypoint,
        sources: _SourceRepository,
        resources: _ResourceFactoryProvider,
        listener: _ListenerFactory,
        server: _ServerFactory,
        sink: _EventSink,
    ) -> None:
        self.entrypoint = entrypoint
        self.sources = sources
        self.resources = resources
        self.listener = listener
        self.server = server
        self.sink = sink


def _harness(
    *,
    stage150: dict[str, Any] | None = None,
    concurrent: bool = True,
    transport: bool = True,
    bootstrap: _Bootstrap | None = None,
    fail_source_key: str | None = None,
    listener_fail: bool = False,
    server: _Server | None = None,
    app_factory: Any = _app_factory,
) -> _Harness:
    settings = ProjectSettings(
        data_dir=Path("unused-data"),
        artifact_dir=Path("unused-artifacts"),
        enable_concurrent_sidecar_agent=concurrent,
        enable_local_agent_http_transport=transport,
    )
    paths = CanonicalAgentServiceSourcePaths.from_settings(settings)
    sources = _SourceRepository(
        stage150=stage150 or _valid_stage150(),
        fail_key=fail_source_key,
    )
    resources = _ResourceFactoryProvider()
    listener = _ListenerFactory(fail=listener_fail)
    server_factory = _ServerFactory(server or _Server())
    sink = _EventSink()
    selected_bootstrap = bootstrap or _Bootstrap()
    entrypoint = PrimeQAHybridLocalAgentServiceEntrypoint(
        settings=settings,
        paths=paths,
        source_repository=sources,
        resource_factory_provider=resources,
        bootstrap_factory=lambda: selected_bootstrap,
        app_factory=app_factory,
        config_factory=_config_factory,
        listener_factory=listener,
        server_factory=server_factory,
        event_sink=sink,
    )
    return _Harness(
        entrypoint=entrypoint,
        sources=sources,
        resources=resources,
        listener=listener,
        server=server_factory,
        sink=sink,
    )


def _assert_one_event(
    harness: _Harness,
    *,
    code: int,
) -> PublicSafeAgentServiceTerminalEvent:
    assert len(harness.sink.events) == 1
    event = harness.sink.events[0]
    assert event.exit_code == code
    assert event.binding_host == "127.0.0.1"
    assert event.binding_port == 18080
    assert event.runtime_registered_as_default is False
    return event
