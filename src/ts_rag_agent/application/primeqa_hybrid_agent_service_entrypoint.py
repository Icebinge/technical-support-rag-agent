from __future__ import annotations

import hashlib
import json
import socket
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol, TextIO

import uvicorn

from ts_rag_agent.application.primeqa_hybrid_agent_http_transport import (
    create_primeqa_hybrid_agent_http_app,
    create_primeqa_hybrid_agent_uvicorn_config,
)
from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability import (
    AgentWorkflowObservationSink,
)
from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint_protocol import (
    stage150_service_authorization_summary,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrap,
    PrimeQAHybridConcurrentRuntimeBootstrapResult,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
    _forbidden_keys_found,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery

_ENTRYPOINT_ID = "primeqa_hybrid_local_agent_service_entrypoint_v1"
_BINDING_HOST = "127.0.0.1"
_MIN_PORT = 1024
_MAX_PORT = 65535
_EXPECTED_STAGE150_GUARDS = 37
_EXPECTED_STAGE154_GUARDS = 54
_STAGE154_ANALYSIS_ID = "primeqa_hybrid_langgraph_agent_tool_workflow_validation_v1"
_STAGE154_STATUS = "primeqa_hybrid_langgraph_agent_tool_workflow_implemented_and_validated"
_PUBLIC_EVENT_FIELDS = frozenset(
    {
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
)


class AgentServiceExitCode(IntEnum):
    """Frozen Stage151 process exit status values."""

    CLEAN_SERVER_RETURN = 0
    UNEXPECTED_COMPOSITION_FAILURE = 1
    CLI_CONTRACT_INVALID = 2
    STAGE150_AUTHORIZATION_REJECTED = 3
    ACTIVATION_CONFIGURATION_REJECTED = 4
    STAGE145_OR_RUNTIME_ACTIVATION_REJECTED = 5
    RESOURCE_OR_WARMUP_FAILURE = 6
    SOCKET_BIND_OR_LISTEN_FAILURE = 7
    SERVER_OR_LIFESPAN_FAILURE = 8
    STAGE154_WORKFLOW_AUTHORIZATION_REJECTED = 9


@dataclass(frozen=True)
class AgentServiceSourceFingerprint:
    """Path-free immutable source fingerprint retained by the composition result."""

    source_key: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalAgentServiceSourcePaths:
    """Canonical read-only Stage151 sources derived from ProjectSettings."""

    stage150_http_transport_validation: Path
    stage154_agent_tool_workflow_validation: Path
    stage153_agent_tool_orchestration_protocol: Path
    pyproject: Path
    agent_tool_workflow_source: Path
    concurrent_runtime_source: Path
    stage145_concurrent_runtime_validation: Path
    stage128_agent_retrieval_protocol: Path
    stage125_recall_expansion_protocol: Path
    stage80_dense_sparse_report: Path
    primeqa_technote_documents: Path

    @classmethod
    def from_settings(cls, settings: ProjectSettings) -> CanonicalAgentServiceSourcePaths:
        artifact_dir = settings.artifact_dir.resolve()
        project_root = Path(__file__).resolve().parents[3]
        return cls(
            stage150_http_transport_validation=(
                artifact_dir / "primeqa_hybrid_agent_http_transport_validation_stage150.json"
            ),
            stage154_agent_tool_workflow_validation=(
                artifact_dir / "primeqa_hybrid_agent_tool_workflow_validation_stage154.json"
            ),
            stage153_agent_tool_orchestration_protocol=(
                artifact_dir / "primeqa_hybrid_agent_tool_orchestration_protocol_stage153.json"
            ),
            pyproject=project_root / "pyproject.toml",
            agent_tool_workflow_source=(
                project_root
                / "src"
                / "ts_rag_agent"
                / "application"
                / "primeqa_hybrid_agent_tool_workflow.py"
            ),
            concurrent_runtime_source=(
                project_root
                / "src"
                / "ts_rag_agent"
                / "application"
                / "primeqa_hybrid_concurrent_sidecar_agent_runtime.py"
            ),
            stage145_concurrent_runtime_validation=(
                artifact_dir / "primeqa_hybrid_concurrent_runtime_validation_stage145.json"
            ),
            stage128_agent_retrieval_protocol=(
                artifact_dir / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json"
            ),
            stage125_recall_expansion_protocol=(
                artifact_dir
                / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json"
            ),
            stage80_dense_sparse_report=(
                artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json"
            ),
            primeqa_technote_documents=(
                settings.primeqa_raw_dir.resolve()
                / "TechQA"
                / "training_and_dev"
                / "training_dev_technotes.sections.json"
            ),
        )


@dataclass(frozen=True)
class PublicSafeAgentServiceTerminalEvent:
    """The exact 18-field terminal event frozen by Stage151."""

    entrypoint_id: str
    phase: str
    outcome_code: str
    exit_code: int | None
    binding_host: str
    binding_port: int | None
    source_validation_state: str
    runtime_activation_state: str
    resources_initialized: bool
    warmup_completed: bool
    listener_bound: bool
    server_started: bool
    shutdown_trigger: str
    transport_state: str
    runtime_registered_as_default: bool = False
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if set(payload) != _PUBLIC_EVENT_FIELDS:
            raise ValueError("service terminal event fields do not match Stage151")
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"service terminal event contains forbidden keys: {forbidden}")
        return payload


class AgentServiceTerminalEventSink(Protocol):
    """Write one allowlisted terminal event after service composition ends."""

    def emit(self, event: PublicSafeAgentServiceTerminalEvent) -> None: ...


class JsonLineAgentServiceTerminalEventSink:
    """Emit the terminal event as one flushed JSON line."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def emit(self, event: PublicSafeAgentServiceTerminalEvent) -> None:
        self._stream.write(
            json.dumps(event.to_public_dict(), ensure_ascii=True, sort_keys=True) + "\n"
        )
        self._stream.flush()


@dataclass(frozen=True)
class AgentServiceEntrypointResult:
    """Process-safe result containing no runtime, document, or request content."""

    exit_code: AgentServiceExitCode
    terminal_event: PublicSafeAgentServiceTerminalEvent
    source_fingerprints: tuple[AgentServiceSourceFingerprint, ...]


@dataclass(frozen=True)
class LoadedJsonSource:
    report: Mapping[str, Any]
    fingerprint: AgentServiceSourceFingerprint


class AgentServiceSourceRepository(Protocol):
    """Read and fingerprint canonical service sources without modifying them."""

    def load_json(self, source_key: str, path: Path) -> LoadedJsonSource: ...

    def fingerprint(self, source_key: str, path: Path) -> AgentServiceSourceFingerprint: ...


class FileAgentServiceSourceRepository:
    """SHA-256 source repository used by the production entrypoint."""

    def load_json(self, source_key: str, path: Path) -> LoadedJsonSource:
        content = path.read_bytes()
        report = json.loads(content)
        if not isinstance(report, dict):
            raise ValueError(f"{source_key} must contain a JSON object")
        return LoadedJsonSource(
            report=report,
            fingerprint=AgentServiceSourceFingerprint(
                source_key=source_key,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            ),
        )

    def fingerprint(self, source_key: str, path: Path) -> AgentServiceSourceFingerprint:
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                size_bytes += len(block)
                digest.update(block)
        return AgentServiceSourceFingerprint(
            source_key=source_key,
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
        )


class AgentServiceResourceFactoryProvider(Protocol):
    """Construct, but do not build, the process-owned retrieval resource factory."""

    def create(
        self,
        paths: CanonicalAgentServiceSourcePaths,
    ) -> PrimeQAHybridProcessRuntimeResourceFactory: ...


class FrozenAgentServiceResourceFactoryProvider:
    """Construct the exact frozen Stage128/125/80 process resource graph."""

    def create(
        self,
        paths: CanonicalAgentServiceSourcePaths,
    ) -> PrimeQAHybridProcessRuntimeResourceFactory:
        return PrimeQAHybridProcessRuntimeResourceFactory(
            stage128_protocol_path=paths.stage128_agent_retrieval_protocol,
            stage125_protocol_path=paths.stage125_recall_expansion_protocol,
            stage80_report_path=paths.stage80_dense_sparse_report,
            documents_path=paths.primeqa_technote_documents,
        )


class AgentServiceListenerFactory(Protocol):
    """Bind and listen once on the frozen loopback interface."""

    def create(self, *, port: int, backlog: int) -> socket.socket: ...


class LoopbackAgentServiceListenerFactory:
    """Own the one allowed production bind/listen attempt."""

    def create(self, *, port: int, backlog: int) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((_BINDING_HOST, port))
            listener.listen(backlog)
            listener.set_inheritable(True)
            return listener
        except BaseException:
            listener.close()
            raise


class AgentServiceServer(Protocol):
    """Subset of uvicorn.Server required by the composition root."""

    started: bool

    def run(self, sockets: list[socket.socket] | None = None) -> None: ...


class AgentServiceServerFactory(Protocol):
    """Create the server that will run on the entrypoint's calling thread."""

    def create(self, config: uvicorn.Config) -> AgentServiceServer: ...


class UvicornAgentServiceServerFactory:
    """Create one Uvicorn server; Server.run remains on the caller's main thread."""

    def create(self, config: uvicorn.Config) -> AgentServiceServer:
        return uvicorn.Server(config=config)


@dataclass
class _ExecutionState:
    source_validation_state: str = "not_evaluated"
    runtime_activation_state: str = "not_evaluated"
    resources_initialized: bool = False
    warmup_completed: bool = False
    listener_bound: bool = False
    server_started: bool = False
    transport_state: str = "not_created"
    source_fingerprints: list[AgentServiceSourceFingerprint] = field(default_factory=list)


class PrimeQAHybridLocalAgentServiceEntrypoint:
    """Strict one-process composition root for the non-default local Agent service."""

    def __init__(
        self,
        *,
        settings: ProjectSettings,
        paths: CanonicalAgentServiceSourcePaths | None = None,
        source_repository: AgentServiceSourceRepository | None = None,
        resource_factory_provider: AgentServiceResourceFactoryProvider | None = None,
        bootstrap_factory: Callable[[], PrimeQAHybridConcurrentRuntimeBootstrap] = (
            PrimeQAHybridConcurrentRuntimeBootstrap
        ),
        app_factory: Callable[..., Any] = create_primeqa_hybrid_agent_http_app,
        config_factory: Callable[..., uvicorn.Config] = create_primeqa_hybrid_agent_uvicorn_config,
        listener_factory: AgentServiceListenerFactory | None = None,
        server_factory: AgentServiceServerFactory | None = None,
        event_sink: AgentServiceTerminalEventSink | None = None,
        workflow_observation_sink: AgentWorkflowObservationSink | None = None,
    ) -> None:
        self._settings = settings
        self._paths = paths or CanonicalAgentServiceSourcePaths.from_settings(settings)
        self._source_repository = source_repository or FileAgentServiceSourceRepository()
        self._resource_factory_provider = (
            resource_factory_provider or FrozenAgentServiceResourceFactoryProvider()
        )
        self._bootstrap_factory = bootstrap_factory
        self._app_factory = app_factory
        self._config_factory = config_factory
        self._listener_factory = listener_factory or LoopbackAgentServiceListenerFactory()
        self._server_factory = server_factory or UvicornAgentServiceServerFactory()
        self._event_sink = event_sink or JsonLineAgentServiceTerminalEventSink()
        self._workflow_observation_sink = workflow_observation_sink

    def run(self, *, port: int) -> AgentServiceEntrypointResult:
        state = _ExecutionState()
        result = self._execute(port=port, state=state)
        self._event_sink.emit(result.terminal_event)
        return result

    def _execute(
        self,
        *,
        port: int,
        state: _ExecutionState,
    ) -> AgentServiceEntrypointResult:
        if not _MIN_PORT <= port <= _MAX_PORT:
            return self._result(
                code=AgentServiceExitCode.CLI_CONTRACT_INVALID,
                outcome="cli_contract_invalid",
                port=port,
                state=state,
                shutdown_trigger="startup_rejected",
            )

        try:
            stage150 = self._source_repository.load_json(
                "stage150_http_transport_validation",
                self._paths.stage150_http_transport_validation,
            )
            state.source_fingerprints.append(stage150.fingerprint)
        except Exception:
            state.source_validation_state = "stage150_rejected"
            return self._result(
                code=AgentServiceExitCode.STAGE150_AUTHORIZATION_REJECTED,
                outcome="stage150_authorization_rejected",
                port=port,
                state=state,
                shutdown_trigger="startup_rejected",
            )

        if not _stage150_authorized(stage150.report):
            state.source_validation_state = "stage150_rejected"
            return self._result(
                code=AgentServiceExitCode.STAGE150_AUTHORIZATION_REJECTED,
                outcome="stage150_authorization_rejected",
                port=port,
                state=state,
                shutdown_trigger="startup_rejected",
            )
        state.source_validation_state = "stage150_authorized"

        if not (
            self._settings.enable_concurrent_sidecar_agent
            and self._settings.enable_local_agent_http_transport
        ):
            state.runtime_activation_state = "configuration_rejected"
            return self._result(
                code=AgentServiceExitCode.ACTIVATION_CONFIGURATION_REJECTED,
                outcome="activation_configuration_rejected",
                port=port,
                state=state,
                shutdown_trigger="startup_rejected",
            )
        state.runtime_activation_state = "configuration_authorized"

        try:
            stage154 = self._source_repository.load_json(
                "stage154_agent_tool_workflow_validation",
                self._paths.stage154_agent_tool_workflow_validation,
            )
            state.source_fingerprints.append(stage154.fingerprint)
            stage154_sources = self._fingerprint_stage154_sources()
            state.source_fingerprints.extend(stage154_sources)
        except Exception:
            state.source_validation_state = "stage154_rejected"
            state.runtime_activation_state = "rejected"
            return self._result(
                code=AgentServiceExitCode.STAGE154_WORKFLOW_AUTHORIZATION_REJECTED,
                outcome="stage154_workflow_authorization_rejected",
                port=port,
                state=state,
                shutdown_trigger="startup_rejected",
            )

        if not _stage154_authorized(stage154.report, stage154_sources):
            state.source_validation_state = "stage154_rejected"
            state.runtime_activation_state = "rejected"
            return self._result(
                code=AgentServiceExitCode.STAGE154_WORKFLOW_AUTHORIZATION_REJECTED,
                outcome="stage154_workflow_authorization_rejected",
                port=port,
                state=state,
                shutdown_trigger="startup_rejected",
            )
        state.source_validation_state = "stage154_authorized"

        try:
            stage145 = self._source_repository.load_json(
                "stage145_concurrent_runtime_validation",
                self._paths.stage145_concurrent_runtime_validation,
            )
            state.source_fingerprints.append(stage145.fingerprint)
        except Exception:
            state.source_validation_state = "stage145_rejected"
            state.runtime_activation_state = "rejected"
            return self._result(
                code=AgentServiceExitCode.STAGE145_OR_RUNTIME_ACTIVATION_REJECTED,
                outcome="stage145_or_runtime_activation_rejected",
                port=port,
                state=state,
                shutdown_trigger="startup_rejected",
            )

        try:
            state.source_fingerprints.extend(self._fingerprint_retrieval_sources())
            resource_factory = self._resource_factory_provider.create(self._paths)
        except Exception:
            return self._result(
                code=AgentServiceExitCode.RESOURCE_OR_WARMUP_FAILURE,
                outcome="resource_or_warmup_failure",
                port=port,
                state=state,
                shutdown_trigger="startup_failure",
            )

        try:
            bootstrap_result = self._bootstrap_factory().start(
                settings=self._settings,
                stage145_report=stage145.report,
                resource_factory=resource_factory,
                warmup_question=builtin_label_free_agent_service_warmup_query(),
                observation_sink=self._workflow_observation_sink,
            )
        except Exception:
            build_count = int(getattr(resource_factory, "build_count", 0))
            code = (
                AgentServiceExitCode.RESOURCE_OR_WARMUP_FAILURE
                if build_count > 0
                else AgentServiceExitCode.STAGE145_OR_RUNTIME_ACTIVATION_REJECTED
            )
            state.source_validation_state = (
                "stage145_authorized" if build_count > 0 else "stage145_rejected"
            )
            state.runtime_activation_state = "rejected"
            return self._result(
                code=code,
                outcome=(
                    "resource_or_warmup_failure"
                    if code is AgentServiceExitCode.RESOURCE_OR_WARMUP_FAILURE
                    else "stage145_or_runtime_activation_rejected"
                ),
                port=port,
                state=state,
                shutdown_trigger="startup_failure",
            )

        if not _bootstrap_eligible(bootstrap_result):
            state.source_validation_state = "stage145_rejected"
            state.runtime_activation_state = "rejected"
            return self._result(
                code=AgentServiceExitCode.STAGE145_OR_RUNTIME_ACTIVATION_REJECTED,
                outcome="stage145_or_runtime_activation_rejected",
                port=port,
                state=state,
                shutdown_trigger="startup_rejected",
            )

        state.source_validation_state = "stage145_authorized"
        state.runtime_activation_state = "eligible"
        state.resources_initialized = True
        state.warmup_completed = True

        try:
            app = self._app_factory(
                settings=self._settings,
                bootstrap_result=bootstrap_result,
            )
            config = self._config_factory(app=app, port=port)
            server = self._server_factory.create(config)
            state.transport_state = "created"
        except Exception:
            return self._result(
                code=AgentServiceExitCode.UNEXPECTED_COMPOSITION_FAILURE,
                outcome="unexpected_composition_failure",
                port=port,
                state=state,
                shutdown_trigger="startup_failure",
            )

        try:
            listener = self._listener_factory.create(port=port, backlog=config.backlog)
            state.listener_bound = True
        except Exception:
            return self._result(
                code=AgentServiceExitCode.SOCKET_BIND_OR_LISTEN_FAILURE,
                outcome="socket_bind_or_listen_failure",
                port=port,
                state=state,
                shutdown_trigger="startup_failure",
            )

        try:
            try:
                server.run(sockets=[listener])
            except KeyboardInterrupt:
                state.server_started = bool(server.started)
                state.transport_state = _transport_state(app)
                event = self._event(
                    code=None,
                    outcome="external_signal",
                    port=port,
                    state=state,
                    shutdown_trigger="external_signal",
                )
                self._event_sink.emit(event)
                raise
            except (Exception, SystemExit):
                state.server_started = bool(server.started)
                state.transport_state = _transport_state(app)
                return self._result(
                    code=AgentServiceExitCode.SERVER_OR_LIFESPAN_FAILURE,
                    outcome="server_or_lifespan_failure",
                    port=port,
                    state=state,
                    shutdown_trigger="server_failure",
                )

            state.server_started = bool(server.started)
            state.transport_state = _transport_state(app)
            if not state.server_started:
                return self._result(
                    code=AgentServiceExitCode.SERVER_OR_LIFESPAN_FAILURE,
                    outcome="server_or_lifespan_failure",
                    port=port,
                    state=state,
                    shutdown_trigger="server_failure",
                )
            return self._result(
                code=AgentServiceExitCode.CLEAN_SERVER_RETURN,
                outcome="clean_server_return",
                port=port,
                state=state,
                shutdown_trigger="server_return",
            )
        finally:
            listener.close()

    def _fingerprint_retrieval_sources(self) -> tuple[AgentServiceSourceFingerprint, ...]:
        sources = (
            (
                "stage128_agent_retrieval_protocol",
                self._paths.stage128_agent_retrieval_protocol,
            ),
            (
                "stage125_recall_expansion_protocol",
                self._paths.stage125_recall_expansion_protocol,
            ),
            ("stage80_dense_sparse_report", self._paths.stage80_dense_sparse_report),
            ("primeqa_technote_documents", self._paths.primeqa_technote_documents),
        )
        return tuple(
            self._source_repository.fingerprint(source_key, path) for source_key, path in sources
        )

    def _fingerprint_stage154_sources(self) -> tuple[AgentServiceSourceFingerprint, ...]:
        sources = (
            (
                "stage153_protocol",
                self._paths.stage153_agent_tool_orchestration_protocol,
            ),
            ("pyproject", self._paths.pyproject),
            ("workflow_source", self._paths.agent_tool_workflow_source),
            ("concurrent_runtime_source", self._paths.concurrent_runtime_source),
        )
        return tuple(
            self._source_repository.fingerprint(source_key, path) for source_key, path in sources
        )

    def _result(
        self,
        *,
        code: AgentServiceExitCode,
        outcome: str,
        port: int,
        state: _ExecutionState,
        shutdown_trigger: str,
    ) -> AgentServiceEntrypointResult:
        return AgentServiceEntrypointResult(
            exit_code=code,
            terminal_event=self._event(
                code=code,
                outcome=outcome,
                port=port,
                state=state,
                shutdown_trigger=shutdown_trigger,
            ),
            source_fingerprints=tuple(state.source_fingerprints),
        )

    @staticmethod
    def _event(
        *,
        code: AgentServiceExitCode | None,
        outcome: str,
        port: int,
        state: _ExecutionState,
        shutdown_trigger: str,
    ) -> PublicSafeAgentServiceTerminalEvent:
        return PublicSafeAgentServiceTerminalEvent(
            entrypoint_id=_ENTRYPOINT_ID,
            phase="terminal",
            outcome_code=outcome,
            exit_code=int(code) if code is not None else None,
            binding_host=_BINDING_HOST,
            binding_port=port,
            source_validation_state=state.source_validation_state,
            runtime_activation_state=state.runtime_activation_state,
            resources_initialized=state.resources_initialized,
            warmup_completed=state.warmup_completed,
            listener_bound=state.listener_bound,
            server_started=state.server_started,
            shutdown_trigger=shutdown_trigger,
            transport_state=state.transport_state,
        )


def builtin_label_free_agent_service_warmup_query() -> PrimeQARuntimeQuery:
    """Return the single built-in serving-shape warmup query with no gold fields."""

    return PrimeQARuntimeQuery(
        id="local-agent-service-warmup",
        title="Service installation verification",
        text="How can I verify a service configuration after installation?",
    )


def cli_contract_failure_event(*, port: int | None = None) -> PublicSafeAgentServiceTerminalEvent:
    """Create the terminal event for an exact CLI parse rejection."""

    return PublicSafeAgentServiceTerminalEvent(
        entrypoint_id=_ENTRYPOINT_ID,
        phase="terminal",
        outcome_code="cli_contract_invalid",
        exit_code=int(AgentServiceExitCode.CLI_CONTRACT_INVALID),
        binding_host=_BINDING_HOST,
        binding_port=port,
        source_validation_state="not_evaluated",
        runtime_activation_state="not_evaluated",
        resources_initialized=False,
        warmup_completed=False,
        listener_bound=False,
        server_started=False,
        shutdown_trigger="startup_rejected",
        transport_state="not_created",
    )


def activation_configuration_failure_event(
    *,
    port: int,
) -> PublicSafeAgentServiceTerminalEvent:
    """Create a public event when ProjectSettings itself rejects activation input."""

    return PublicSafeAgentServiceTerminalEvent(
        entrypoint_id=_ENTRYPOINT_ID,
        phase="terminal",
        outcome_code="activation_configuration_rejected",
        exit_code=int(AgentServiceExitCode.ACTIVATION_CONFIGURATION_REJECTED),
        binding_host=_BINDING_HOST,
        binding_port=port,
        source_validation_state="not_evaluated",
        runtime_activation_state="configuration_rejected",
        resources_initialized=False,
        warmup_completed=False,
        listener_bound=False,
        server_started=False,
        shutdown_trigger="startup_rejected",
        transport_state="not_created",
    )


def parse_exact_agent_service_cli(argv: Sequence[str]) -> int:
    """Parse only the frozen `--port <PORT>` shape; no aliases or defaults exist."""

    if len(argv) != 2 or argv[0] != "--port":
        raise ValueError("expected exactly --port <PORT>")
    raw_port = argv[1]
    if not raw_port.isascii() or not raw_port.isdecimal():
        raise ValueError("port must contain decimal ASCII digits")
    port = int(raw_port)
    if not _MIN_PORT <= port <= _MAX_PORT:
        raise ValueError("port is outside the frozen range")
    return port


def _stage150_authorized(report: Mapping[str, Any]) -> bool:
    summary = stage150_service_authorization_summary(report)
    return bool(
        summary.get("source_identity_valid") is True
        and summary.get("source_guard_count") == _EXPECTED_STAGE150_GUARDS
        and summary.get("source_passed_guard_count") == _EXPECTED_STAGE150_GUARDS
        and summary.get("all_source_guards_passed") is True
        and summary.get("transport_implemented") is True
        and summary.get("in_process_asgi_validated") is True
        and summary.get("real_loopback_socket_validated") is True
        and summary.get("transport_disabled_by_default") is True
        and summary.get("local_loopback_only") is True
        and summary.get("transport_contract_exact") is True
        and summary.get("network_service_persistently_running") is False
        and summary.get("runtime_registered_as_default") is False
        and summary.get("closed_boundaries_preserved") is True
        and summary.get("test_split_loaded") is False
        and summary.get("test_metrics_run") is False
        and summary.get("private_keys_found") == []
    )


def _stage154_authorized(
    report: Mapping[str, Any],
    current_sources: Sequence[AgentServiceSourceFingerprint],
) -> bool:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    expected_sources = report.get("source_files") or {}
    current_by_key = {row.source_key: row for row in current_sources}
    source_keys = (
        "stage153_protocol",
        "pyproject",
        "workflow_source",
        "concurrent_runtime_source",
    )
    sources_exact = all(
        key in current_by_key
        and (expected_sources.get(key) or {}).get("size_bytes") == current_by_key[key].size_bytes
        and (expected_sources.get(key) or {}).get("sha256") == current_by_key[key].sha256
        for key in source_keys
    )
    return bool(
        report.get("stage") == "Stage 154"
        and report.get("analysis_id") == _STAGE154_ANALYSIS_ID
        and report.get("source_unchanged_after_validation") is True
        and len(checks) == _EXPECTED_STAGE154_GUARDS
        and all(row.get("passed") is True for row in checks)
        and decision.get("status") == _STAGE154_STATUS
        and decision.get("workflow_implemented") is True
        and decision.get("langgraph_adapter_validated") is True
        and decision.get("facade_http_request_path_validated") is True
        and decision.get("real_resource_service_lifecycle_validated") is True
        and decision.get("runtime_registered_as_default") is False
        and decision.get("remote_exposure_authorized") is False
        and decision.get("test_gate_opened") is False
        and decision.get("test_metrics_run") is False
        and decision.get("queue_actions_enabled") is False
        and decision.get("retry_actions_enabled") is False
        and decision.get("fallback_strategies_enabled") is False
        and set(expected_sources) == set(source_keys)
        and set(current_by_key) == set(source_keys)
        and sources_exact
    )


def _bootstrap_eligible(result: PrimeQAHybridConcurrentRuntimeBootstrapResult) -> bool:
    trace = result.startup_trace
    return bool(
        result.runtime is not None
        and result.resource_summary is not None
        and trace.activation_state == "eligible"
        and trace.source_validation_state == "eligible"
        and trace.resource_factory_build_count == 1
        and trace.warmup_request_count == 1
        and trace.warmup_candidate_pool_depth == 400
        and trace.resources_initialized
        and trace.runtime_activated
        and not trace.registered_as_runtime_default
        and not trace.test_access_allowed
        and trace.queue_action_count == 0
        and trace.retry_action_count == 0
        and trace.fallback_action_count == 0
    )


def _transport_state(app: Any) -> str:
    transport = getattr(getattr(app, "state", None), "agent_http_transport", None)
    state = getattr(transport, "state", None)
    return str(getattr(state, "value", state or "unknown"))
