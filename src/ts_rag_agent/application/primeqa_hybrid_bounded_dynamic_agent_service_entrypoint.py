from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol

import uvicorn

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_http_transport import (
    BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
    create_primeqa_hybrid_bounded_dynamic_agent_http_app,
    create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    PrimeQAHybridBoundedDynamicAgentRuntime,
    create_primeqa_hybrid_bounded_dynamic_agent_runtime,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    Qwen3VLTransformersTextGenerationBackend,
    StrictStructuredDecisionRouter,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery

BOUNDED_DYNAMIC_AGENT_SERVICE_ENTRYPOINT_ID = (
    "primeqa_hybrid_bounded_dynamic_agent_service_entrypoint_v1"
)
EXPECTED_STAGE157_ARTIFACT_SHA256 = (
    "2351015d2c7447e6a5e1c2fe99b6583f0b9067e126ef2bfdd87b0b80c725c3e1"
)
EXPECTED_STAGE157_GUARD_COUNT = 47
EXPECTED_STAGE157_STATUS = "primeqa_hybrid_bounded_dynamic_agent_runtime_implemented_and_validated"
EXPECTED_ROUTER_SOURCE_SHA256 = "d9eeaff5fbb9c97a689efdee72d17f699cce47d1c94361047a74c90906442195"
EXPECTED_RUNTIME_SOURCE_SHA256 = "e3d38c5e81a86ac9454b2573fea455b94393f911238918df7b3247038273a071"
_MIN_PORT = 1024
_MAX_PORT = 65535


class BoundedDynamicAgentServiceExitCode(IntEnum):
    CLEAN_SERVER_RETURN = 0
    CLI_CONTRACT_INVALID = 1
    ACTIVATION_CONFIGURATION_REJECTED = 2
    STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED = 3
    RESOURCE_INITIALIZATION_FAILED = 4
    MODEL_INITIALIZATION_FAILED = 5
    WARMUP_FAILED = 6
    APP_COMPOSITION_FAILED = 7
    SERVER_OR_LIFESPAN_FAILED = 8


@dataclass(frozen=True)
class BoundedDynamicAgentServiceSourceFingerprint:
    source_key: str
    size_bytes: int
    sha256: str

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalBoundedDynamicAgentServicePaths:
    stage157_validation: Path
    router_source: Path
    runtime_source: Path
    stage128_protocol: Path
    stage125_protocol: Path
    stage80_report: Path
    documents: Path
    model_snapshot: Path

    @classmethod
    def from_settings(
        cls,
        settings: ProjectSettings,
    ) -> CanonicalBoundedDynamicAgentServicePaths:
        if settings.bounded_dynamic_agent_model_snapshot is None:
            raise ValueError("bounded dynamic Agent model snapshot is required")
        artifact_dir = settings.artifact_dir.resolve()
        project_root = Path(__file__).resolve().parents[3]
        return cls(
            stage157_validation=(
                artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_runtime_stage157.json"
            ),
            router_source=(
                project_root
                / "src"
                / "ts_rag_agent"
                / "application"
                / "primeqa_hybrid_structured_decision_router.py"
            ),
            runtime_source=(
                project_root
                / "src"
                / "ts_rag_agent"
                / "application"
                / "primeqa_hybrid_bounded_dynamic_agent_runtime.py"
            ),
            stage128_protocol=(
                artifact_dir / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json"
            ),
            stage125_protocol=(
                artifact_dir
                / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json"
            ),
            stage80_report=(
                artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json"
            ),
            documents=(
                settings.primeqa_raw_dir.resolve()
                / "TechQA"
                / "training_and_dev"
                / "training_dev_technotes.sections.json"
            ),
            model_snapshot=settings.bounded_dynamic_agent_model_snapshot,
        )


@dataclass(frozen=True)
class BoundedDynamicAgentWarmupSummary:
    selected_action: str
    terminal_state: str
    retrieval_call_count: int
    model_decision_count: int
    composition_call_count: int
    verification_call_count: int
    diagnostic_observation_count: int
    input_token_count: int
    output_token_count: int
    generation_latency_ms: float
    completed_turn_count_before_close: int
    retained_state_bytes_before_close: int
    thread_opened_after_close: bool

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedBoundedDynamicAgentService:
    app: Any
    runtime: PrimeQAHybridBoundedDynamicAgentRuntime
    backend: Qwen3VLTransformersTextGenerationBackend
    source_fingerprints: tuple[BoundedDynamicAgentServiceSourceFingerprint, ...]
    resource_factory_build_count: int
    retrieval_encoder_device: str
    warmup: BoundedDynamicAgentWarmupSummary
    timing_seconds: Mapping[str, float]


@dataclass(frozen=True)
class PublicSafeBoundedDynamicAgentServiceTerminalEvent:
    entrypoint_id: str
    outcome_code: str
    exit_code: int
    binding_host: str
    binding_port: int
    source_authorized: bool
    activation_authorized: bool
    resources_initialized: bool
    model_initialized: bool
    warmup_completed: bool
    app_composed: bool
    server_started: bool
    source_fingerprint_count: int
    queue_action_count: int = 0
    retry_action_count: int = 0
    fallback_action_count: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDynamicAgentServiceResult:
    exit_code: BoundedDynamicAgentServiceExitCode
    terminal_event: PublicSafeBoundedDynamicAgentServiceTerminalEvent


class BoundedDynamicAgentServiceTerminalEventSink(Protocol):
    def emit(self, event: PublicSafeBoundedDynamicAgentServiceTerminalEvent) -> None: ...


class JsonLineBoundedDynamicAgentServiceTerminalEventSink:
    def emit(self, event: PublicSafeBoundedDynamicAgentServiceTerminalEvent) -> None:
        print(json.dumps(event.to_public_dict(), ensure_ascii=True, separators=(",", ":")))


class BoundedDynamicAgentServer(Protocol):
    started: bool

    def run(self) -> None: ...


class BoundedDynamicAgentServerFactory(Protocol):
    def create(self, config: uvicorn.Config) -> BoundedDynamicAgentServer: ...


class UvicornBoundedDynamicAgentServerFactory:
    def create(self, config: uvicorn.Config) -> BoundedDynamicAgentServer:
        return uvicorn.Server(config=config)


class BoundedDynamicAgentServiceSourceAuthorizer(Protocol):
    def authorize(
        self,
        paths: CanonicalBoundedDynamicAgentServicePaths,
    ) -> tuple[BoundedDynamicAgentServiceSourceFingerprint, ...]: ...


class ExactStage157ServiceSourceAuthorizer:
    def authorize(
        self,
        paths: CanonicalBoundedDynamicAgentServicePaths,
    ) -> tuple[BoundedDynamicAgentServiceSourceFingerprint, ...]:
        stage157 = _load_json_object(paths.stage157_validation)
        return _authorize_stage157_and_sources(stage157=stage157, paths=paths)


class BoundedDynamicAgentResourceFactoryProvider(Protocol):
    def create(self, paths: CanonicalBoundedDynamicAgentServicePaths) -> Any: ...


class FrozenBoundedDynamicAgentResourceFactoryProvider:
    def create(self, paths: CanonicalBoundedDynamicAgentServicePaths) -> Any:
        return PrimeQAHybridProcessRuntimeResourceFactory(
            stage128_protocol_path=paths.stage128_protocol,
            stage125_protocol_path=paths.stage125_protocol,
            stage80_report_path=paths.stage80_report,
            documents_path=paths.documents,
            encoder_device="cpu",
        )


class BoundedDynamicAgentBackendLoader(Protocol):
    def load(self, snapshot_path: Path) -> Qwen3VLTransformersTextGenerationBackend: ...


class ExactLocalQwenBackendLoader:
    def load(self, snapshot_path: Path) -> Qwen3VLTransformersTextGenerationBackend:
        return Qwen3VLTransformersTextGenerationBackend.load_local(
            snapshot_path=snapshot_path,
        )


class PrimeQAHybridBoundedDynamicAgentServiceEntrypoint:
    """Strict composition root for the separate Stage158 local service."""

    def __init__(
        self,
        *,
        settings: ProjectSettings,
        paths: CanonicalBoundedDynamicAgentServicePaths | None = None,
        app_factory: Callable[..., Any] = create_primeqa_hybrid_bounded_dynamic_agent_http_app,
        config_factory: Callable[..., uvicorn.Config] = (
            create_primeqa_hybrid_bounded_dynamic_agent_uvicorn_config
        ),
        source_authorizer: BoundedDynamicAgentServiceSourceAuthorizer | None = None,
        resource_factory_provider: BoundedDynamicAgentResourceFactoryProvider | None = None,
        backend_loader: BoundedDynamicAgentBackendLoader | None = None,
        server_factory: BoundedDynamicAgentServerFactory | None = None,
        event_sink: BoundedDynamicAgentServiceTerminalEventSink | None = None,
    ) -> None:
        self._settings = settings
        self._paths = paths
        self._app_factory = app_factory
        self._config_factory = config_factory
        self._source_authorizer = source_authorizer or ExactStage157ServiceSourceAuthorizer()
        self._resource_factory_provider = (
            resource_factory_provider or FrozenBoundedDynamicAgentResourceFactoryProvider()
        )
        self._backend_loader = backend_loader or ExactLocalQwenBackendLoader()
        self._server_factory = server_factory or UvicornBoundedDynamicAgentServerFactory()
        self._event_sink = event_sink or JsonLineBoundedDynamicAgentServiceTerminalEventSink()

    def prepare(self) -> PreparedBoundedDynamicAgentService:
        if not (
            self._settings.enable_bounded_dynamic_agent_runtime
            and self._settings.enable_bounded_dynamic_agent_http_transport
        ):
            raise BoundedDynamicAgentStartupError(
                BoundedDynamicAgentServiceExitCode.ACTIVATION_CONFIGURATION_REJECTED
            )
        try:
            paths = self._paths or CanonicalBoundedDynamicAgentServicePaths.from_settings(
                self._settings
            )
        except Exception as error:
            raise BoundedDynamicAgentStartupError(
                BoundedDynamicAgentServiceExitCode.ACTIVATION_CONFIGURATION_REJECTED
            ) from error

        started_at = time.perf_counter()
        fingerprints = self._source_authorizer.authorize(paths)
        source_authorized_at = time.perf_counter()

        try:
            resource_factory = self._resource_factory_provider.create(paths)
            resources = resource_factory.build_shared()
        except Exception as error:
            raise BoundedDynamicAgentStartupError(
                BoundedDynamicAgentServiceExitCode.RESOURCE_INITIALIZATION_FAILED
            ) from error
        resources_built_at = time.perf_counter()

        try:
            backend = self._backend_loader.load(paths.model_snapshot)
            router = StrictStructuredDecisionRouter(backend=backend)
        except Exception as error:
            raise BoundedDynamicAgentStartupError(
                BoundedDynamicAgentServiceExitCode.MODEL_INITIALIZATION_FAILED
            ) from error
        model_loaded_at = time.perf_counter()

        runtime = create_primeqa_hybrid_bounded_dynamic_agent_runtime(
            candidate_pool_retriever=resources.candidate_pool_retriever,
            decision_router=router,
        )
        try:
            warmup = _run_label_free_warmup(runtime)
        except Exception as error:
            raise BoundedDynamicAgentStartupError(
                BoundedDynamicAgentServiceExitCode.WARMUP_FAILED
            ) from error
        warmup_completed_at = time.perf_counter()

        try:
            app = self._app_factory(
                settings=self._settings,
                runtime=runtime,
            )
        except Exception as error:
            raise BoundedDynamicAgentStartupError(
                BoundedDynamicAgentServiceExitCode.APP_COMPOSITION_FAILED
            ) from error
        app_composed_at = time.perf_counter()
        return PreparedBoundedDynamicAgentService(
            app=app,
            runtime=runtime,
            backend=backend,
            source_fingerprints=fingerprints,
            resource_factory_build_count=resource_factory.build_count,
            retrieval_encoder_device="cpu",
            warmup=warmup,
            timing_seconds={
                "source_authorization": round(source_authorized_at - started_at, 6),
                "retrieval_resource_build": round(
                    resources_built_at - source_authorized_at,
                    6,
                ),
                "model_load": round(model_loaded_at - resources_built_at, 6),
                "warmup": round(warmup_completed_at - model_loaded_at, 6),
                "app_composition": round(app_composed_at - warmup_completed_at, 6),
                "total_prepare": round(app_composed_at - started_at, 6),
            },
        )

    def run(self, *, port: int) -> BoundedDynamicAgentServiceResult:
        if not _MIN_PORT <= port <= _MAX_PORT:
            return self._emit_result(
                port=port,
                code=BoundedDynamicAgentServiceExitCode.CLI_CONTRACT_INVALID,
                outcome="cli_contract_invalid",
            )
        try:
            prepared = self.prepare()
        except BoundedDynamicAgentStartupError as error:
            return self._emit_result(
                port=port,
                code=error.exit_code,
                outcome=error.exit_code.name.lower(),
            )
        try:
            config = self._config_factory(app=prepared.app, port=port)
            server = self._server_factory.create(config)
            server.run()
        except (Exception, SystemExit):
            return self._emit_result(
                port=port,
                code=BoundedDynamicAgentServiceExitCode.SERVER_OR_LIFESPAN_FAILED,
                outcome="server_or_lifespan_failed",
                prepared=prepared,
                server_started=bool(getattr(locals().get("server"), "started", False)),
            )
        if not server.started:
            return self._emit_result(
                port=port,
                code=BoundedDynamicAgentServiceExitCode.SERVER_OR_LIFESPAN_FAILED,
                outcome="server_or_lifespan_failed",
                prepared=prepared,
                server_started=False,
            )
        return self._emit_result(
            port=port,
            code=BoundedDynamicAgentServiceExitCode.CLEAN_SERVER_RETURN,
            outcome="clean_server_return",
            prepared=prepared,
            server_started=bool(server.started),
        )

    def _emit_result(
        self,
        *,
        port: int,
        code: BoundedDynamicAgentServiceExitCode,
        outcome: str,
        prepared: PreparedBoundedDynamicAgentService | None = None,
        server_started: bool = False,
    ) -> BoundedDynamicAgentServiceResult:
        progress = _startup_progress(code=code, prepared=prepared)
        event = PublicSafeBoundedDynamicAgentServiceTerminalEvent(
            entrypoint_id=BOUNDED_DYNAMIC_AGENT_SERVICE_ENTRYPOINT_ID,
            outcome_code=outcome,
            exit_code=int(code),
            binding_host=BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
            binding_port=port,
            source_authorized=progress["source_authorized"],
            activation_authorized=progress["activation_authorized"],
            resources_initialized=progress["resources_initialized"],
            model_initialized=progress["model_initialized"],
            warmup_completed=progress["warmup_completed"],
            app_composed=progress["app_composed"],
            server_started=server_started,
            source_fingerprint_count=(len(prepared.source_fingerprints) if prepared else 0),
        )
        self._event_sink.emit(event)
        return BoundedDynamicAgentServiceResult(exit_code=code, terminal_event=event)


class BoundedDynamicAgentStartupError(RuntimeError):
    def __init__(self, exit_code: BoundedDynamicAgentServiceExitCode) -> None:
        self.exit_code = exit_code
        super().__init__(exit_code.name.lower())


def builtin_bounded_dynamic_agent_warmup_query() -> PrimeQARuntimeQuery:
    return PrimeQARuntimeQuery(
        id="bounded-dynamic-agent-service-warmup",
        title="Service installation verification",
        text="How can I verify a service configuration after installation?",
    )


def _run_label_free_warmup(
    runtime: PrimeQAHybridBoundedDynamicAgentRuntime,
) -> BoundedDynamicAgentWarmupSummary:
    handle = "stage158-warmup-thread"
    runtime.open_thread(handle)
    try:
        run = runtime.run_turn(
            opaque_thread_handle=handle,
            question=builtin_bounded_dynamic_agent_warmup_query(),
        )
        before_close = runtime.thread_summary(handle)
    finally:
        after_close = runtime.close_thread(handle)
    metrics = run.workflow_run.router_metrics
    if metrics is None:
        raise RuntimeError("bounded dynamic service warmup returned no router metrics")
    trace = run.public_safe_trace
    return BoundedDynamicAgentWarmupSummary(
        selected_action=trace.selected_action,
        terminal_state=trace.terminal_state,
        retrieval_call_count=trace.retrieval_call_count,
        model_decision_count=trace.model_decision_count,
        composition_call_count=trace.composition_call_count,
        verification_call_count=trace.verification_call_count,
        diagnostic_observation_count=trace.diagnostic_observation_count,
        input_token_count=metrics.input_token_count,
        output_token_count=metrics.output_token_count,
        generation_latency_ms=metrics.generation_latency_ms,
        completed_turn_count_before_close=before_close.completed_turn_count,
        retained_state_bytes_before_close=before_close.retained_state_bytes,
        thread_opened_after_close=after_close.opened,
    )


def _authorize_stage157_and_sources(
    *,
    stage157: Mapping[str, Any],
    paths: CanonicalBoundedDynamicAgentServicePaths,
) -> tuple[BoundedDynamicAgentServiceSourceFingerprint, ...]:
    stage157_fingerprint = _fingerprint("stage157_validation", paths.stage157_validation)
    checks = stage157.get("guard_checks") or []
    decision = stage157.get("decision") or {}
    if not (
        stage157_fingerprint.sha256 == EXPECTED_STAGE157_ARTIFACT_SHA256
        and stage157.get("stage") == "Stage 157"
        and decision.get("status") == EXPECTED_STAGE157_STATUS
        and len(checks) == EXPECTED_STAGE157_GUARD_COUNT
        and all(check.get("passed") is True for check in checks)
        and decision.get("test_gate_opened") is False
        and decision.get("test_metrics_run") is False
    ):
        raise BoundedDynamicAgentStartupError(
            BoundedDynamicAgentServiceExitCode.STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED
        )
    router = _fingerprint("structured_router_source", paths.router_source)
    runtime = _fingerprint("bounded_dynamic_runtime_source", paths.runtime_source)
    if (
        router.sha256 != EXPECTED_ROUTER_SOURCE_SHA256
        or runtime.sha256 != EXPECTED_RUNTIME_SOURCE_SHA256
    ):
        raise BoundedDynamicAgentStartupError(
            BoundedDynamicAgentServiceExitCode.STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED
        )
    expected_model = (stage157.get("source_files") or {}).get("model_snapshot_files") or {}
    model_files = (
        ("model_config", paths.model_snapshot / "config.json", expected_model.get("config")),
        (
            "model_weights",
            paths.model_snapshot / "model.safetensors",
            expected_model.get("weights"),
        ),
        (
            "model_tokenizer",
            paths.model_snapshot / "tokenizer.json",
            expected_model.get("tokenizer"),
        ),
    )
    model_fingerprints: list[BoundedDynamicAgentServiceSourceFingerprint] = []
    for source_key, path, expected in model_files:
        fingerprint = _fingerprint(source_key, path)
        if not isinstance(expected, Mapping) or (
            fingerprint.sha256 != expected.get("sha256")
            or fingerprint.size_bytes != expected.get("size_bytes")
        ):
            raise BoundedDynamicAgentStartupError(
                BoundedDynamicAgentServiceExitCode.STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED
            )
        model_fingerprints.append(fingerprint)
    retrieval_fingerprints = tuple(
        _fingerprint(source_key, path)
        for source_key, path in (
            ("stage128_protocol", paths.stage128_protocol),
            ("stage125_protocol", paths.stage125_protocol),
            ("stage80_report", paths.stage80_report),
            ("technote_documents", paths.documents),
        )
    )
    return (
        stage157_fingerprint,
        router,
        runtime,
        *model_fingerprints,
        *retrieval_fingerprints,
    )


def parse_exact_bounded_dynamic_agent_service_cli(arguments: Sequence[str]) -> int:
    if len(arguments) != 2 or arguments[0] != "--port":
        raise ValueError("expected exact CLI shape: --port PORT")
    raw = arguments[1]
    if not raw.isascii() or not raw.isdecimal() or raw.startswith("+"):
        raise ValueError("port must be canonical ASCII decimal")
    port = int(raw)
    if not _MIN_PORT <= port <= _MAX_PORT:
        raise ValueError("port must be between 1024 and 65535")
    return port


def bounded_dynamic_agent_service_entrypoint_contract() -> dict[str, Any]:
    return {
        "entrypoint_id": BOUNDED_DYNAMIC_AGENT_SERVICE_ENTRYPOINT_ID,
        "required_activation_flags": [
            "TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_RUNTIME",
            "TS_RAG_ENABLE_BOUNDED_DYNAMIC_AGENT_HTTP_TRANSPORT",
        ],
        "required_model_snapshot_setting": "TS_RAG_BOUNDED_DYNAMIC_AGENT_MODEL_SNAPSHOT",
        "default_enabled": False,
        "binding_host": BOUNDED_DYNAMIC_AGENT_BINDING_HOST,
        "minimum_port": _MIN_PORT,
        "maximum_port": _MAX_PORT,
        "stage157_artifact_sha256": EXPECTED_STAGE157_ARTIFACT_SHA256,
        "stage157_guard_count": EXPECTED_STAGE157_GUARD_COUNT,
        "router_source_sha256": EXPECTED_ROUTER_SOURCE_SHA256,
        "runtime_source_sha256": EXPECTED_RUNTIME_SOURCE_SHA256,
        "source_gate_before_resource_build": True,
        "resource_build_before_model_load": True,
        "warmup_before_listener": True,
        "warmup_thread_closed_before_listener": True,
        "retrieval_encoder_device": "cpu",
        "router_gpu_device": "cuda:0",
        "runtime_registered_as_default": False,
        "existing_service_changed": False,
        "remote_exposure_authorized": False,
        "persistent_state_enabled": False,
        "test_access_allowed": False,
        "queue_actions_allowed": False,
        "retry_actions_allowed": False,
        "fallback_strategies_allowed": False,
    }


def _startup_progress(
    *,
    code: BoundedDynamicAgentServiceExitCode,
    prepared: PreparedBoundedDynamicAgentService | None,
) -> dict[str, bool]:
    if prepared is not None:
        return {
            "activation_authorized": True,
            "source_authorized": True,
            "resources_initialized": True,
            "model_initialized": True,
            "warmup_completed": True,
            "app_composed": True,
        }
    activation_authorized = code not in {
        BoundedDynamicAgentServiceExitCode.CLI_CONTRACT_INVALID,
        BoundedDynamicAgentServiceExitCode.ACTIVATION_CONFIGURATION_REJECTED,
    }
    source_authorized = code in {
        BoundedDynamicAgentServiceExitCode.RESOURCE_INITIALIZATION_FAILED,
        BoundedDynamicAgentServiceExitCode.MODEL_INITIALIZATION_FAILED,
        BoundedDynamicAgentServiceExitCode.WARMUP_FAILED,
        BoundedDynamicAgentServiceExitCode.APP_COMPOSITION_FAILED,
    }
    resources_initialized = code in {
        BoundedDynamicAgentServiceExitCode.MODEL_INITIALIZATION_FAILED,
        BoundedDynamicAgentServiceExitCode.WARMUP_FAILED,
        BoundedDynamicAgentServiceExitCode.APP_COMPOSITION_FAILED,
    }
    model_initialized = code in {
        BoundedDynamicAgentServiceExitCode.WARMUP_FAILED,
        BoundedDynamicAgentServiceExitCode.APP_COMPOSITION_FAILED,
    }
    warmup_completed = code is BoundedDynamicAgentServiceExitCode.APP_COMPOSITION_FAILED
    return {
        "activation_authorized": activation_authorized,
        "source_authorized": source_authorized,
        "resources_initialized": resources_initialized,
        "model_initialized": model_initialized,
        "warmup_completed": warmup_completed,
        "app_composed": False,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BoundedDynamicAgentStartupError(
            BoundedDynamicAgentServiceExitCode.STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED
        ) from error
    if not isinstance(value, dict):
        raise BoundedDynamicAgentStartupError(
            BoundedDynamicAgentServiceExitCode.STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED
        )
    return value


def _fingerprint(
    source_key: str,
    path: Path,
) -> BoundedDynamicAgentServiceSourceFingerprint:
    try:
        digest = hashlib.sha256()
        with path.expanduser().resolve(strict=True).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        size_bytes = path.stat().st_size
    except Exception as error:
        raise BoundedDynamicAgentStartupError(
            BoundedDynamicAgentServiceExitCode.STAGE157_OR_SOURCE_AUTHORIZATION_REJECTED
        ) from error
    return BoundedDynamicAgentServiceSourceFingerprint(
        source_key=source_key,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )
