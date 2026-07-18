from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability_validation import (
    RecordingAgentWorkflowObservationSink,
    run_primeqa_hybrid_agent_runtime_observability_validation,
    write_primeqa_hybrid_agent_runtime_observability_visualizations,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow_validation import (
    _question,
    _StaticRetriever,
    _toolset,
)
from ts_rag_agent.config import ProjectSettings

_ROOT = Path(__file__).resolve().parents[1]
_APPLICATION = _ROOT / "src" / "ts_rag_agent" / "application"
_ARTIFACTS = _ROOT / "artifacts"


def test_unconfirmed_preflight_runs_synthetic_checks_but_skips_real_lifecycle(
    tmp_path: Path,
) -> None:
    report = _run(tmp_path=tmp_path, confirmed=False)

    assert report["stage"] == "Stage 155"
    assert report["synthetic_validation"]["complete"]["event_count"] == 9
    assert report["synthetic_validation"]["concurrency_four"]["event_count"] == 36
    assert report["real_resource_service_lifecycle"]["executed"] is False
    assert report["real_observation"]["event_count"] == 0
    assert report["decision"]["status"] == "stage155_validation_blocked"
    assert "stage155_user_confirmed" in report["decision"]["failed_checks"]
    assert "real_lifecycle_executed" in report["decision"]["failed_checks"]
    assert report["public_safe_contract"]["test_metrics_run"] is False


def test_confirmed_validation_accepts_exact_fake_real_evidence(tmp_path: Path) -> None:
    report = _run(tmp_path=tmp_path, confirmed=True)

    assert all(check["passed"] is True for check in report["guard_checks"])
    assert report["decision"]["status"] == (
        "primeqa_hybrid_agent_runtime_activation_observability_validated"
    )
    assert report["decision"]["activation_protocol_frozen"] is True
    assert report["real_resource_service_lifecycle"]["exit_code"] == 0
    assert report["real_observation"]["invocation_count"] == 2
    assert report["real_observation"]["event_count"] == 18
    assert report["real_observation"]["node_event_count"] == 14
    assert report["real_observation"]["failed_event_count"] == 0
    assert report["source_unchanged_after_validation"] is True
    assert report["public_safe_contract"]["forbidden_keys_found"] == []


def test_stage155_visualizations_write_ten_parseable_svgs(tmp_path: Path) -> None:
    report = _run(tmp_path=tmp_path, confirmed=True)

    artifacts = write_primeqa_hybrid_agent_runtime_observability_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(artifacts) == 10
    for artifact in artifacts:
        path = Path(artifact.path)
        assert path.is_file()
        assert ET.parse(path).getroot().tag.endswith("svg")


def test_stage155_report_contains_no_private_request_or_document_content(
    tmp_path: Path,
) -> None:
    report = _run(tmp_path=tmp_path, confirmed=True)
    serialized = json.dumps(report, ensure_ascii=False)

    assert "private-stage155" not in serialized
    assert '"document_id":' not in serialized
    assert '"question_text":' not in serialized
    assert '"raw_answer_text":' not in serialized
    assert report["public_safe_contract"]["request_content_saved"] is False
    assert report["public_safe_contract"]["document_identifiers_saved"] is False


def _run(*, tmp_path: Path, confirmed: bool) -> dict[str, Any]:
    stage154_path = _current_stage154_fixture(tmp_path)
    return run_primeqa_hybrid_agent_runtime_observability_validation(
        stage154_validation_path=stage154_path,
        stage153_protocol_path=(
            _ARTIFACTS / "primeqa_hybrid_agent_tool_orchestration_protocol_stage153.json"
        ),
        pyproject_path=_ROOT / "pyproject.toml",
        workflow_source_path=_APPLICATION / "primeqa_hybrid_agent_tool_workflow.py",
        concurrent_runtime_source_path=(
            _APPLICATION / "primeqa_hybrid_concurrent_sidecar_agent_runtime.py"
        ),
        observability_source_path=(_APPLICATION / "primeqa_hybrid_agent_runtime_observability.py"),
        service_entrypoint_source_path=(
            _APPLICATION / "primeqa_hybrid_agent_service_entrypoint.py"
        ),
        settings=ProjectSettings(
            _env_file=None,
            data_dir=_ROOT / "data",
            artifact_dir=_ARTIFACTS,
            enable_concurrent_sidecar_agent=True,
            enable_local_agent_http_transport=True,
        ),
        port=18155,
        user_confirmed_validation=confirmed,
        confirmation_note="test-only synthetic lifecycle",
        real_lifecycle_runner=_fake_real_lifecycle,
    )


def _current_stage154_fixture(tmp_path: Path) -> Path:
    source_path = _ARTIFACTS / "primeqa_hybrid_agent_tool_workflow_validation_stage154.json"
    report = json.loads(source_path.read_text(encoding="utf-8"))
    paths = {
        "stage153_protocol": (
            _ARTIFACTS / "primeqa_hybrid_agent_tool_orchestration_protocol_stage153.json"
        ),
        "pyproject": _ROOT / "pyproject.toml",
        "workflow_source": _APPLICATION / "primeqa_hybrid_agent_tool_workflow.py",
        "concurrent_runtime_source": (
            _APPLICATION / "primeqa_hybrid_concurrent_sidecar_agent_runtime.py"
        ),
    }
    report["source_files"] = {name: _fingerprint(path) for name, path in paths.items()}
    report["source_unchanged_after_validation"] = True
    fixture = tmp_path / "stage154-current.json"
    fixture.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return fixture


def _fake_real_lifecycle(
    *,
    settings: ProjectSettings,
    port: int,
    workflow_observation_sink: RecordingAgentWorkflowObservationSink,
) -> dict[str, Any]:
    _ = settings, port
    workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=_toolset(_StaticRetriever(prefix="real-fixture"), refuse=False),
        observation_sink=workflow_observation_sink,
    )
    workflow.run(_question("test-real-warmup"))
    workflow.run(_question("test-real-answer"))
    return {
        "executed": True,
        "exit_code": 0,
        "source_fingerprints": [
            {"source_key": f"source-{index}", "size_bytes": 1, "sha256": "0" * 64}
            for index in range(11)
        ],
        "http_probe": {
            "liveness_status": 200,
            "readiness_status": 200,
            "answer_status": 200,
        },
        "transport_closed": True,
        "listener_released": True,
        "test_metrics_run": False,
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
