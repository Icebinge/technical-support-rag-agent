from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow_validation import (
    run_primeqa_hybrid_agent_tool_workflow_validation,
    write_primeqa_hybrid_agent_tool_workflow_validation_visualizations,
)

_ROOT = Path(__file__).resolve().parents[1]
_STAGE153 = _ROOT / "artifacts" / "primeqa_hybrid_agent_tool_orchestration_protocol_stage153.json"
_STAGE152 = _ROOT / "artifacts" / "primeqa_hybrid_agent_service_entrypoint_validation_stage152.json"
_PYPROJECT = _ROOT / "pyproject.toml"
_WORKFLOW = _ROOT / "src" / "ts_rag_agent" / "application" / "primeqa_hybrid_agent_tool_workflow.py"
_CONCURRENT_RUNTIME = (
    _ROOT
    / "src"
    / "ts_rag_agent"
    / "application"
    / "primeqa_hybrid_concurrent_sidecar_agent_runtime.py"
)


def test_preflight_fails_only_confirmation_and_real_support_guards() -> None:
    report = _run(confirmed=False, support=None)

    failed = [check["name"] for check in report["guard_checks"] if not check["passed"]]
    assert failed == [
        "stage154_user_confirmed",
        "real_support_available",
        "real_support_status_valid",
        "real_support_guards_exact",
        "real_resource_lifecycle_executed",
        "real_http_path_200",
        "real_service_released",
        "real_test_metrics_not_run",
    ]
    assert len(report["guard_checks"]) == 54
    assert report["decision"]["workflow_implemented"] is False
    assert report["synthetic_validation"]["facade_http_integration"]["answer_status"] == 200
    assert report["public_safe_contract"]["test_split_loaded"] is False


def test_formal_validation_passes_with_saved_real_lifecycle_support() -> None:
    report = _run(confirmed=True, support=_STAGE152)

    assert len(report["guard_checks"]) == 54
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["status"] == (
        "primeqa_hybrid_langgraph_agent_tool_workflow_implemented_and_validated"
    )
    assert report["decision"]["workflow_implemented"] is True
    assert report["decision"]["langgraph_adapter_validated"] is True
    assert report["decision"]["test_gate_opened"] is False
    assert report["dependency_evidence"]["direct_agent_dependencies"] == ["langgraph==1.2.9"]
    assert report["synthetic_validation"]["complete_equivalence"]["equal"] is True
    assert report["synthetic_validation"]["refuse_equivalence"]["equal"] is True
    assert report["synthetic_validation"]["concurrency"]["isolation_failure_count"] == 0
    assert report["stage152_current_service_support"]["candidate_pool_depth_recorded"] is False


def test_tampered_stage153_source_is_rejected(tmp_path: Path) -> None:
    tampered = json.loads(_STAGE153.read_text(encoding="utf-8"))
    tampered["decision"]["status"] = "tampered"
    path = tmp_path / "stage153.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    report = run_primeqa_hybrid_agent_tool_workflow_validation(
        stage153_protocol_path=path,
        pyproject_path=_PYPROJECT,
        workflow_source_path=_WORKFLOW,
        concurrent_runtime_source_path=_CONCURRENT_RUNTIME,
        stage152_support_validation_path=_STAGE152,
        user_confirmed_validation=True,
        confirmation_note="test tampered source",
    )

    failed = [check["name"] for check in report["guard_checks"] if not check["passed"]]
    assert "stage153_status_valid" in failed
    assert report["decision"]["workflow_implemented"] is False


def test_visualizations_are_ten_parseable_svg_files(tmp_path: Path) -> None:
    report = _run(confirmed=True, support=_STAGE152)

    visualizations = write_primeqa_hybrid_agent_tool_workflow_validation_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def _run(*, confirmed: bool, support: Path | None) -> dict:
    return run_primeqa_hybrid_agent_tool_workflow_validation(
        stage153_protocol_path=_STAGE153,
        pyproject_path=_PYPROJECT,
        workflow_source_path=_WORKFLOW,
        concurrent_runtime_source_path=_CONCURRENT_RUNTIME,
        stage152_support_validation_path=support,
        user_confirmed_validation=confirmed,
        confirmation_note="test Stage154 validation",
    )
