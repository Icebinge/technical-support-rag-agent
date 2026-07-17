from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_agent_service_entrypoint_validation import (
    run_primeqa_hybrid_agent_service_entrypoint_validation,
    write_primeqa_hybrid_agent_service_entrypoint_validation_visualizations,
)
from ts_rag_agent.config import ProjectSettings

_ROOT = Path(__file__).resolve().parents[1]


def test_stage152_preflight_runs_source_gate_and_synthetic_cases_only() -> None:
    report = run_primeqa_hybrid_agent_service_entrypoint_validation(
        stage151_protocol_path=(
            _ROOT / "artifacts" / "primeqa_hybrid_agent_service_entrypoint_protocol_stage151.json"
        ),
        settings=_settings(),
        port=18152,
        user_confirmed_validation=False,
        confirmation_note="test preflight",
    )

    assert report["source_gate_passed"] is True
    assert len(report["synthetic_composition_cases"]) == 9
    assert all(row["passed"] for row in report["synthetic_composition_cases"].values())
    assert report["real_resource_service_lifecycle"] == {
        "executed": False,
        "reason": "source_gate_or_user_confirmation_not_satisfied",
    }
    assert report["public_safe_contract"]["train_split_loaded"] is False
    assert report["public_safe_contract"]["dev_split_loaded"] is False
    assert report["public_safe_contract"]["test_split_loaded"] is False
    assert report["public_safe_contract"]["test_metrics_run"] is False
    assert report["decision"]["status"] == "primeqa_hybrid_agent_service_entrypoint_rejected"
    assert "stage152_user_confirmed" in report["decision"]["failed_checks"]
    assert "real_source_fingerprints_sha256" in report["decision"]["failed_checks"]


def test_stage152_visualizations_are_ten_parseable_svg_files(tmp_path: Path) -> None:
    report = run_primeqa_hybrid_agent_service_entrypoint_validation(
        stage151_protocol_path=(
            _ROOT / "artifacts" / "primeqa_hybrid_agent_service_entrypoint_protocol_stage151.json"
        ),
        settings=_settings(),
        port=18152,
        user_confirmed_validation=False,
        confirmation_note="test visualization preflight",
    )

    visualizations = write_primeqa_hybrid_agent_service_entrypoint_validation_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        assert Path(visualization.path).is_file()
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def _settings() -> ProjectSettings:
    return ProjectSettings(
        data_dir=_ROOT / "data",
        artifact_dir=_ROOT / "artifacts",
        enable_optional_sidecar_agent=False,
        enable_concurrent_sidecar_agent=True,
        enable_local_agent_http_transport=True,
    )
