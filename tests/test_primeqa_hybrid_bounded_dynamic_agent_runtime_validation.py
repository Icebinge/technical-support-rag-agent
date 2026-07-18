from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime_validation import (
    _canonical_synthetic_runtime_cases,
    _fingerprint,
    _load_json_object,
    _stage156_summary,
    write_stage157_visualizations,
)

_ROOT = Path(__file__).resolve().parents[1]
_STAGE156 = _ROOT / "artifacts" / "primeqa_hybrid_bounded_agent_state_protocol_stage156.json"


def test_stage157_canonical_synthetic_cases_cover_both_branches_and_rejections() -> None:
    cases = _canonical_synthetic_runtime_cases()

    assert set(cases) == {
        "compose_branch",
        "early_refuse_branch",
        "malformed_schema_rejected",
        "unauthorized_action_rejected",
        "thread_state_isolated",
    }
    assert all(case["passed"] for case in cases.values())
    serialized = json.dumps(cases)
    assert "Generated private validation question" not in serialized
    assert "synthetic-thread-a" not in serialized


def test_stage157_source_gate_accepts_exact_stage156_artifact() -> None:
    report = _load_json_object(_STAGE156)
    summary = _stage156_summary(report, _fingerprint(_STAGE156))

    assert summary == {
        "identity_exact": True,
        "fingerprint_exact": True,
        "guard_count": 43,
        "passed_guard_count": 43,
        "all_guards_passed": True,
        "runtime_registered_as_default": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
    }


def test_stage157_visualizations_are_ten_parseable_svgs(tmp_path: Path) -> None:
    report = {
        "guard_checks": [
            {"name": "synthetic_guard_a", "passed": True},
            {"name": "synthetic_guard_b", "passed": True},
        ],
        "stage156_summary": {"passed_guard_count": 43, "guard_count": 43},
        "source_unchanged_after_validation": True,
        "graph_topology": {
            "node_count": 9,
            "conditional_edge_count": 1,
            "conditional_target_edge_count": 2,
            "compile_count": 1,
        },
        "synthetic_runtime_cases": _canonical_synthetic_runtime_cases(),
        "real_non_test_runtime_probe": {
            "runtime_trace": {
                "model_decision_count": 1,
                "retrieval_call_count": 1,
                "composition_call_count": 1,
                "verification_call_count": 1,
                "diagnostic_observation_count": 1,
                "completed_turn_count": 1,
                "retained_state_bytes": 128,
            },
            "router_metrics": {"input_token_count": 700, "output_token_count": 9},
            "thread_summary_after_close": {"opened": False},
        },
        "model_runtime": {"peak_gpu_memory_bytes": 4_500_000_000},
        "timing_seconds": {
            "model_load": 12.0,
            "retrieval_resource_build": 20.0,
            "real_turn": 1.0,
            "total": 33.0,
        },
        "closed_boundaries": {
            "test_split_loaded": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "persistent_state_enabled": False,
            "second_retrieval_enabled": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }

    visualizations = write_stage157_visualizations(report=report, output_dir=tmp_path)

    assert len(visualizations) == 10
    for visualization in visualizations:
        path = Path(visualization.path)
        assert path.stat().st_size > 100
        assert ET.parse(path).getroot().tag.endswith("svg")
