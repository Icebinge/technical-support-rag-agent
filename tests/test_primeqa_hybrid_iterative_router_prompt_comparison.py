from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.compare_primeqa_hybrid_iterative_router_prompts import app, main
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_iterative_router_prompt_comparison as stage170
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    IterativeDecisionAction,
)


def test_synthetic_profile_ranking_uses_frozen_quality_order() -> None:
    profiles = {
        "b": _synthetic(0.8, 0.9, 0.7),
        "a": _synthetic(0.8, 0.9, 0.7),
        "c": _synthetic(0.7, 1.0, 1.0),
    }

    ranked = stage170.rank_synthetic_profiles(profiles)

    assert ranked == ("a", "b", "c")


def test_train_aggregation_and_fold_stability_keep_private_ids_out() -> None:
    outcomes = (
        _outcome(
            "initial_gold_visible",
            "0",
            IterativeDecisionAction.COMPOSE.value,
            IterativeDecisionAction.COMPOSE.value,
        ),
        _outcome(
            "alternate_only_gold_visible",
            "0",
            IterativeDecisionAction.INSPECT.value,
            IterativeDecisionAction.COMPOSE.value,
        ),
        _outcome(
            "unanswerable",
            "1",
            IterativeDecisionAction.INSPECT.value,
            IterativeDecisionAction.REFUSE.value,
        ),
    )

    aggregate = stage170._aggregate_train_outcomes(outcomes)
    folds = stage170._fold_stability(outcomes)

    assert aggregate["initial_gold_visible"]["initial_action_counts"] == {
        "compose_grounded_answer": 1
    }
    assert aggregate["alternate_only_gold_visible"]["inspect_then_compose_count"] == 1
    assert folds["fold_count"] == 2
    assert folds["folds"]["0"]["alternate_only_inspect_rate"] == 1.0
    assert folds["folds"]["1"]["initial_visible_compose_rate"] is None


def test_final_profile_selection_prefers_gate_count_then_quality() -> None:
    reports = {
        "profile-a": {
            "quality_gate_pass_count": 7,
            "quality_metrics": {
                "synthetic_phase_action_accuracy": 0.9,
                "real_initial_visible_compose_rate": 0.8,
            },
        },
        "profile-b": {
            "quality_gate_pass_count": 8,
            "quality_metrics": {
                "synthetic_phase_action_accuracy": 0.8,
                "real_initial_visible_compose_rate": 0.7,
            },
        },
    }

    assert stage170._select_final_profile(reports) == "profile-b"


def test_visualizations_write_five_parseable_svgs(tmp_path: Path) -> None:
    visualizations = stage170.write_stage170_visualizations(
        report=_visual_report(), output_dir=tmp_path
    )

    assert len(visualizations) == 5
    for visualization in visualizations:
        ET.parse(visualization.path)

    resource_root = ET.parse(tmp_path / "resource_peaks.svg").getroot()
    title = resource_root.find("{http://www.w3.org/2000/svg}title")
    assert title is not None
    assert title.text == "Stage 170 process and GPU resource peaks"


def test_cli_exposes_no_development_or_test_paths() -> None:
    result = CliRunner().invoke(app, ["--help"])
    parameters = inspect.signature(main).parameters

    assert result.exit_code == 0
    assert "model_snapshot" in parameters
    assert not ({"dev", "development", "dev_split"} & set(parameters))
    assert not ({"test", "test_split"} & set(parameters))


def _synthetic(action: float, clarification: float, path: float) -> dict:
    return {
        "phase_action_accuracy": action,
        "clarification_kind_accuracy": clarification,
        "exact_path_accuracy": path,
    }


def _observation(action: str) -> stage169.RouterCallObservation:
    return stage169.RouterCallObservation(
        action=action,
        clarification_kind=None,
        schema_valid=True,
        input_token_count=100,
        output_token_count=10,
        generation_latency_ms=500.0,
        process_working_set_bytes=1,
        process_private_usage_bytes=1,
        system_available_memory_bytes=1,
        gpu_peak_allocated_bytes=1,
        gpu_peak_reserved_bytes=1,
    )


def _outcome(
    stratum: str, fold_id: str, initial_action: str, final_action: str
) -> stage170.ProfileTrainOutcome:
    return stage170.ProfileTrainOutcome(
        stratum=stratum,
        fold_id=fold_id,
        initial=_observation(initial_action),
        final=_observation(final_action),
    )


def _visual_report() -> dict:
    profiles = ("profile-a", "profile-b")
    comparison = {
        profile: {
            "quality_gate_pass_count": 6,
            "quality_metrics": {
                "real_initial_visible_compose_rate": 0.8,
                "real_alternate_only_inspect_rate": 0.7,
                "real_alternate_only_final_compose_rate": 0.8,
                "real_alternate_only_path_success_rate": 0.6,
                "real_insufficient_final_compose_rate": 0.1,
                "latency_ms": {"p95": 1200.0},
            },
        }
        for profile in profiles
    }
    return {
        "synthetic_screen": {
            "ranked_profile_ids": [*profiles, "profile-c"],
            "selected_train_finalists": list(profiles),
            "profiles": {
                "profile-a": {"phase_action_accuracy": 0.9},
                "profile-b": {"phase_action_accuracy": 0.8},
                "profile-c": {"phase_action_accuracy": 0.7},
            },
        },
        "stage169_baseline": {"quality_gate_pass_count": 3},
        "train_comparison": comparison,
        "resource_consumption": {
            "process_peak_working_set_bytes": 7 * 1024**3,
            "process_peak_private_usage_bytes": 12 * 1024**3,
            "gpu_peak_allocated_bytes": 5 * 1024**3,
            "gpu_peak_reserved_bytes": 6 * 1024**3,
        },
    }
