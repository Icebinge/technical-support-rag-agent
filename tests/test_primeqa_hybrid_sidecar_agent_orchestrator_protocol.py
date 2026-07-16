from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator_protocol import (
    _forbidden_keys_found,
    freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol,
    write_primeqa_hybrid_sidecar_agent_orchestrator_protocol_visualizations,
)


def test_protocol_freezes_orchestrator_with_unproven_effectiveness(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path, _stage135_report())

    report = freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol(
        stage135_validation_path=source,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage136 orchestrator protocol",
    )

    assert report["decision"]["status"].endswith("protocol_frozen")
    assert report["decision"]["can_run_stage137_train_dev_validation_now"] is True
    assert report["decision"]["can_claim_citation_verification_effectiveness"] is False
    assert report["decision"]["sidecar_can_generate_answer_text"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert (
        report["frozen_protocol"]["effectiveness_boundary"]["source_train_append_opportunities"]
        == 9
    )
    assert report["frozen_protocol"]["effectiveness_boundary"]["source_train_sidecar_captures"] == 0
    assert report["public_safe_contract"]["forbidden_keys_found"] == []


def test_protocol_blocks_without_confirmation(tmp_path: Path) -> None:
    source = _write_source(tmp_path, _stage135_report())

    report = freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol(
        stage135_validation_path=source,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    assert report["decision"]["status"].endswith("protocol_blocked")
    assert report["decision"]["can_run_stage137_train_dev_validation_now"] is False
    assert "user_confirmed_stage136_protocol" in report["decision"]["failed_checks"]


def test_protocol_blocks_if_stage135_negative_boundary_is_hidden(
    tmp_path: Path,
) -> None:
    source_report = _stage135_report()
    source_report["split_observation_reports"]["train"]["sidecar_incremental_gold_hit_count"] = 1
    source = _write_source(tmp_path, source_report)

    report = freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol(
        stage135_validation_path=source,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage136 orchestrator protocol",
    )

    assert report["decision"]["status"].endswith("protocol_blocked")
    assert "stage135_train_opportunity_boundary_recorded" in report["decision"]["failed_checks"]


def test_protocol_blocks_if_zero_valued_safety_field_is_missing(tmp_path: Path) -> None:
    source_report = _stage135_report()
    del source_report["split_observation_reports"]["train"][
        "primary_context_identity_violation_count"
    ]
    source = _write_source(tmp_path, source_report)

    report = freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol(
        stage135_validation_path=source,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage136 orchestrator protocol",
    )

    assert report["decision"]["status"].endswith("protocol_blocked")
    assert "stage135_primary_identity_preserved" in report["decision"]["failed_checks"]


def test_protocol_blocks_if_zero_valued_capture_field_is_missing(tmp_path: Path) -> None:
    source_report = _stage135_report()
    del source_report["split_observation_reports"]["dev"]["sidecar_incremental_gold_hit_count"]
    source = _write_source(tmp_path, source_report)

    report = freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol(
        stage135_validation_path=source,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage136 orchestrator protocol",
    )

    assert report["decision"]["status"].endswith("protocol_blocked")
    assert "stage135_dev_opportunity_boundary_recorded" in report["decision"]["failed_checks"]


def test_protocol_visualizations_are_public_safe(tmp_path: Path) -> None:
    source = _write_source(tmp_path, _stage135_report())
    report = freeze_primeqa_hybrid_sidecar_agent_orchestrator_protocol(
        stage135_validation_path=source,
        user_confirmed_protocol=True,
        confirmation_note="unit test confirmed Stage136 orchestrator protocol",
    )

    artifacts = write_primeqa_hybrid_sidecar_agent_orchestrator_protocol_visualizations(
        report,
        tmp_path / "visuals",
    )

    assert len(artifacts) == 5
    assert all(Path(artifact.path).exists() for artifact in artifacts)
    assert _forbidden_keys_found(report) == set()


def _write_source(tmp_path: Path, report: dict) -> Path:
    path = tmp_path / "stage135.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _stage135_report() -> dict:
    split_reports = {
        "train": _split_report(rows=562, opportunities=9),
        "dev": _split_report(rows=121, opportunities=1),
    }
    return {
        "stage": "Stage 135",
        "analysis_id": (
            "primeqa_hybrid_stage116_answer_context_stage128_sidecar_observation_validation_v1"
        ),
        "split_observation_reports": split_reports,
        "source_answer_invariance": {
            split: {
                "verified_average_token_f1_delta": 0.0,
                "verified_gold_citation_count_delta": 0,
                "changed_verified_answer_count": 0,
            }
            for split in ("train", "dev")
        },
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(1, 31)],
        "decision": {
            "status": "primeqa_hybrid_sidecar_observation_validation_passed",
            "sidecar_observation_protocol_validated": True,
            "can_implement_train_dev_agent_orchestrator_now": True,
            "direct_stage128_all400_answer_context_remains_blocked": True,
            "sidecar_can_generate_answer_text": False,
            "sidecar_can_replace_primary_context": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {"forbidden_keys_found": []},
    }


def _split_report(*, rows: int, opportunities: int) -> dict:
    return {
        "row_count": rows,
        "primary_context_identity_violation_count": 0,
        "answer_generation_context_identity_violation_count": 0,
        "sidecar_answer_context_leak_count": 0,
        "sidecar_primary_overlap_count": 0,
        "sidecar_observation_availability_rate": 1.0,
        "append_pool_incremental_gold_hit_count": opportunities,
        "sidecar_incremental_gold_hit_count": 0,
        "sidecar_capture_rate_of_append_gold_opportunities": 0.0,
    }
