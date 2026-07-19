from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_train_history_isolation_protocol as protocol
from ts_rag_agent.application import primeqa_hybrid_train_history_isolation_validation as stage165
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 165 transition-label correction"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_train_history_isolation_transition_correction_v1"
_ORIGINAL_PUBLIC_SHA256 = "f152653c5c85e03add95c06ff898ff442744733078c3339b662fa406d17874a1"
_ORIGINAL_PRIVATE_BYTE_SHA256 = "ce4b5b281093319696a51251d475a3fc5fa6b7dac2e7f9659464fe1d8e55ad1b"
_ORIGINAL_PRIVATE_CANONICAL_SHA256 = (
    "2f1047124b56714180fb42654d389aa4fd30640ee2c6380c9b45df55df2bf784"
)
_ORIGINAL_ANALYSIS_ID = "primeqa_hybrid_train_history_isolation_sharded_diagnostics_v2"
_ORIGINAL_STATUS = "primeqa_hybrid_train_history_isolation_not_train_safe"
_PRIVATE_ARTIFACT_ID = "primeqa_hybrid_stage165_train_history_isolation_private_v1"
_WORSENED_KEY = "synthetic_refusal_to_isolated_false_answer_count"
_IMPROVED_KEY = "synthetic_false_answer_to_isolated_refusal_count"
_DIRECTIONAL_KEYS = frozenset({_WORSENED_KEY, _IMPROVED_KEY})


@dataclass(frozen=True)
class Stage165TransitionCorrectionVisualization:
    name: str
    path: str


def run_stage165_transition_correction(
    *,
    original_public_report_path: Path,
    original_private_report_path: Path,
    user_confirmed_stage_continuation: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Correct directional labels from immutable Stage165 artifacts without rerunning data."""

    fingerprints_before = {
        "original_public": _fingerprint(original_public_report_path),
        "original_private": _fingerprint(original_private_report_path),
    }
    _authorize_fingerprints(fingerprints_before)
    original = _load_json_object(original_public_report_path)
    original_private = _load_json_object(original_private_report_path)
    _authorize_reports(original=original, original_private=original_private)

    observations = tuple(protocol.Stage165ArmObservation(**row) for row in original_private["rows"])
    corrected_diagnostics = protocol.summarize_stage165_pairs(observations)
    original_diagnostics = original["paired_diagnostics"]
    original_snapshot = _without_directional_transition_counts(original_diagnostics)
    corrected_snapshot = _without_directional_transition_counts(corrected_diagnostics)
    original_snapshot_sha256 = _canonical_json_sha256(original_snapshot)
    corrected_snapshot_sha256 = _canonical_json_sha256(corrected_snapshot)
    corrected_context = {**original, "paired_diagnostics": corrected_diagnostics}
    corrected_decision = stage165._decision(corrected_context)

    original_safety = original_diagnostics["unanswerable_post_first_safety_effect"]
    corrected_safety = corrected_diagnostics["unanswerable_post_first_safety_effect"]
    corrected_section_count = _directional_summary_count(corrected_diagnostics)
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Read-only correction of two reversed directional transition labels in the "
            "immutable Stage165 train-only paired diagnostics. The original artifacts are "
            "preserved. No split, corpus, retrieval, Agent, model, feature fit, threshold "
            "search, development, or test evaluation is loaded or executed."
        ),
        "user_confirmation": {
            "stage_continuation_confirmed": bool(user_confirmed_stage_continuation),
            "confirmation_note": confirmation_note,
            "correction_required_before_next_analysis": True,
        },
        "source_authorization": {
            "fingerprints_before": fingerprints_before,
            "original_private_canonical_sha256": _canonical_json_sha256(original_private),
            "original_artifacts_remain_canonical": True,
        },
        "correction": {
            "affected_function": "_unanswerable_pair_effect",
            "affected_semantics": "directional_transition_labels_only",
            "corrected_summary_section_count": corrected_section_count,
            "original_post_first_unanswerable": {
                _WORSENED_KEY: int(original_safety[_WORSENED_KEY]),
                _IMPROVED_KEY: int(original_safety[_IMPROVED_KEY]),
            },
            "corrected_post_first_unanswerable": {
                _WORSENED_KEY: int(corrected_safety[_WORSENED_KEY]),
                _IMPROVED_KEY: int(corrected_safety[_IMPROVED_KEY]),
            },
            "correct_semantics": {
                _WORSENED_KEY: ("synthetic_history refused and isolated answered falsely"),
                _IMPROVED_KEY: ("synthetic_history answered falsely and isolated refused"),
            },
        },
        "metric_integrity": {
            "direction_excluded_snapshot_sha256_before": original_snapshot_sha256,
            "direction_excluded_snapshot_sha256_after": corrected_snapshot_sha256,
            "direction_excluded_snapshot_changed": original_snapshot != corrected_snapshot,
            "false_answer_rate_difference_before": original_safety[
                "false_answer_rate_difference_isolated_minus_synthetic"
            ],
            "false_answer_rate_difference_after": corrected_safety[
                "false_answer_rate_difference_isolated_minus_synthetic"
            ],
            "mcnemar_p_before": original_safety["mcnemar_exact_two_sided_p"],
            "mcnemar_p_after": corrected_safety["mcnemar_exact_two_sided_p"],
            "original_decision": original["decision"],
            "corrected_decision": corrected_decision,
            "decision_changed": original["decision"] != corrected_decision,
        },
        "corrected_paired_diagnostics": corrected_diagnostics,
        "execution_counts": {
            "public_reports_loaded": 1,
            "private_reports_loaded": 1,
            "observation_rows_read": len(observations),
            "train_rows_loaded": 0,
            "development_rows_loaded": 0,
            "test_rows_loaded": 0,
            "documents_loaded": 0,
            "retrieval_runs": 0,
            "agent_runs": 0,
            "model_generation_runs": 0,
            "feature_fit_runs": 0,
            "threshold_search_runs": 0,
        },
        "closed_boundaries": {
            "train_loaded": False,
            "development_loaded": False,
            "test_loaded": False,
            "retrieval_run": False,
            "agent_run": False,
            "model_run": False,
            "policy_selected": False,
            "runtime_registered_as_default": False,
            "fallback_strategies_enabled": False,
        },
    }
    report["guard_checks"] = _correction_guards(report)
    report["public_safe_contract"] = stage165._public_safe_contract(report)
    all_passed = all(check["passed"] for check in report["guard_checks"])
    report["decision"] = {
        "status": (
            "primeqa_hybrid_stage165_transition_correction_completed"
            if all_passed
            else "primeqa_hybrid_stage165_transition_correction_invalid"
        ),
        "all_correction_guards_passed": all_passed,
        "failed_correction_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "stage165_candidate_status_unchanged": corrected_decision["status"],
        "candidate_eligible_for_frozen_dev_validation": corrected_decision[
            "candidate_eligible_for_frozen_dev_validation"
        ],
        "diagnostic_only": True,
        "policy_selected": False,
        "development_gate_opened": False,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": "design_train_only_runtime_feature_safety_gate_diagnostics",
    }

    fingerprints_after = {
        "original_public": _fingerprint(original_public_report_path),
        "original_private": _fingerprint(original_private_report_path),
    }
    if fingerprints_after != fingerprints_before:
        raise ValueError("original Stage165 artifacts changed during correction")
    report["source_authorization"]["fingerprints_after"] = fingerprints_after
    report["source_authorization"]["original_artifacts_unchanged_after_correction"] = True
    return report


def write_stage165_transition_correction_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[Stage165TransitionCorrectionVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = report["correction"]["original_post_first_unanswerable"]
    corrected = report["correction"]["corrected_post_first_unanswerable"]
    specs = {
        "stage165_corrected_unanswerable_transitions.svg": (
            "Stage165 corrected post-first unanswerable transitions",
            [
                _bar("isolation worsened false answer", corrected[_WORSENED_KEY]),
                _bar("isolation improved to refusal", corrected[_IMPROVED_KEY]),
                _bar(
                    "discordant pairs",
                    int(corrected[_WORSENED_KEY]) + int(corrected[_IMPROVED_KEY]),
                ),
            ],
        ),
        "stage165_transition_label_before_after.svg": (
            "Stage165 transition-label correction audit",
            [
                _bar("original labeled worsened", original[_WORSENED_KEY]),
                _bar("corrected worsened", corrected[_WORSENED_KEY]),
                _bar("original labeled improved", original[_IMPROVED_KEY]),
                _bar("corrected improved", corrected[_IMPROVED_KEY]),
            ],
        ),
    }
    artifacts = []
    for filename, (title, bars) in specs.items():
        path = output_dir / filename
        path.write_text(
            render_horizontal_bar_chart_svg(title=title, bars=bars, x_label="paired train rows"),
            encoding="utf-8",
        )
        artifacts.append(Stage165TransitionCorrectionVisualization(name=filename, path=str(path)))
    return artifacts


def _authorize_fingerprints(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    expected = {
        "original_public": _ORIGINAL_PUBLIC_SHA256,
        "original_private": _ORIGINAL_PRIVATE_BYTE_SHA256,
    }
    mismatches = {
        name: fingerprints[name]["sha256"]
        for name, expected_sha256 in expected.items()
        if fingerprints[name]["sha256"] != expected_sha256
    }
    if mismatches:
        raise ValueError(f"Stage165 correction source fingerprint mismatch: {mismatches}")


def _authorize_reports(
    *,
    original: Mapping[str, Any],
    original_private: Mapping[str, Any],
) -> None:
    if original.get("analysis_id") != _ORIGINAL_ANALYSIS_ID:
        raise ValueError("Stage165 correction requires the exact original public analysis")
    if original.get("decision", {}).get("status") != _ORIGINAL_STATUS:
        raise ValueError("Stage165 correction requires the original unsafe decision")
    if len(original.get("guard_checks", [])) != 23 or not all(
        check.get("passed") is True for check in original["guard_checks"]
    ):
        raise ValueError("Stage165 correction requires the original 23/23 process guards")
    if original_private.get("artifact_id") != _PRIVATE_ARTIFACT_ID:
        raise ValueError("Stage165 correction private artifact id mismatch")
    if original_private.get("arm_row_count") != 1124:
        raise ValueError("Stage165 correction requires 1124 private arm rows")
    if len(original_private.get("rows", [])) != 1124:
        raise ValueError("Stage165 correction private row count mismatch")
    if _canonical_json_sha256(original_private) != _ORIGINAL_PRIVATE_CANONICAL_SHA256:
        raise ValueError("Stage165 correction private canonical content mismatch")
    safety = original["paired_diagnostics"]["unanswerable_post_first_safety_effect"]
    if safety.get(_WORSENED_KEY) != 3 or safety.get(_IMPROVED_KEY) != 22:
        raise ValueError("Stage165 correction requires the known reversed 3/22 labels")


def _correction_guards(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = report["source_authorization"]
    correction = report["correction"]
    integrity = report["metric_integrity"]
    counts = report["execution_counts"]
    closed = report["closed_boundaries"]
    corrected = correction["corrected_post_first_unanswerable"]
    return [
        _check(
            "user_confirmed_stage_continuation",
            report["user_confirmation"]["stage_continuation_confirmed"] is True,
        ),
        _check(
            "original_artifact_fingerprints_exact",
            source["fingerprints_before"]["original_public"]["sha256"] == _ORIGINAL_PUBLIC_SHA256
            and source["fingerprints_before"]["original_private"]["sha256"]
            == _ORIGINAL_PRIVATE_BYTE_SHA256
            and source["original_private_canonical_sha256"] == _ORIGINAL_PRIVATE_CANONICAL_SHA256,
        ),
        _check(
            "all_private_observations_read_once",
            counts["private_reports_loaded"] == 1 and counts["observation_rows_read"] == 1124,
        ),
        _check(
            "directional_transition_labels_corrected",
            corrected[_WORSENED_KEY] == 22 and corrected[_IMPROVED_KEY] == 3,
        ),
        _check(
            "discordant_transition_total_preserved",
            sum(correction["original_post_first_unanswerable"].values())
            == sum(corrected.values())
            == 25,
        ),
        _check(
            "non_directional_metric_snapshot_unchanged",
            integrity["direction_excluded_snapshot_changed"] is False
            and integrity["direction_excluded_snapshot_sha256_before"]
            == integrity["direction_excluded_snapshot_sha256_after"],
        ),
        _check(
            "aggregate_safety_statistics_unchanged",
            integrity["false_answer_rate_difference_before"]
            == integrity["false_answer_rate_difference_after"]
            == 0.126667
            and integrity["mcnemar_p_before"] == integrity["mcnemar_p_after"] == 0.000157,
        ),
        _check(
            "stage165_decision_unchanged",
            integrity["decision_changed"] is False
            and integrity["corrected_decision"]["status"] == _ORIGINAL_STATUS
            and integrity["corrected_decision"]["candidate_eligible_for_frozen_dev_validation"]
            is False,
        ),
        _check(
            "no_data_retrieval_agent_model_fit_or_search_rerun",
            all(
                counts[name] == 0
                for name in (
                    "train_rows_loaded",
                    "development_rows_loaded",
                    "test_rows_loaded",
                    "documents_loaded",
                    "retrieval_runs",
                    "agent_runs",
                    "model_generation_runs",
                    "feature_fit_runs",
                    "threshold_search_runs",
                )
            ),
        ),
        _check(
            "development_test_runtime_and_fallback_closed",
            closed["development_loaded"] is False
            and closed["test_loaded"] is False
            and closed["policy_selected"] is False
            and closed["runtime_registered_as_default"] is False
            and closed["fallback_strategies_enabled"] is False,
        ),
    ]


def _without_directional_transition_counts(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_directional_transition_counts(nested)
            for key, nested in value.items()
            if key not in _DIRECTIONAL_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_without_directional_transition_counts(item) for item in value]
    return value


def _directional_summary_count(value: Any) -> int:
    if isinstance(value, Mapping):
        current = int(_DIRECTIONAL_KEYS <= set(value))
        return current + sum(_directional_summary_count(nested) for nested in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_directional_summary_count(item) for item in value)
    return 0


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bar(label: str, value: int | float) -> BarDatum:
    return BarDatum(label=label, value=float(value), value_label=str(value))


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}
