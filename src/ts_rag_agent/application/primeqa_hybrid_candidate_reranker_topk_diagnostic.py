from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    DEFAULT_MODEL_NAMES,
    CandidateRerankerCVResult,
    write_cv_visualizations,
)
from ts_rag_agent.application.candidate_score_guarded_policy_evaluation import (
    VisualizationArtifact,
    write_candidate_score_guarded_policy_visualizations,
)
from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    CANDIDATE_SCORE_GTE_60_LABEL,
    CandidateScoreGuardedPolicySplitValidationResult,
)
from ts_rag_agent.application.primeqa_hybrid_candidate_reranker_development import (
    run_primeqa_hybrid_candidate_reranker_development,
)

_STAGE = "Stage 73"
_CREATED_AT = "2026-07-15"
_DEFAULT_MAX_ANSWER_CANDIDATES = 10


@dataclass(frozen=True)
class PrimeQAHybridCandidateRerankerTopKDiagnosticVisualization:
    """One generated Stage73 top-k diagnostic visualization."""

    group: str
    name: str
    path: str


@dataclass(frozen=True)
class PrimeQAHybridCandidateRerankerTopKDiagnosticRun:
    """In-memory Stage73 top-k diagnostic run."""

    report: dict[str, Any]
    train_cv_result: CandidateRerankerCVResult
    split_validation_results: list[CandidateScoreGuardedPolicySplitValidationResult]


def run_primeqa_hybrid_candidate_reranker_topk_diagnostic(
    *,
    candidate_dataset_path: Path,
    candidate_summary_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    fold_count: int = 5,
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES,
    policy_model_names: tuple[str, ...] | None = None,
    max_answer_candidates: int = _DEFAULT_MAX_ANSWER_CANDIDATES,
) -> PrimeQAHybridCandidateRerankerTopKDiagnosticRun:
    """Run a train/dev-only top-k answer proxy diagnostic for reranker policies."""

    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")

    base_run = run_primeqa_hybrid_candidate_reranker_development(
        candidate_dataset_path=candidate_dataset_path,
        candidate_summary_path=candidate_summary_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        fold_count=fold_count,
        model_names=model_names,
        policy_model_names=policy_model_names,
        max_answer_candidates=max_answer_candidates,
    )
    report = deepcopy(base_run.report)
    topk_summary = _topk_diagnostic_summary(
        report=report,
        max_answer_candidates=max_answer_candidates,
    )
    stage73_guard_checks = _stage73_guard_checks(
        report=report,
        max_answer_candidates=max_answer_candidates,
    )
    all_guard_checks = [*report["guard_checks"], *stage73_guard_checks]
    report.update(
        {
            "stage": _STAGE,
            "created_at": _CREATED_AT,
            "analysis_scope": (
                f"Train/dev-only top{max_answer_candidates} candidate-reranker "
                "answer proxy diagnostic for primeqa_hybrid_stage68_v1. This "
                "stage reuses the Stage71 split-respecting runner with a larger "
                "answer-candidate window, keeps the frozen test split locked, "
                "does not run final metrics, and does not change the default "
                "runtime."
            ),
            "source_stage": {
                "runner": "Stage 71 candidate-reranker development runner",
                "reason": (
                    "Reuse the existing train/dev-only split validation path while "
                    "recording a separate Stage73 top-k diagnostic report."
                ),
            },
            "topk_diagnostic": topk_summary,
            "guard_checks": all_guard_checks,
            "decision": _decision(
                guard_checks=all_guard_checks,
                topk_summary=topk_summary,
                max_answer_candidates=max_answer_candidates,
            ),
        }
    )
    return PrimeQAHybridCandidateRerankerTopKDiagnosticRun(
        report=report,
        train_cv_result=base_run.train_cv_result,
        split_validation_results=base_run.split_validation_results,
    )


def write_primeqa_hybrid_candidate_reranker_topk_diagnostic_visualizations(
    run: PrimeQAHybridCandidateRerankerTopKDiagnosticRun,
    output_dir: Path,
) -> list[PrimeQAHybridCandidateRerankerTopKDiagnosticVisualization]:
    """Write Stage73 top-k diagnostic SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    visualizations: list[PrimeQAHybridCandidateRerankerTopKDiagnosticVisualization] = []
    for artifact in write_cv_visualizations(
        result=run.train_cv_result,
        output_dir=output_dir,
    ):
        visualizations.append(
            PrimeQAHybridCandidateRerankerTopKDiagnosticVisualization(
                group="train_only_model_cv",
                name=artifact.name,
                path=artifact.path,
            )
        )

    for split_validation_result in run.split_validation_results:
        model_token = _model_token(split_validation_result.model_name)
        for artifact in write_candidate_score_guarded_policy_visualizations(
            result=split_validation_result.train_cv_evaluation,
            output_dir=output_dir,
            artifact_prefix=f"stage73_{model_token}_train_cv",
            title_prefix=f"Stage 73 {model_token} train-only CV",
        ):
            visualizations.append(_visualization_from_artifact(
                group=f"{split_validation_result.model_name}_train_cv_policy",
                artifact=artifact,
            ))
        for artifact in write_candidate_score_guarded_policy_visualizations(
            result=split_validation_result.holdout_evaluation,
            output_dir=output_dir,
            artifact_prefix=f"stage73_{model_token}_dev_holdout",
            title_prefix=f"Stage 73 {model_token} train-to-dev holdout",
        ):
            visualizations.append(_visualization_from_artifact(
                group=f"{split_validation_result.model_name}_train_to_dev_policy",
                artifact=artifact,
            ))
    return visualizations


def _topk_diagnostic_summary(
    *,
    report: dict[str, Any],
    max_answer_candidates: int,
) -> dict[str, Any]:
    topk_mode_name = f"top{max_answer_candidates}_leading_candidate_rewrite"
    train_cv_policy_rows = []
    dev_holdout_policy_rows = []
    for split_validation in report["train_to_dev_policy_validations"]:
        model_name = split_validation["model_name"]
        train_cv_policy_rows.extend(
            _policy_rows_for_evaluation(
                model_name=model_name,
                evaluation=split_validation["train_cv_evaluation"],
                topk_mode_name=topk_mode_name,
            )
        )
        dev_holdout_policy_rows.extend(
            _policy_rows_for_evaluation(
                model_name=model_name,
                evaluation=split_validation["holdout_evaluation"],
                topk_mode_name=topk_mode_name,
            )
        )

    best_by_delta = max(
        dev_holdout_policy_rows,
        key=lambda row: (
            row["average_delta_vs_baseline"],
            -row["regressed_count"],
            row["gold_citation_delta"],
        ),
    )
    lowest_regression = min(
        dev_holdout_policy_rows,
        key=lambda row: (
            row["regressed_count"],
            -row["average_delta_vs_baseline"],
            -row["gold_citation_delta"],
        ),
    )
    return {
        "max_answer_candidates": max_answer_candidates,
        "topk_mode_name": topk_mode_name,
        "train_cv_policy_rows": train_cv_policy_rows,
        "dev_holdout_policy_rows": dev_holdout_policy_rows,
        "policy_rows": dev_holdout_policy_rows,
        "best_policy_by_dev_topk_delta": best_by_delta,
        "lowest_regression_policy": lowest_regression,
        "max_dev_topk_delta": best_by_delta["average_delta_vs_baseline"],
        "min_regressed_count": lowest_regression["regressed_count"],
        "selected_reference_policy": _selected_reference_policy(
            dev_holdout_policy_rows
        ),
    }


def _policy_rows_for_evaluation(
    *,
    model_name: str,
    evaluation: dict[str, Any],
    topk_mode_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for policy in evaluation["policies"]:
        metrics = _mode_metrics(policy=policy, mode_name=topk_mode_name)
        rows.append(
            {
                "model_name": model_name,
                "selection_scope": evaluation["selection_scope"],
                "policy_label": policy["label"],
                "average_delta_vs_baseline": metrics["average_delta_vs_baseline"],
                "policy_average_answer_token_f1": (
                    metrics["policy_average_answer_token_f1"]
                ),
                "baseline_average_answer_token_f1": (
                    metrics["baseline_average_answer_token_f1"]
                ),
                "oracle_average_answer_token_f1": (
                    metrics["oracle_average_answer_token_f1"]
                ),
                "oracle_gap_closed_rate": metrics["oracle_gap_closed_rate"],
                "replacement_count": metrics["replacement_count"],
                "changed_answer_count": metrics["changed_answer_count"],
                "improved_count": metrics["improved_count"],
                "regressed_count": metrics["regressed_count"],
                "tied_count": metrics["tied_count"],
                "gold_citation_delta": metrics["gold_citation_delta"],
                "citation_lost_count": metrics["citation_lost_count"],
                "citation_gained_count": metrics["citation_gained_count"],
            }
        )
    return rows


def _selected_reference_policy(policy_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in policy_rows:
        if (
            row["model_name"] == "logistic_best_candidate"
            and row["policy_label"] == CANDIDATE_SCORE_GTE_60_LABEL
        ):
            return row
    return None


def _mode_metrics(*, policy: dict[str, Any], mode_name: str) -> dict[str, Any]:
    for mode_evaluation in policy["mode_evaluations"]:
        if mode_evaluation["mode_name"] == mode_name:
            return mode_evaluation["metrics"]
    raise ValueError(f"Missing mode {mode_name!r} for policy {policy['label']!r}")


def _stage73_guard_checks(
    *,
    report: dict[str, Any],
    max_answer_candidates: int,
) -> list[dict[str, Any]]:
    observed_topk_values = sorted(
        {
            split_validation["max_answer_candidates"]
            for split_validation in report["train_to_dev_policy_validations"]
        }
    )
    observed_split_pairs = sorted(
        {
            (
                split_validation["train_split"],
                split_validation["evaluation_split"],
            )
            for split_validation in report["train_to_dev_policy_validations"]
        }
    )
    return [
        {
            "name": "stage73_topk_window_matches_requested_value",
            "passed": observed_topk_values == [max_answer_candidates],
            "observed": observed_topk_values,
            "expected": [max_answer_candidates],
        },
        {
            "name": "stage73_split_validations_are_train_to_dev",
            "passed": observed_split_pairs == [("train", "dev")],
            "observed": [
                {"train_split": train_split, "evaluation_split": evaluation_split}
                for train_split, evaluation_split in observed_split_pairs
            ],
            "expected": {"train_split": "train", "evaluation_split": "dev"},
        },
        {
            "name": "stage73_final_test_metrics_not_run",
            "passed": True,
            "observed": "not_run",
            "expected": "not_run",
        },
        {
            "name": "stage73_default_runtime_policy_unchanged",
            "passed": True,
            "observed": "unchanged",
            "expected": "unchanged",
        },
    ]


def _decision(
    *,
    guard_checks: list[dict[str, Any]],
    topk_summary: dict[str, Any],
    max_answer_candidates: int,
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_candidate_reranker_topk_diagnostic_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
            "recommended_next_stage": (
                "Fix Stage73 train/dev top-k diagnostic guard failures before any "
                "further reranker development."
            ),
        }
    return {
        "status": "primeqa_hybrid_candidate_reranker_topk_diagnostic_completed",
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 74: choose whether to refine train/dev reranker policy gates "
            "using top3/top10 diagnostics, without using test for evaluation or tuning."
        ),
        "reason": (
            f"Stage73 is a train/dev-only top{max_answer_candidates} diagnostic. "
            "It does not open a final-test gate and does not change runtime defaults."
        ),
        "observed_best_dev_topk_delta": topk_summary["max_dev_topk_delta"],
        "observed_min_regressed_count": topk_summary["min_regressed_count"],
    }


def _visualization_from_artifact(
    *,
    group: str,
    artifact: VisualizationArtifact,
) -> PrimeQAHybridCandidateRerankerTopKDiagnosticVisualization:
    return PrimeQAHybridCandidateRerankerTopKDiagnosticVisualization(
        group=group,
        name=artifact.name,
        path=artifact.path,
    )


def _model_token(model_name: str) -> str:
    return model_name.replace("-", "_").replace(".", "_")


def write_stage73_report(
    *,
    report: dict[str, Any],
    visualizations: list[PrimeQAHybridCandidateRerankerTopKDiagnosticVisualization],
    output_path: Path,
) -> dict[str, Any]:
    """Write the Stage73 report with visualization metadata included."""

    report_with_visualizations = {
        **report,
        "visualizations": [
            {
                "group": visualization.group,
                "name": visualization.name,
                "path": visualization.path,
            }
            for visualization in visualizations
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_with_visualizations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_with_visualizations
