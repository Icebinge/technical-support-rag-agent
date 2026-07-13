from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    cross_validated_candidate_reranker_selections,
    split_validated_candidate_reranker_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    CandidateRerankerPolicyConfig,
)
from ts_rag_agent.application.candidate_score_guarded_policy_evaluation import (
    STAGE39_MAIN_POLICY_LABEL,
    CandidateScoreGuardedPolicyEvaluationResult,
    VisualizationArtifact,
    default_stage39_policy_specs,
    evaluate_candidate_score_guarded_policies_from_selections,
    write_candidate_score_guarded_policy_visualizations,
)
from ts_rag_agent.application.guarded_candidate_reranker_answer_experiment import (
    SINGLE_CANDIDATE_MODE,
    GuardedCandidateAnswerMetrics,
)


@dataclass(frozen=True)
class CandidateScoreGuardedPolicySplitValidationResult:
    """Stage 40 split-respecting candidate-score policy validation."""

    model_name: str
    train_split: str
    evaluation_split: str
    train_fold_count: int
    max_answer_candidates: int
    train_question_count: int
    evaluation_question_count: int
    train_cv_evaluation: CandidateScoreGuardedPolicyEvaluationResult
    holdout_evaluation: CandidateScoreGuardedPolicyEvaluationResult
    findings: list[str]
    analysis_scope: str


@dataclass(frozen=True)
class CandidateScoreGuardedPolicySplitValidationVisualizations:
    """SVG artifacts generated for Stage 40."""

    train_cv: list[VisualizationArtifact]
    holdout: list[VisualizationArtifact]


CANDIDATE_SCORE_GTE_60_LABEL = "candidate_score_gte_60"


def evaluate_candidate_score_guarded_policy_split_validation(
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    model_name: str = "logistic_best_candidate",
    train_split: str = "train",
    evaluation_split: str = "dev",
    train_fold_count: int = 5,
    max_answer_candidates: int = 3,
    policies: Sequence[tuple[str, CandidateRerankerPolicyConfig]] | None = None,
) -> CandidateScoreGuardedPolicySplitValidationResult:
    """Evaluate candidate-score guarded policies with a train-to-validation boundary."""

    if train_fold_count < 2:
        raise ValueError("train_fold_count must be at least 2")
    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")
    normalized_train_split = _normalize_split_name(train_split)
    normalized_evaluation_split = _normalize_split_name(evaluation_split)
    if normalized_train_split == normalized_evaluation_split:
        raise ValueError("train_split and evaluation_split must be different")

    train_rows = _rows_for_split(rows, normalized_train_split)
    evaluation_rows = _rows_for_split(rows, normalized_evaluation_split)
    train_question_count = _question_count(train_rows)
    evaluation_question_count = _question_count(evaluation_rows)
    policy_specs = tuple(default_stage39_policy_specs() if policies is None else policies)
    if not policy_specs:
        raise ValueError("policies must not be empty")

    train_cv_selections = cross_validated_candidate_reranker_selections(
        rows=train_rows,
        model_name=model_name,
        fold_count=train_fold_count,
    )
    train_cv_evaluation = evaluate_candidate_score_guarded_policies_from_selections(
        selections=train_cv_selections,
        rows=train_rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        model_name=model_name,
        max_answer_candidates=max_answer_candidates,
        policies=policy_specs,
        selection_scope="train_only_grouped_cv",
        fold_count=train_fold_count,
        train_split=normalized_train_split,
        evaluation_split=normalized_train_split,
        train_question_count=train_question_count,
        evaluation_question_count=train_question_count,
        analysis_scope=(
            "Offline train-only grouped-CV policy evidence. Only the training split "
            "is used for this stage of candidate-score policy observation. Runtime "
            "behavior is not changed."
        ),
    )

    holdout_selections = split_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        train_split=normalized_train_split,
        validation_split=normalized_evaluation_split,
    )
    holdout_evaluation = evaluate_candidate_score_guarded_policies_from_selections(
        selections=holdout_selections,
        rows=evaluation_rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        model_name=model_name,
        max_answer_candidates=max_answer_candidates,
        policies=policy_specs,
        selection_scope="train_to_validation_holdout",
        fold_count=None,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        train_question_count=train_question_count,
        evaluation_question_count=evaluation_question_count,
        analysis_scope=(
            "Offline split-respecting holdout evaluation. The candidate reranker is "
            "fit only on the training split and evaluated only on the validation "
            "split. Held-out test data is not used, and runtime behavior is not "
            "changed."
        ),
    )

    findings = _findings(
        train_cv_evaluation=train_cv_evaluation,
        holdout_evaluation=holdout_evaluation,
    )
    return CandidateScoreGuardedPolicySplitValidationResult(
        model_name=model_name,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        train_fold_count=train_fold_count,
        max_answer_candidates=max_answer_candidates,
        train_question_count=train_question_count,
        evaluation_question_count=evaluation_question_count,
        train_cv_evaluation=train_cv_evaluation,
        holdout_evaluation=holdout_evaluation,
        findings=findings,
        analysis_scope=(
            "Stage 40 validates fixed candidate-score guarded policies with a split "
            "boundary: train-only grouped-CV evidence first, then train-to-validation "
            "holdout evaluation. No held-out test set is used and no runtime default "
            "is changed."
        ),
    )


def candidate_score_guarded_policy_split_validation_to_dict(
    result: CandidateScoreGuardedPolicySplitValidationResult,
) -> dict[str, Any]:
    """Convert a Stage 40 split validation result to a JSON-safe dictionary."""

    return asdict(result)


def write_candidate_score_guarded_policy_split_validation_visualizations(
    result: CandidateScoreGuardedPolicySplitValidationResult,
    output_dir: Path,
) -> CandidateScoreGuardedPolicySplitValidationVisualizations:
    """Write SVG charts for Stage 40 split-respecting validation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return CandidateScoreGuardedPolicySplitValidationVisualizations(
        train_cv=write_candidate_score_guarded_policy_visualizations(
            result=result.train_cv_evaluation,
            output_dir=output_dir,
            artifact_prefix="stage40_train_cv",
            title_prefix="Stage 40 train-only CV",
        ),
        holdout=write_candidate_score_guarded_policy_visualizations(
            result=result.holdout_evaluation,
            output_dir=output_dir,
            artifact_prefix="stage40_holdout",
            title_prefix=(
                f"Stage 40 {result.train_split}-to-{result.evaluation_split} holdout"
            ),
        ),
    )


def _findings(
    train_cv_evaluation: CandidateScoreGuardedPolicyEvaluationResult,
    holdout_evaluation: CandidateScoreGuardedPolicyEvaluationResult,
) -> list[str]:
    train_topk_mode = _topk_mode_name(train_cv_evaluation)
    holdout_topk_mode = _topk_mode_name(holdout_evaluation)
    train_best_label, train_best_metrics = _best_policy_by_delta(
        train_cv_evaluation,
        train_topk_mode,
    )
    holdout_best_label, holdout_best_metrics = _best_policy_by_delta(
        holdout_evaluation,
        holdout_topk_mode,
    )
    findings = [
        (
            f"Train-only CV best {train_topk_mode} policy is {train_best_label}: "
            f"delta {train_best_metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {train_best_metrics.regressed_count}."
        ),
        (
            f"Holdout best {holdout_topk_mode} policy is {holdout_best_label}: "
            f"delta {holdout_best_metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {holdout_best_metrics.regressed_count}."
        ),
    ]
    if _has_policy(holdout_evaluation, CANDIDATE_SCORE_GTE_60_LABEL):
        main_holdout = _mode_metrics(
            holdout_evaluation,
            STAGE39_MAIN_POLICY_LABEL,
            holdout_topk_mode,
        )
        score60_holdout = _mode_metrics(
            holdout_evaluation,
            CANDIDATE_SCORE_GTE_60_LABEL,
            holdout_topk_mode,
        )
        findings.append(
            f"candidate_score_gte_60 changes holdout {holdout_topk_mode} delta from "
            f"{main_holdout.average_delta_vs_baseline:+.4f} to "
            f"{score60_holdout.average_delta_vs_baseline:+.4f}, regressions from "
            f"{main_holdout.regressed_count} to {score60_holdout.regressed_count}, "
            f"and gold citation delta from {main_holdout.gold_citation_delta:+d} to "
            f"{score60_holdout.gold_citation_delta:+d}."
        )
        main_single = _mode_metrics(
            holdout_evaluation,
            STAGE39_MAIN_POLICY_LABEL,
            SINGLE_CANDIDATE_MODE,
        )
        score60_single = _mode_metrics(
            holdout_evaluation,
            CANDIDATE_SCORE_GTE_60_LABEL,
            SINGLE_CANDIDATE_MODE,
        )
        findings.append(
            "candidate_score_gte_60 holdout single-candidate delta diff versus main is "
            f"{_delta_diff(score60_single, main_single):+.4f}."
        )
    return findings


def _rows_for_split(
    rows: Sequence[Mapping[str, Any]],
    split: str,
) -> list[Mapping[str, Any]]:
    split_rows = [row for row in rows if str(row["split"]).lower() == split]
    if not split_rows:
        raise ValueError(f"No rows found for split: {split}")
    return split_rows


def _question_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({f"{row['split']}::{row['question_id']}" for row in rows})


def _topk_mode_name(result: CandidateScoreGuardedPolicyEvaluationResult) -> str:
    for mode_name in _mode_names(result):
        if mode_name != SINGLE_CANDIDATE_MODE:
            return mode_name
    raise ValueError("evaluation result must include a top-k mode")


def _mode_names(result: CandidateScoreGuardedPolicyEvaluationResult) -> list[str]:
    return [evaluation.mode_name for evaluation in result.policies[0].mode_evaluations]


def _best_policy_by_delta(
    result: CandidateScoreGuardedPolicyEvaluationResult,
    mode_name: str,
) -> tuple[str, GuardedCandidateAnswerMetrics]:
    return max(
        (
            (policy.label, _mode_metrics(result, policy.label, mode_name))
            for policy in result.policies
        ),
        key=lambda item: (
            item[1].average_delta_vs_baseline,
            -item[1].regressed_count,
            item[1].gold_citation_delta,
        ),
    )


def _mode_metrics(
    result: CandidateScoreGuardedPolicyEvaluationResult,
    policy_label: str,
    mode_name: str,
) -> GuardedCandidateAnswerMetrics:
    for policy in result.policies:
        if policy.label == policy_label:
            for evaluation in policy.mode_evaluations:
                if evaluation.mode_name == mode_name:
                    return evaluation.metrics
    raise ValueError(f"Missing policy/mode metrics: {policy_label} / {mode_name}")


def _delta_diff(
    candidate_metrics: GuardedCandidateAnswerMetrics,
    main_metrics: GuardedCandidateAnswerMetrics,
) -> float:
    return round(
        candidate_metrics.average_delta_vs_baseline
        - main_metrics.average_delta_vs_baseline,
        4,
    )


def _has_policy(
    result: CandidateScoreGuardedPolicyEvaluationResult,
    policy_label: str,
) -> bool:
    return any(policy.label == policy_label for policy in result.policies)


def _normalize_split_name(split_name: str) -> str:
    normalized = split_name.strip().lower()
    if not normalized:
        raise ValueError("split name must not be empty")
    return normalized
