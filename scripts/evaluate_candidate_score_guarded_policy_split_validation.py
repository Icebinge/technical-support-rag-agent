from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.candidate_score_guarded_policy_evaluation import (
    CandidateScoreGuardedPolicyEvaluationResult,
)
from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    candidate_score_guarded_policy_split_validation_to_dict,
    evaluate_candidate_score_guarded_policy_split_validation,
    write_candidate_score_guarded_policy_split_validation_visualizations,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

app = typer.Typer(help="Run split-respecting candidate-score guarded policy validation.")


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Candidate reranker model name."),
    ] = "logistic_best_candidate",
    train_split: Annotated[
        str,
        typer.Option("--train-split", help="Split used to fit the candidate reranker."),
    ] = "train",
    evaluation_split: Annotated[
        str,
        typer.Option("--evaluation-split", help="Split used for holdout evaluation."),
    ] = "dev",
    train_fold_count: Annotated[
        int,
        typer.Option(
            "--train-fold-count",
            help="Number of train-only CV folds for policy evidence.",
        ),
    ] = 5,
    max_answer_candidates: Annotated[
        int,
        typer.Option(
            "--max-answer-candidates",
            help="Top-k size for the leading-candidate rewrite proxy.",
        ),
    ] = 3,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output split-validation JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run Stage 40 train-to-validation candidate-score policy validation."""

    _ensure_file_exists(dataset)
    normalized_train_split = _parse_split_name(train_split)
    normalized_evaluation_split = _parse_split_name(evaluation_split)
    _validate_options(
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        train_fold_count=train_fold_count,
        max_answer_candidates=max_answer_candidates,
    )

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / f"candidate_score_guarded_policy_split_validation_{dataset.stem}.json"
    )
    visualization_output_dir = visualization_dir or output_path.with_suffix("")

    rows = load_candidate_reranker_rows(dataset)
    gold_answers = _load_gold_answers(
        settings=settings,
        splits=[normalized_train_split, normalized_evaluation_split],
    )
    result = evaluate_candidate_score_guarded_policy_split_validation(
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name=model,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        train_fold_count=train_fold_count,
        max_answer_candidates=max_answer_candidates,
    )
    visualizations = write_candidate_score_guarded_policy_split_validation_visualizations(
        result=result,
        output_dir=visualization_output_dir,
    )
    result_dict = candidate_score_guarded_policy_split_validation_to_dict(result)
    result_dict["source_paths"] = {
        "dataset": str(dataset),
        "gold_answer_splits": [normalized_train_split, normalized_evaluation_split],
    }
    result_dict["visualizations"] = {
        "train_cv": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations.train_cv
        ],
        "holdout": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations.holdout
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(
        json.dumps(
            {
                "model_name": result.model_name,
                "train_split": result.train_split,
                "evaluation_split": result.evaluation_split,
                "train_fold_count": result.train_fold_count,
                "max_answer_candidates": result.max_answer_candidates,
                "train_question_count": result.train_question_count,
                "evaluation_question_count": result.evaluation_question_count,
                "train_cv": _evaluation_summary(result.train_cv_evaluation),
                "holdout": _evaluation_summary(result.holdout_evaluation),
                "findings": result.findings,
                "output": str(output_path),
                "visualization_dir": str(visualization_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _evaluation_summary(
    result: CandidateScoreGuardedPolicyEvaluationResult,
) -> dict:
    return {
        "selection_scope": result.selection_scope,
        "fold_count": result.fold_count,
        "policies": [
            {
                "label": policy.label,
                "modes": [
                    {
                        "mode_name": evaluation.mode_name,
                        "policy_average_answer_token_f1": (
                            evaluation.metrics.policy_average_answer_token_f1
                        ),
                        "average_delta_vs_baseline": (
                            evaluation.metrics.average_delta_vs_baseline
                        ),
                        "replacement_count": evaluation.metrics.replacement_count,
                        "regressed_count": evaluation.metrics.regressed_count,
                        "citation_lost_count": evaluation.metrics.citation_lost_count,
                        "citation_gained_count": evaluation.metrics.citation_gained_count,
                        "gold_citation_delta": evaluation.metrics.gold_citation_delta,
                    }
                    for evaluation in policy.mode_evaluations
                ],
            }
            for policy in result.policies
        ],
    }


def _load_gold_answers(settings: ProjectSettings, splits: list[str]) -> dict[str, str]:
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    gold_answers = {}
    for split in sorted(set(splits)):
        questions_path = _resolve_questions_path(training_dir, split)
        _ensure_file_exists(questions_path)
        for question in load_primeqa_questions(questions_path):
            if question.answerable:
                gold_answers[f"{split}::{question.id}"] = question.answer
    return gold_answers


def _parse_split_name(raw_split: str) -> str:
    split_name = raw_split.strip().lower()
    if not split_name:
        raise typer.BadParameter("split name must not be empty.")
    allowed = {"dev", "train"}
    if split_name not in allowed:
        raise typer.BadParameter("split must be either dev or train.")
    return split_name


def _validate_options(
    train_split: str,
    evaluation_split: str,
    train_fold_count: int,
    max_answer_candidates: int,
) -> None:
    if train_split == evaluation_split:
        raise typer.BadParameter("--train-split and --evaluation-split must differ.")
    if train_fold_count < 2:
        raise typer.BadParameter("--train-fold-count must be at least 2.")
    if max_answer_candidates <= 0:
        raise typer.BadParameter("--max-answer-candidates must be positive.")


def _resolve_questions_path(training_dir: Path, split: str) -> Path:
    if split == "dev":
        return training_dir / "dev_Q_A.json"
    if split == "train":
        return training_dir / "training_Q_A.json"
    raise typer.BadParameter("split must be either dev or train.")


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
