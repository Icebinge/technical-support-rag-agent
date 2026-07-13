from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.runtime_document_risk_risk_aware_selection import (
    runtime_document_risk_risk_aware_selection_to_dict,
    select_risk_aware_runtime_document_risk_proxy,
    write_runtime_document_risk_risk_aware_visualizations,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

app = typer.Typer(help="Select runtime proxy guards with risk-aware train-CV objectives.")


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    stage43_report: Annotated[
        Path,
        typer.Option("--stage43-report", help="Stage 43 runtime proxy JSON report."),
    ],
    stage44_report: Annotated[
        Path,
        typer.Option("--stage44-report", help="Stage 44 train-CV JSON report."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Candidate reranker model name."),
    ] = "logistic_best_candidate",
    train_split: Annotated[
        str,
        typer.Option("--train-split", help="Split used for train-only CV."),
    ] = "train",
    evaluation_split: Annotated[
        str,
        typer.Option("--evaluation-split", help="Split used only for holdout confirmation."),
    ] = "dev",
    train_fold_count: Annotated[
        int,
        typer.Option("--train-fold-count", help="Number of train-only grouped-CV folds."),
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
        typer.Option("--output", help="Output Stage 45 risk-aware JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run Stage 45 risk-aware train-CV runtime proxy selection."""

    _ensure_file_exists(dataset)
    _ensure_file_exists(stage43_report)
    _ensure_file_exists(stage44_report)
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
        / f"runtime_document_risk_risk_aware_selection_{dataset.stem}.json"
    )
    visualization_output_dir = visualization_dir or output_path.with_suffix("")

    rows = load_candidate_reranker_rows(dataset)
    stage43 = _load_json_object(stage43_report)
    stage44 = _load_json_object(stage44_report)
    gold_answers = _load_gold_answers(
        settings=settings,
        splits=[normalized_train_split, normalized_evaluation_split],
    )
    result = select_risk_aware_runtime_document_risk_proxy(
        stage43_report=stage43,
        stage44_report=stage44,
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name=model,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        train_fold_count=train_fold_count,
        max_answer_candidates=max_answer_candidates,
    )
    visualizations = write_runtime_document_risk_risk_aware_visualizations(
        result=result,
        output_dir=visualization_output_dir,
    )
    result_dict = runtime_document_risk_risk_aware_selection_to_dict(result)
    result_dict["source_paths"] = {
        "dataset": str(dataset),
        "stage43_report": str(stage43_report),
        "stage44_report": str(stage44_report),
        "gold_answer_splits": [normalized_train_split, normalized_evaluation_split],
    }
    result_dict["visualizations"] = {
        "objectives": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations.objectives
        ],
        "guards": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations.guards
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
                "mode_name": f"top{result.max_answer_candidates}_leading_candidate_rewrite",
                "primary_objective_label": result.primary_objective_label,
                "primary_selected_guard_label": result.primary_selected_guard_label,
                "primary_train_cv_metrics": _metrics_summary(
                    result.primary_train_cv_metrics
                ),
                "primary_holdout_metrics": _metrics_summary(
                    result.primary_holdout_metrics
                ),
                "objectives": [
                    {
                        "label": objective.label,
                        "constraint_summary": objective.constraint_summary,
                        "feasible_guard_labels": objective.feasible_guard_labels,
                        "selected_guard_label": objective.selected_guard_label,
                        "selected_train_cv_metrics": _metrics_summary(
                            objective.selected_train_cv_metrics
                        ),
                        "selected_holdout_metrics": _metrics_summary(
                            objective.selected_holdout_metrics
                        ),
                    }
                    for objective in result.objective_evaluations
                ],
                "findings": result.findings,
                "output": str(output_path),
                "visualization_dir": str(visualization_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _metrics_summary(metrics) -> dict:
    return {
        "policy_average_answer_token_f1": metrics.policy_average_answer_token_f1,
        "average_delta_vs_baseline": metrics.average_delta_vs_baseline,
        "replacement_count": metrics.replacement_count,
        "regressed_count": metrics.regressed_count,
        "citation_lost_count": metrics.citation_lost_count,
        "citation_gained_count": metrics.citation_gained_count,
        "gold_citation_delta": metrics.gold_citation_delta,
    }


def _load_json_object(path: Path) -> dict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"JSON file must contain an object: {path}")
    return loaded


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
