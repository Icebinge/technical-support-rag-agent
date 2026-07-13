from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.candidate_score_holdout_changed_case_audit import (
    analyze_candidate_score_holdout_changed_cases,
    candidate_score_holdout_changed_case_audit_to_dict,
    write_candidate_score_holdout_changed_case_visualizations,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

app = typer.Typer(help="Audit Stage 40 candidate-score holdout changed cases.")


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    stage40_report: Annotated[
        Path,
        typer.Option("--stage40-report", help="Stage 40 split-validation JSON report."),
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
        typer.Option("--evaluation-split", help="Split audited for holdout cases."),
    ] = "dev",
    max_answer_candidates: Annotated[
        int,
        typer.Option(
            "--max-answer-candidates",
            help="Top-k size for the leading-candidate rewrite proxy.",
        ),
    ] = 3,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum case examples retained."),
    ] = 50,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage 41 audit JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run Stage 41 candidate-score holdout changed-case audit."""

    _ensure_file_exists(dataset)
    _ensure_file_exists(stage40_report)
    normalized_train_split = _parse_split_name(train_split)
    normalized_evaluation_split = _parse_split_name(evaluation_split)
    _validate_options(
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / f"candidate_score_holdout_changed_cases_{dataset.stem}.json"
    )
    visualization_output_dir = visualization_dir or output_path.with_suffix("")

    rows = load_candidate_reranker_rows(dataset)
    report = _load_json_object(stage40_report)
    gold_answers = _load_gold_answers(
        settings=settings,
        split=normalized_evaluation_split,
    )
    audit = analyze_candidate_score_holdout_changed_cases(
        stage40_report=report,
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name=model,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    visualizations = write_candidate_score_holdout_changed_case_visualizations(
        audit=audit,
        output_dir=visualization_output_dir,
    )
    audit_dict = candidate_score_holdout_changed_case_audit_to_dict(audit)
    audit_dict["source_paths"] = {
        "dataset": str(dataset),
        "stage40_report": str(stage40_report),
        "gold_answer_split": normalized_evaluation_split,
    }
    audit_dict["visualizations"] = [
        {"name": visualization.name, "path": visualization.path}
        for visualization in visualizations
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(
        json.dumps(
            {
                "model_name": audit.model_name,
                "train_split": audit.train_split,
                "evaluation_split": audit.evaluation_split,
                "mode_name": audit.mode_name,
                "metrics": {
                    "question_count": audit.metrics.question_count,
                    "changed_case_count": audit.metrics.changed_case_count,
                    "average_candidate_delta_vs_main": (
                        audit.metrics.average_candidate_delta_vs_main
                    ),
                    "candidate_improved_vs_main_count": (
                        audit.metrics.candidate_improved_vs_main_count
                    ),
                    "candidate_regressed_vs_main_count": (
                        audit.metrics.candidate_regressed_vs_main_count
                    ),
                    "main_delta_vs_baseline": audit.metrics.main_delta_vs_baseline,
                    "candidate_delta_vs_baseline": (
                        audit.metrics.candidate_delta_vs_baseline
                    ),
                    "main_regressed_count": audit.metrics.main_regressed_count,
                    "candidate_regressed_count": audit.metrics.candidate_regressed_count,
                    "main_citation_lost_count": (
                        audit.metrics.main_citation_lost_count
                    ),
                    "candidate_citation_lost_count": (
                        audit.metrics.candidate_citation_lost_count
                    ),
                    "main_gold_citation_delta": audit.metrics.main_gold_citation_delta,
                    "candidate_gold_citation_delta": (
                        audit.metrics.candidate_gold_citation_delta
                    ),
                },
                "residual_regression_cases": [
                    {
                        "question_key": f"{case.split}::{case.question_id}",
                        "route": case.question_route,
                        "delta": case.f1_delta_vs_baseline,
                        "leading_rank": case.leading_candidate_rank,
                        "leading_score": case.leading_candidate_score,
                        "document_transition": (
                            case.baseline_to_candidate_document_transition
                        ),
                    }
                    for case in audit.residual_regression_cases
                ],
                "findings": audit.findings,
                "output": str(output_path),
                "visualization_dir": str(visualization_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_json_object(path: Path) -> dict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"JSON file must contain an object: {path}")
    return loaded


def _load_gold_answers(settings: ProjectSettings, split: str) -> dict[str, str]:
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    questions_path = _resolve_questions_path(training_dir, split)
    _ensure_file_exists(questions_path)
    gold_answers = {}
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
    max_answer_candidates: int,
    sample_limit: int,
) -> None:
    if train_split == evaluation_split:
        raise typer.BadParameter("--train-split and --evaluation-split must differ.")
    if max_answer_candidates <= 0:
        raise typer.BadParameter("--max-answer-candidates must be positive.")
    if sample_limit < 0:
        raise typer.BadParameter("--sample-limit must be non-negative.")


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
