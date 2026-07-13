from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.runtime_document_risk_proxy_search import (
    runtime_document_risk_proxy_search_to_dict,
    search_runtime_document_risk_proxies,
    write_runtime_document_risk_proxy_visualizations,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

app = typer.Typer(help="Search runtime-available document-risk proxy guards.")


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    stage42_report: Annotated[
        Path,
        typer.Option("--stage42-report", help="Stage 42 document-aware JSON report."),
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
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage 43 proxy-search JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run Stage 43 runtime-available document-risk proxy search."""

    _ensure_file_exists(dataset)
    _ensure_file_exists(stage42_report)
    normalized_train_split = _parse_split_name(train_split)
    normalized_evaluation_split = _parse_split_name(evaluation_split)
    _validate_options(
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        max_answer_candidates=max_answer_candidates,
    )

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / f"runtime_document_risk_proxy_search_{dataset.stem}.json"
    )
    visualization_output_dir = visualization_dir or output_path.with_suffix("")

    rows = load_candidate_reranker_rows(dataset)
    report = _load_json_object(stage42_report)
    gold_answers = _load_gold_answers(
        settings=settings,
        split=normalized_evaluation_split,
    )
    result = search_runtime_document_risk_proxies(
        stage42_report=report,
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name=model,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        max_answer_candidates=max_answer_candidates,
    )
    visualizations = write_runtime_document_risk_proxy_visualizations(
        result=result,
        output_dir=visualization_output_dir,
    )
    result_dict = runtime_document_risk_proxy_search_to_dict(result)
    result_dict["source_paths"] = {
        "dataset": str(dataset),
        "stage42_report": str(stage42_report),
        "gold_answer_split": normalized_evaluation_split,
    }
    result_dict["visualizations"] = [
        {"name": visualization.name, "path": visualization.path}
        for visualization in visualizations
    ]

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
                "mode_name": result.mode_name,
                "guards": [
                    {
                        "label": evaluation.label,
                        "feature_scope": evaluation.feature_scope,
                        "policy_average_answer_token_f1": (
                            evaluation.metrics.policy_average_answer_token_f1
                        ),
                        "average_delta_vs_baseline": (
                            evaluation.metrics.average_delta_vs_baseline
                        ),
                        "replacement_count": evaluation.metrics.replacement_count,
                        "regressed_count": evaluation.metrics.regressed_count,
                        "citation_lost_count": (
                            evaluation.metrics.citation_lost_count
                        ),
                        "citation_gained_count": (
                            evaluation.metrics.citation_gained_count
                        ),
                        "gold_citation_delta": (
                            evaluation.metrics.gold_citation_delta
                        ),
                        "changed_count_vs_score60": (
                            evaluation.changed_count_vs_score60
                        ),
                        "fixed_main_regression_count": (
                            evaluation.fixed_main_regression_count
                        ),
                        "introduced_regression_vs_main_count": (
                            evaluation.introduced_regression_vs_main_count
                        ),
                    }
                    for evaluation in result.guard_evaluations
                ],
                "findings": result.findings,
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
) -> None:
    if train_split == evaluation_split:
        raise typer.BadParameter("--train-split and --evaluation-split must differ.")
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
