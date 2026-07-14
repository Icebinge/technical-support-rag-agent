from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.candidate_reranker_cv import DEFAULT_MODEL_NAMES
from ts_rag_agent.application.primeqa_hybrid_candidate_reranker_changed_case_review import (
    review_primeqa_hybrid_candidate_reranker_changed_cases,
    write_primeqa_hybrid_candidate_reranker_changed_case_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage72 PrimeQA hybrid candidate-reranker changed-case review."
)


@app.command()
def main(
    stage71_report: Annotated[
        Path | None,
        typer.Option("--stage71-report", help="Stage71 candidate-reranker report."),
    ] = None,
    candidate_dataset: Annotated[
        Path | None,
        typer.Option("--candidate-dataset", help="Stage69 train/dev candidate JSONL."),
    ] = None,
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train JSONL path."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Frozen Stage68 dev JSONL path."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage72 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    models: Annotated[
        str,
        typer.Option("--models", help="Comma-separated model names to review."),
    ] = ",".join(DEFAULT_MODEL_NAMES),
    max_answer_candidates: Annotated[
        int,
        typer.Option(
            "--max-answer-candidates",
            help="Top-k size for leading-candidate rewrite proxy.",
        ),
    ] = 3,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum public-safe case samples retained."),
    ] = 20,
) -> None:
    """Run Stage72 public-safe changed-case review and write the report."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    stage71_report_path = stage71_report or (
        settings.artifact_dir
        / "primeqa_hybrid_candidate_reranker_development_stage71.json"
    )
    candidate_dataset_path = candidate_dataset or (
        settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_candidates.jsonl"
    )
    train_split_path = train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    dev_split_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_candidate_reranker_changed_case_review_stage72.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals"
    )

    for path in [
        stage71_report_path,
        candidate_dataset_path,
        train_split_path,
        dev_split_path,
    ]:
        _ensure_file_exists(path)

    report = review_primeqa_hybrid_candidate_reranker_changed_cases(
        stage71_report_path=stage71_report_path,
        candidate_dataset_path=candidate_dataset_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        model_names=_parse_models(models),
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    visualizations = write_primeqa_hybrid_candidate_reranker_changed_case_visualizations(
        report=report,
        output_dir=visualization_output_dir,
    )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path}
            for artifact in visualizations
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid changed-case review report: {output_path}")


def _parse_models(raw_models: str) -> tuple[str, ...]:
    model_names = tuple(model.strip() for model in raw_models.split(",") if model.strip())
    if not model_names:
        raise typer.BadParameter("--models must not be empty.")
    return model_names


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "split_contract": report["split_contract"],
        "model_summaries": [
            {
                "model_name": review["model_name"],
                "policy_reviews": [
                    {
                        "policy_label": policy["policy_label"],
                        "delta": policy["metrics"]["average_delta_vs_baseline"],
                        "regressions": policy["metrics"]["regressed_count"],
                        "changed_cases": policy["changed_vs_baseline_summary"][
                            "changed_case_count"
                        ],
                        "gold_citation_delta": policy["metrics"]["gold_citation_delta"],
                    }
                    for policy in review["policy_reviews"]
                ],
                "candidate_score_vs_main": (
                    review["candidate_score_gte_60_vs_stage36_main"]["summary"]
                ),
                "findings": review["findings"],
            }
            for review in report["model_reviews"]
        ],
        "cross_model_summary": report["cross_model_summary"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


MappingForSummary = dict[str, Any]


if __name__ == "__main__":
    app()
