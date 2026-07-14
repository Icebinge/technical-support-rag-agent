from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.candidate_reranker_cv import DEFAULT_MODEL_NAMES
from ts_rag_agent.application.primeqa_hybrid_candidate_reranker_topk_diagnostic import (
    run_primeqa_hybrid_candidate_reranker_topk_diagnostic,
    write_primeqa_hybrid_candidate_reranker_topk_diagnostic_visualizations,
    write_stage73_report,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help=(
        "Run Stage73 train/dev-only top10 candidate-reranker diagnostic on "
        "PrimeQA hybrid data."
    )
)


@app.command()
def main(
    candidate_dataset: Annotated[
        Path | None,
        typer.Option("--candidate-dataset", help="Stage69 train/dev candidate JSONL."),
    ] = None,
    candidate_summary: Annotated[
        Path | None,
        typer.Option("--candidate-summary", help="Stage69 candidate summary JSON."),
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
        typer.Option("--output", help="Output Stage73 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    fold_count: Annotated[
        int,
        typer.Option("--fold-count", help="Train-only grouped CV fold count."),
    ] = 5,
    models: Annotated[
        str,
        typer.Option("--models", help="Comma-separated model names for train-only CV."),
    ] = ",".join(DEFAULT_MODEL_NAMES),
    policy_models: Annotated[
        str | None,
        typer.Option(
            "--policy-models",
            help=(
                "Comma-separated models used for guarded policy validation. "
                "Defaults to --models."
            ),
        ),
    ] = None,
    max_answer_candidates: Annotated[
        int,
        typer.Option(
            "--max-answer-candidates",
            help="Top-k size for leading-candidate rewrite proxy.",
        ),
    ] = 10,
) -> None:
    """Run Stage73 top10 diagnostic and write the report."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    candidate_dataset_path = candidate_dataset or (
        settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_candidates.jsonl"
    )
    candidate_summary_path = candidate_summary or (
        settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_candidates.summary.json"
    )
    train_split_path = train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    dev_split_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals"
    )

    for path in [
        candidate_dataset_path,
        candidate_summary_path,
        train_split_path,
        dev_split_path,
    ]:
        _ensure_file_exists(path)

    model_names = _parse_models(models)
    policy_model_names = _parse_models(policy_models) if policy_models else model_names
    run = run_primeqa_hybrid_candidate_reranker_topk_diagnostic(
        candidate_dataset_path=candidate_dataset_path,
        candidate_summary_path=candidate_summary_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        fold_count=fold_count,
        model_names=model_names,
        policy_model_names=policy_model_names,
        max_answer_candidates=max_answer_candidates,
    )
    visualizations = (
        write_primeqa_hybrid_candidate_reranker_topk_diagnostic_visualizations(
            run=run,
            output_dir=visualization_output_dir,
        )
    )
    report = write_stage73_report(
        report=run.report,
        visualizations=visualizations,
        output_path=output_path,
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid top10 diagnostic report: {output_path}")


def _parse_models(raw_models: str) -> tuple[str, ...]:
    model_names = tuple(model.strip() for model in raw_models.split(",") if model.strip())
    if not model_names:
        raise typer.BadParameter("--models must not be empty.")
    return model_names


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "split_contract": report["split_contract"],
        "topk_diagnostic": report["topk_diagnostic"],
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


if __name__ == "__main__":
    app()
