from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.candidate_reranker_cv import DEFAULT_MODEL_NAMES
from ts_rag_agent.application.primeqa_hybrid_candidate_reranker_development import (
    run_primeqa_hybrid_candidate_reranker_development,
    write_primeqa_hybrid_candidate_reranker_development_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage71 train/dev candidate-reranker development on PrimeQA hybrid data."
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
        typer.Option("--output", help="Output Stage71 report JSON path."),
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
    ] = 3,
) -> None:
    """Run Stage71 train/dev candidate-reranker development and write the report."""

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
        / "primeqa_hybrid_candidate_reranker_development_stage71.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_candidate_reranker_development_stage71_visuals"
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
    run = run_primeqa_hybrid_candidate_reranker_development(
        candidate_dataset_path=candidate_dataset_path,
        candidate_summary_path=candidate_summary_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        fold_count=fold_count,
        model_names=model_names,
        policy_model_names=policy_model_names,
        max_answer_candidates=max_answer_candidates,
    )
    visualizations = write_primeqa_hybrid_candidate_reranker_development_visualizations(
        run=run,
        output_dir=visualization_output_dir,
    )
    report = {
        **run.report,
        "visualizations": [
            {
                "group": artifact.group,
                "name": artifact.name,
                "path": artifact.path,
            }
            for artifact in visualizations
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid candidate-reranker development report: {output_path}")


def _parse_models(raw_models: str) -> tuple[str, ...]:
    model_names = tuple(model.strip() for model in raw_models.split(",") if model.strip())
    if not model_names:
        raise typer.BadParameter("--models must not be empty.")
    return model_names


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    train_cv = report["train_only_model_cv"]
    split_validations = report["train_to_dev_policy_validations"]
    return {
        "stage": report["stage"],
        "split_contract": report["split_contract"],
        "train_only_cv": {
            "fold_count": train_cv["fold_count"],
            "best_model_name": train_cv["best_model_name"],
            "models": [
                {
                    "model_name": model["model_name"],
                    "baseline_average_token_f1": (
                        model["aggregate_validation"]["baseline_average_token_f1"]
                    ),
                    "selected_average_token_f1": (
                        model["aggregate_validation"]["selected_average_token_f1"]
                    ),
                    "average_delta_vs_top_candidate": (
                        model["aggregate_validation"]["average_delta_vs_top_candidate"]
                    ),
                    "regressed_count": model["aggregate_validation"]["f1_regressed_count"],
                }
                for model in train_cv["models"]
            ],
        },
        "train_to_dev_policy_validations": [
            {
                "model_name": split_validation["model_name"],
                "train_question_count": split_validation["train_question_count"],
                "evaluation_question_count": split_validation["evaluation_question_count"],
                "findings": split_validation["findings"],
            }
            for split_validation in split_validations
        ],
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
