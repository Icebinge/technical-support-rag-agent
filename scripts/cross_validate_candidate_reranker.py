from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_cv import (
    DEFAULT_MODEL_NAMES,
    candidate_reranker_cv_result_to_dict,
    cross_validate_candidate_rerankers,
    write_cv_visualizations,
)
from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Cross-validate baseline candidate reranker models.")


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    fold_count: Annotated[
        int,
        typer.Option("--fold-count", help="Number of deterministic question folds."),
    ] = 5,
    models: Annotated[
        str,
        typer.Option(
            "--models",
            help="Comma-separated model names.",
        ),
    ] = ",".join(DEFAULT_MODEL_NAMES),
    f1_tie_margin: Annotated[
        float,
        typer.Option(
            "--f1-tie-margin",
            help="Absolute F1 margin treated as tie for win/loss counts.",
        ),
    ] = 0.0,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output CV JSON report path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run grouped k-fold CV for baseline candidate rerankers."""

    _ensure_file_exists(dataset)
    model_names = _parse_models(models)
    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir / f"candidate_reranker_cv_{dataset.stem}.json"
    )
    visualization_output_dir = visualization_dir or output_path.with_suffix("")

    rows = load_candidate_reranker_rows(dataset)
    result = cross_validate_candidate_rerankers(
        rows=rows,
        fold_count=fold_count,
        model_names=model_names,
        f1_tie_margin=f1_tie_margin,
    )
    visualizations = write_cv_visualizations(
        result=result,
        output_dir=visualization_output_dir,
    )
    result_dict = candidate_reranker_cv_result_to_dict(result)
    result_dict["source_paths"] = {"dataset": str(dataset)}
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
                "fold_count": result.fold_count,
                "best_model_name": result.best_model_name,
                "models": [
                    {
                        "model_name": model.model_name,
                        "target_name": model.target_name,
                        "baseline_average_token_f1": (
                            model.aggregate_validation.baseline_average_token_f1
                        ),
                        "selected_average_token_f1": (
                            model.aggregate_validation.selected_average_token_f1
                        ),
                        "average_delta_vs_top_candidate": (
                            model.aggregate_validation.average_delta_vs_top_candidate
                        ),
                        "oracle_gap_closed_rate": (
                            model.aggregate_validation.oracle_gap_closed_rate
                        ),
                        "selected_best_candidate_rate": (
                            model.aggregate_validation.selected_best_candidate_rate
                        ),
                    }
                    for model in result.models
                ],
                "output": str(output_path),
                "visualization_dir": str(visualization_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_models(raw_models: str) -> list[str]:
    model_names = [model.strip() for model in raw_models.split(",") if model.strip()]
    if not model_names:
        raise typer.BadParameter("--models must not be empty.")
    return model_names


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
