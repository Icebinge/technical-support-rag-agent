from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.candidate_reranker_error_analysis import (
    analyze_candidate_reranker_errors,
    candidate_reranker_error_analysis_to_dict,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Analyze candidate-reranker CV improvements and regressions.")


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
    fold_count: Annotated[
        int,
        typer.Option("--fold-count", help="Number of deterministic question folds."),
    ] = 5,
    f1_tie_margin: Annotated[
        float,
        typer.Option(
            "--f1-tie-margin",
            help="Absolute F1 margin treated as tie for outcome counts.",
        ),
    ] = 0.0,
    deep_rank_min: Annotated[
        int,
        typer.Option("--deep-rank-min", help="Selected rank threshold for deep-rank cases."),
    ] = 6,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum examples retained per sample bucket."),
    ] = 10,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output error-analysis JSON path."),
    ] = None,
) -> None:
    """Run offline grouped-CV error analysis for one candidate reranker model."""

    _ensure_file_exists(dataset)
    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir / f"candidate_reranker_error_analysis_{dataset.stem}.json"
    )

    rows = load_candidate_reranker_rows(dataset)
    result = analyze_candidate_reranker_errors(
        rows=rows,
        model_name=model,
        fold_count=fold_count,
        f1_tie_margin=f1_tie_margin,
        deep_rank_min=deep_rank_min,
        sample_limit=sample_limit,
    )
    result_dict = candidate_reranker_error_analysis_to_dict(result)
    result_dict["source_paths"] = {"dataset": str(dataset)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(
        json.dumps(
            {
                "model_name": result.model_name,
                "fold_count": result.fold_count,
                "summary": {
                    "question_count": result.summary.question_count,
                    "improved_count": result.summary.improved_count,
                    "regressed_count": result.summary.regressed_count,
                    "tied_count": result.summary.tied_count,
                    "average_delta_vs_top_candidate": (
                        result.summary.average_delta_vs_top_candidate
                    ),
                    "selected_missed_gold_document_count": (
                        result.summary.selected_missed_gold_document_count
                    ),
                    "selected_deep_rank_count": result.summary.selected_deep_rank_count,
                },
                "top_regression_routes": [
                    {
                        "route": route.segment_name,
                        "question_count": route.question_count,
                        "regressed_count": route.regressed_count,
                        "regressed_rate": route.regressed_rate,
                        "average_delta_vs_top_candidate": (
                            route.average_delta_vs_top_candidate
                        ),
                    }
                    for route in result.route_summaries[:5]
                ],
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
