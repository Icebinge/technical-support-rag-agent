from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_stage51_candidate_distribution_review import (
    review_msqa_stage51_candidate_distribution,
    write_msqa_stage51_candidate_distribution_visualizations,
)

app = typer.Typer(help="Review the MSQA Stage 51 candidate adapter distribution.")


@app.command()
def main(
    adapter_report: Annotated[
        Path,
        typer.Option("--adapter-report", help="Stage 61 adapter dry-run JSON."),
    ],
    candidate_jsonl: Annotated[
        Path,
        typer.Option("--candidate-jsonl", help="Stage 61 candidate JSONL."),
    ],
    stage31_summary: Annotated[
        Path,
        typer.Option("--stage31-summary", help="Stage 31 reranker summary JSON."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 62 distribution report JSON."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write Stage 62 MSQA candidate distribution review artifacts."""

    for path in [adapter_report, candidate_jsonl, stage31_summary]:
        _ensure_file_exists(path)
    report = review_msqa_stage51_candidate_distribution(
        adapter_report_path=adapter_report,
        candidate_jsonl_path=candidate_jsonl,
        stage31_summary_path=stage31_summary,
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_msqa_stage51_candidate_distribution_visualizations(
            report=report,
            output_dir=visualization_dir,
        )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved MSQA Stage 51 candidate distribution review: {output}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage61_adapter_summary": report["stage61_adapter_summary"],
        "stage31_training_candidate_contract": (
            report["stage31_training_candidate_contract"]
        ),
        "stage61_candidate_count_per_query": report[
            "stage61_candidate_distribution"
        ]["candidate_count_per_query"],
        "stage31_candidate_count_per_question": report[
            "stage31_candidate_distribution"
        ]["candidate_count_per_question"],
        "candidate_pool_comparison": report["candidate_pool_comparison"],
        "fairness_checks": report["fairness_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


if __name__ == "__main__":
    app()
