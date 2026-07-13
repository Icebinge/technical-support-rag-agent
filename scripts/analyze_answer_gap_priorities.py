from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.answer_gap_priority_analysis import (
    analyze_answer_gap_priorities,
    answer_gap_priority_analysis_to_dict,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Prioritize answer-gap routes and buckets across reports.")


@app.command()
def main(
    answer_gap_reports: Annotated[
        str,
        typer.Option(
            "--answer-gap-reports",
            help="Comma-separated input answer-gap JSON reports.",
        ),
    ],
    min_cases: Annotated[
        int,
        typer.Option("--min-cases", help="Minimum cases required for a priority group."),
    ] = 3,
    sample_limit_per_group: Annotated[
        int,
        typer.Option(
            "--sample-limit-per-group",
            help="Representative cases saved for each priority group.",
        ),
    ] = 5,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON report path."),
    ] = None,
) -> None:
    """Analyze answer-gap priorities across one or more reports."""

    report_paths = _parse_report_paths(answer_gap_reports)
    _validate_options(report_paths=report_paths, min_cases=min_cases)

    settings = ProjectSettings()
    output_path = output or (settings.artifact_dir / "answer_gap_priority_analysis.json")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    analysis = analyze_answer_gap_priorities(
        answer_gap_reports=reports,
        min_cases=min_cases,
        sample_limit_per_group=sample_limit_per_group,
    )
    result_dict = answer_gap_priority_analysis_to_dict(analysis)
    result_dict["source_reports"] = [str(path) for path in report_paths]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(
        json.dumps(
            {
                "total_cases": result_dict["total_cases"],
                "top_priority_summary": result_dict["top_priority_summary"],
                "top_route_priorities": result_dict["route_priorities"][:5],
                "top_route_bucket_priorities": result_dict["route_bucket_priorities"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    typer.echo(f"Saved answer-gap priority analysis: {output_path}")


def _parse_report_paths(raw_reports: str) -> list[Path]:
    paths = [
        Path(raw_path.strip())
        for raw_path in raw_reports.split(",")
        if raw_path.strip()
    ]
    if not paths:
        raise typer.BadParameter("--answer-gap-reports must not be empty.")
    return paths


def _validate_options(report_paths: list[Path], min_cases: int) -> None:
    if min_cases <= 0:
        raise typer.BadParameter("--min-cases must be positive.")
    for path in report_paths:
        if not path.exists():
            raise typer.BadParameter(f"Missing answer-gap report: {path}")


if __name__ == "__main__":
    app()
