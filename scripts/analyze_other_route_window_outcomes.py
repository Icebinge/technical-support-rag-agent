from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.other_route_window_outcome_analysis import (
    analyze_other_route_window_outcomes,
    other_route_window_outcome_analysis_to_dict,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Analyze answer-window outcomes for other-route selector comparisons."
)


@app.command()
def main(
    baseline_reports: Annotated[
        str,
        typer.Option(
            "--baseline-reports",
            help="Comma-separated baseline answer-gap JSON reports.",
        ),
    ],
    challenger_reports: Annotated[
        str,
        typer.Option(
            "--challenger-reports",
            help="Comma-separated challenger answer-gap JSON reports.",
        ),
    ],
    source_labels: Annotated[
        str,
        typer.Option(
            "--source-labels",
            help="Comma-separated labels matching the report order.",
        ),
    ],
    min_cases: Annotated[
        int,
        typer.Option("--min-cases", help="Minimum cases required for a subtype summary."),
    ] = 3,
    f1_win_margin: Annotated[
        float,
        typer.Option("--f1-win-margin", help="Minimum F1 delta required for a win."),
    ] = 0.03,
    sample_limit_per_subtype: Annotated[
        int,
        typer.Option(
            "--sample-limit-per-subtype",
            help="Representative cases saved for each subtype.",
        ),
    ] = 5,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON report path."),
    ] = None,
) -> None:
    """Analyze other-route answer-window wins and losses by subtype."""

    baseline_paths = _parse_report_paths(baseline_reports)
    challenger_paths = _parse_report_paths(challenger_reports)
    labels = _parse_source_labels(source_labels)
    _validate_options(
        baseline_paths=baseline_paths,
        challenger_paths=challenger_paths,
        source_labels=labels,
        min_cases=min_cases,
        f1_win_margin=f1_win_margin,
        sample_limit_per_subtype=sample_limit_per_subtype,
    )

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir / "other_route_window_outcome_analysis.json"
    )
    baseline_data = [json.loads(path.read_text(encoding="utf-8")) for path in baseline_paths]
    challenger_data = [
        json.loads(path.read_text(encoding="utf-8")) for path in challenger_paths
    ]

    analysis = analyze_other_route_window_outcomes(
        baseline_reports=baseline_data,
        challenger_reports=challenger_data,
        source_labels=labels,
        min_cases=min_cases,
        f1_win_margin=f1_win_margin,
        sample_limit_per_subtype=sample_limit_per_subtype,
    )
    report = other_route_window_outcome_analysis_to_dict(analysis)
    report["source_reports"] = [
        {
            "source_label": label,
            "baseline_report": str(baseline_path),
            "challenger_report": str(challenger_path),
        }
        for label, baseline_path, challenger_path in zip(
            labels,
            baseline_paths,
            challenger_paths,
            strict=True,
        )
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(
        json.dumps(
            {
                "total_cases": report["total_cases"],
                "stable_answer_window_subtypes": report[
                    "stable_answer_window_subtypes"
                ],
                "mixed_subtypes": report["mixed_subtypes"],
                "top_summary": report["top_summary"],
                "top_overall_subtypes": report["overall_subtype_summaries"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    typer.echo(f"Saved other-route window outcome analysis: {output_path}")


def _parse_report_paths(raw_reports: str) -> list[Path]:
    paths = [
        Path(raw_path.strip())
        for raw_path in raw_reports.split(",")
        if raw_path.strip()
    ]
    if not paths:
        raise typer.BadParameter("report list must not be empty.")
    return paths


def _parse_source_labels(raw_labels: str) -> list[str]:
    labels = [label.strip() for label in raw_labels.split(",") if label.strip()]
    if not labels:
        raise typer.BadParameter("--source-labels must not be empty.")
    return labels


def _validate_options(
    baseline_paths: list[Path],
    challenger_paths: list[Path],
    source_labels: list[str],
    min_cases: int,
    f1_win_margin: float,
    sample_limit_per_subtype: int,
) -> None:
    if len(baseline_paths) != len(challenger_paths):
        raise typer.BadParameter(
            "--baseline-reports and --challenger-reports must have the same length."
        )
    if len(baseline_paths) != len(source_labels):
        raise typer.BadParameter("--source-labels length must match report lists.")
    if len(set(source_labels)) != len(source_labels):
        raise typer.BadParameter("--source-labels values must be unique.")
    if min_cases <= 0:
        raise typer.BadParameter("--min-cases must be positive.")
    if f1_win_margin < 0:
        raise typer.BadParameter("--f1-win-margin must be non-negative.")
    if sample_limit_per_subtype < 0:
        raise typer.BadParameter("--sample-limit-per-subtype must be non-negative.")

    for path in [*baseline_paths, *challenger_paths]:
        if not path.exists():
            raise typer.BadParameter(f"Missing report: {path}")


if __name__ == "__main__":
    app()
