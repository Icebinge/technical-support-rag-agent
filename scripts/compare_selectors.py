from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.selector_comparison_analysis import (
    compare_selector_reports,
    comparison_result_to_dict,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Compare two selector answer-gap reports question by question.")


@app.command()
def main(
    baseline_report: Annotated[
        Path,
        typer.Option("--baseline-report", help="Answer-gap JSON for the baseline selector."),
    ],
    challenger_report: Annotated[
        Path,
        typer.Option("--challenger-report", help="Answer-gap JSON for the challenger selector."),
    ],
    baseline_label: Annotated[
        str,
        typer.Option("--baseline-label", help="Human-readable baseline selector name."),
    ] = "baseline",
    challenger_label: Annotated[
        str,
        typer.Option("--challenger-label", help="Human-readable challenger selector name."),
    ] = "challenger",
    f1_win_margin: Annotated[
        float,
        typer.Option("--f1-win-margin", help="Minimum F1 delta required for a win."),
    ] = 0.03,
    sample_limit_per_bucket: Annotated[
        int,
        typer.Option("--sample-limit-per-bucket", help="Saved examples per win bucket."),
    ] = 20,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional comparison JSON path."),
    ] = None,
) -> None:
    """Compare selector reports and save winner/type breakdowns."""

    _validate_options(
        baseline_report=baseline_report,
        challenger_report=challenger_report,
        f1_win_margin=f1_win_margin,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / f"selector_comparison_{_slug(baseline_label)}_vs_{_slug(challenger_label)}.json"
    )

    baseline = json.loads(baseline_report.read_text(encoding="utf-8"))
    challenger = json.loads(challenger_report.read_text(encoding="utf-8"))
    result = compare_selector_reports(
        baseline_report=baseline,
        challenger_report=challenger,
        baseline_label=baseline_label,
        challenger_label=challenger_label,
        f1_win_margin=f1_win_margin,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    report = {
        "baseline_report": str(baseline_report),
        "challenger_report": str(challenger_report),
        "f1_win_margin": f1_win_margin,
        **comparison_result_to_dict(result),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    typer.echo(f"Saved selector comparison report: {output_path}")


def _validate_options(
    baseline_report: Path,
    challenger_report: Path,
    f1_win_margin: float,
    sample_limit_per_bucket: int,
) -> None:
    if not baseline_report.exists():
        raise typer.BadParameter(f"Missing baseline report: {baseline_report}")
    if not challenger_report.exists():
        raise typer.BadParameter(f"Missing challenger report: {challenger_report}")
    if f1_win_margin < 0:
        raise typer.BadParameter("--f1-win-margin must be non-negative.")
    if sample_limit_per_bucket < 0:
        raise typer.BadParameter("--sample-limit-per-bucket must be non-negative.")


def _slug(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


if __name__ == "__main__":
    app()
