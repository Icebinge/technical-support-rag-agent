from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_stage51_compatibility_review import (
    review_msqa_stage51_compatibility,
    write_msqa_stage51_compatibility_visualizations,
)

app = typer.Typer(help="Review MSQA Stage 51 compatibility after Stage 58.")


@app.command()
def main(
    stage58_report: Annotated[
        Path,
        typer.Option("--stage58-report", help="Stage 58 MSQA top-k baseline report."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 59 compatibility report path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write the Stage 59 MSQA Stage 51 compatibility report."""

    _ensure_file_exists(stage58_report)
    report = review_msqa_stage51_compatibility(stage58_report)
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_msqa_stage51_compatibility_visualizations(
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
    typer.echo(f"Saved MSQA Stage 51 compatibility report: {output}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage58_baseline_summary": report["stage58_baseline_summary"],
        "failure_mode_review": {
            "primary_failure_counts": report["failure_mode_review"][
                "primary_failure_counts"
            ],
            "primary_failure_rates": report["failure_mode_review"][
                "primary_failure_rates"
            ],
            "primary_vs_diagnostic_gap": report["failure_mode_review"][
                "primary_vs_diagnostic_gap"
            ],
        },
        "compatibility_gate": report["compatibility_gate"]["summary"],
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
