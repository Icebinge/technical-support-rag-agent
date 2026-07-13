from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.answer_composition_analysis import (
    analyze_answer_composition_report,
    composition_result_to_dict,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Analyze answer composition strategies from an answer-gap report.")


@app.command()
def main(
    answer_gap_report: Annotated[
        Path,
        typer.Option("--answer-gap-report", help="Input answer-gap JSON report."),
    ],
    f1_gain_margin: Annotated[
        float,
        typer.Option("--f1-gain-margin", help="F1 gain margin used to count improvements."),
    ] = 0.03,
    sample_limit_per_bucket: Annotated[
        int,
        typer.Option("--sample-limit-per-bucket", help="Maximum cases saved per bucket."),
    ] = 20,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON report path."),
    ] = None,
) -> None:
    """Analyze composition policies over the selected evidence candidates."""

    if f1_gain_margin < 0:
        raise typer.BadParameter("--f1-gain-margin must be non-negative.")
    if sample_limit_per_bucket < 0:
        raise typer.BadParameter("--sample-limit-per-bucket must be non-negative.")
    if not answer_gap_report.exists():
        raise typer.BadParameter(f"Missing answer gap report: {answer_gap_report}")

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / f"answer_composition_analysis_{answer_gap_report.stem}.json"
    )

    report = json.loads(answer_gap_report.read_text(encoding="utf-8"))
    result = analyze_answer_composition_report(
        answer_gap_report=report,
        f1_gain_margin=f1_gain_margin,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    result_dict = composition_result_to_dict(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(json.dumps(result_dict["summary"], ensure_ascii=False, indent=2))
    typer.echo(f"Saved answer composition analysis: {output_path}")


if __name__ == "__main__":
    app()
