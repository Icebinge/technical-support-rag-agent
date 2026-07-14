from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_stage51_protocol_design import (
    design_msqa_stage51_protocol,
    write_msqa_stage51_protocol_visualizations,
)

app = typer.Typer(help="Design the MSQA Stage 51 source/citation protocol.")


@app.command()
def main(
    schema_probe: Annotated[
        Path,
        typer.Option("--schema-probe", help="Stage 56 MSQA schema probe JSON."),
    ],
    evaluation_split: Annotated[
        Path,
        typer.Option("--evaluation-split", help="Stage 57 MSQA split JSON."),
    ],
    compatibility_review: Annotated[
        Path,
        typer.Option("--compatibility-review", help="Stage 59 compatibility JSON."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 60 protocol design JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write the Stage 60 MSQA Stage 51 protocol design report."""

    for path in [schema_probe, evaluation_split, compatibility_review]:
        _ensure_file_exists(path)
    report = design_msqa_stage51_protocol(
        schema_probe_report_path=schema_probe,
        evaluation_split_report_path=evaluation_split,
        compatibility_review_path=compatibility_review,
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_msqa_stage51_protocol_visualizations(
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
    typer.echo(f"Saved MSQA Stage 51 protocol design: {output}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_constraints": report["current_constraints"],
        "recommended_protocol": report["recommended_protocol"],
        "source_citation_identity_options": [
            _option_summary(option)
            for option in report["source_citation_identity_options"]
        ],
        "candidate_construction_options": [
            _option_summary(option)
            for option in report["candidate_construction_options"]
        ],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


def _option_summary(option: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": option["label"],
        "status": option["status"],
        "coverage_percent": option["coverage_percent"],
        "total_score": option["total_score"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


if __name__ == "__main__":
    app()
