from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.external_eval_dataset_rediscovery import (
    rediscover_external_eval_datasets,
    write_external_eval_rediscovery_visualizations,
)

app = typer.Typer(help="Rediscover external evaluation dataset candidates for Stage66.")


@app.command()
def main(
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage66 rediscovery JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write the Stage66 external dataset rediscovery report."""

    report = rediscover_external_eval_datasets()
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_external_eval_rediscovery_visualizations(
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
    typer.echo(f"Saved external evaluation dataset rediscovery report: {output}")


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "decision": report["decision"],
        "candidates": [
            {
                "label": candidate["label"],
                "status": candidate["status"],
                "fit_score": candidate["scores"]["fit_score"],
                "domain_fit_score": candidate["scores"]["domain_fit_score"],
                "citation_fit_score": candidate["scores"]["citation_fit_score"],
                "license_fit_score": candidate["scores"]["license_fit_score"],
                "adapter_effort_score": candidate["scores"]["adapter_effort_score"],
            }
            for candidate in report["candidates"]
        ],
        "blocked_actions": report["blocked_actions"],
        "visualizations": report["visualizations"],
    }


MappingForSummary = dict[str, Any]


if __name__ == "__main__":
    app()
