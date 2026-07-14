from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.external_eval_dataset_discovery import (
    discover_external_eval_datasets,
    write_external_eval_discovery_visualizations,
)

app = typer.Typer(help="Discover external evaluation dataset candidates for Stage 55.")


@app.command()
def main(
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 55 discovery JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write the Stage 55 external evaluation dataset discovery report."""

    report = discover_external_eval_datasets()
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_external_eval_discovery_visualizations(
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
    typer.echo(f"Saved external evaluation dataset discovery report: {output}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": report["decision"],
        "candidates": [
            {
                "label": candidate["label"],
                "status": candidate["status"],
                "fit_score": candidate["scores"]["fit_score"],
                "domain_fit_score": candidate["scores"]["domain_fit_score"],
                "citation_fit_score": candidate["scores"]["citation_fit_score"],
                "adapter_effort_score": candidate["scores"]["adapter_effort_score"],
            }
            for candidate in report["candidates"]
        ],
        "blocked_actions": report["blocked_actions"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
