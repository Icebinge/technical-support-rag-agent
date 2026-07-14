from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.evaluation_strategy_review import (
    review_evaluation_strategy,
    write_evaluation_strategy_visualizations,
)

app = typer.Typer(help="Review evaluation strategy after held-out leakage blocking.")


@app.command()
def main(
    readiness_review: Annotated[
        Path,
        typer.Option("--readiness-review", help="Stage 52 readiness review JSON."),
    ],
    leakage_report: Annotated[
        Path,
        typer.Option("--leakage-report", help="Stage 53 held-out leakage report JSON."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 54 evaluation strategy JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write the Stage 54 evaluation strategy review."""

    _ensure_file_exists(readiness_review)
    _ensure_file_exists(leakage_report)

    review = review_evaluation_strategy(
        readiness_review=_load_json(readiness_review),
        leakage_report=_load_json(leakage_report),
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_evaluation_strategy_visualizations(
            review=review,
            output_dir=visualization_dir,
        )
    report = {
        **review,
        "paths": {
            "readiness_review": str(readiness_review),
            "leakage_report": str(leakage_report),
        },
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved evaluation strategy review: {output}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_facts": report["current_facts"],
        "rejected_paths": report["rejected_paths"],
        "strategy_options": [
            {
                "label": option["label"],
                "status": option["status"],
                "validity_score": option["validity_score"],
                "effort_score": option["effort_score"],
                "can_support_defaultization": option["can_support_defaultization"],
            }
            for option in report["strategy_options"]
        ],
        "decision_required": report["decision_required"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
