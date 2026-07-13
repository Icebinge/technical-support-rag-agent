from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.verified_rag_report_comparison import (
    compare_verified_rag_reports,
    write_verified_rag_comparison_visualizations,
)

app = typer.Typer(help="Compare two verified RAG JSON reports.")


@app.command()
def main(
    baseline_report: Annotated[
        Path,
        typer.Option("--baseline-report", help="Baseline verified RAG report."),
    ],
    candidate_report: Annotated[
        Path,
        typer.Option("--candidate-report", help="Candidate verified RAG report."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output comparison JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write metric, citation, changed-answer, and visualization comparisons."""

    _ensure_file_exists(baseline_report)
    _ensure_file_exists(candidate_report)
    comparison = compare_verified_rag_reports(
        baseline_report=_load_json(baseline_report),
        candidate_report=_load_json(candidate_report),
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_verified_rag_comparison_visualizations(
            comparison=comparison,
            output_dir=visualization_dir,
        )
    report = {
        **comparison,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved verified RAG comparison: {output}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_policy": report["baseline_report"]["composition_policy"],
        "candidate_policy": report["candidate_report"]["composition_policy"],
        "sample_complete": report["sample_completeness"]["complete"],
        "metric_deltas": report["metric_deltas"],
        "verified_gold_citation_delta": report["exact_gold_citations"]["deltas"][
            "verified_gold_cited_count"
        ],
        "changed_verified_answers": report["changed_answers"]["verified"]["all_count"],
        "verified_answerable_f1_outcomes": {
            "improved": report["verified_answerable_f1_outcomes"]["improved_count"],
            "regressed": report["verified_answerable_f1_outcomes"]["regressed_count"],
            "changed_tied": report["verified_answerable_f1_outcomes"][
                "changed_tied_count"
            ],
        },
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
