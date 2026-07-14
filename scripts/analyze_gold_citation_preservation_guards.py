from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.gold_citation_preservation_guard_analysis import (
    PreservationGuardScenario,
    analyze_gold_citation_preservation_guards,
    write_preservation_guard_visualizations,
)

app = typer.Typer(help="Analyze runtime-only gold-citation preservation guards.")


@app.command()
def main(
    dev_baseline_report: Annotated[
        Path,
        typer.Option("--dev-baseline-report", help="Dev top-k baseline report."),
    ],
    dev_candidate_report: Annotated[
        Path,
        typer.Option("--dev-candidate-report", help="Dev candidate report."),
    ],
    train_baseline_report: Annotated[
        Path,
        typer.Option("--train-baseline-report", help="Train top-k baseline report."),
    ],
    train_candidate_report: Annotated[
        Path,
        typer.Option("--train-candidate-report", help="Train candidate report."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output guard analysis JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write dev/train preservation-guard analysis."""

    paths = [
        dev_baseline_report,
        dev_candidate_report,
        train_baseline_report,
        train_candidate_report,
    ]
    for path in paths:
        _ensure_file_exists(path)

    analysis = analyze_gold_citation_preservation_guards(
        scenarios=[
            PreservationGuardScenario(
                label="dev",
                baseline_report=_load_json(dev_baseline_report),
                candidate_report=_load_json(dev_candidate_report),
            ),
            PreservationGuardScenario(
                label="train",
                baseline_report=_load_json(train_baseline_report),
                candidate_report=_load_json(train_candidate_report),
            ),
        ]
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_preservation_guard_visualizations(
            analysis=analysis,
            output_dir=visualization_dir,
        )
    report = {
        **analysis,
        "paths": {
            "dev_baseline_report": str(dev_baseline_report),
            "dev_candidate_report": str(dev_candidate_report),
            "train_baseline_report": str(train_baseline_report),
            "train_candidate_report": str(train_candidate_report),
        },
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved gold-citation preservation guard analysis: {output}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenarios": [
            {
                "label": scenario["label"],
                "candidate_delta": scenario["candidate_metric_deltas_vs_baseline"],
                "guard_results": [
                    {
                        "guard": result["guard_label"],
                        "blocked": result["blocked_changed_answer_count"],
                        "gold_delta": result["metric_deltas_vs_baseline"][
                            "gold_cited_count"
                        ],
                        "f1_delta": result["metric_deltas_vs_baseline"][
                            "average_token_f1"
                        ],
                        "changed": result["changed_answer_outcomes_vs_baseline"][
                            "changed_verified_answers"
                        ],
                    }
                    for result in scenario["guard_results"]
                ],
            }
            for scenario in report["scenarios"]
        ],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
