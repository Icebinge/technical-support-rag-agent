from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_transition_correction as correction,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Audit the Stage165 unanswerable transition-label correction.")


@app.command()
def main(
    original_public_report: Annotated[
        Path | None,
        typer.Option("--original-public-report", help="Immutable Stage165 public report."),
    ] = None,
    original_private_report: Annotated[
        Path | None,
        typer.Option("--original-private-report", help="Immutable Stage165 private report."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output transition-correction report."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output correction SVG directory."),
    ] = None,
    user_confirmed_stage_continuation: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-stage-continuation/--no-user-confirmed-stage-continuation",
            help="Required confirmation that the user requested the next stage.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Factual stage-continuation note."),
    ] = "not confirmed",
) -> None:
    """Correct transition labels from immutable artifacts without rerunning Agent work."""

    artifact_dir = ProjectSettings().artifact_dir
    report = correction.run_stage165_transition_correction(
        original_public_report_path=original_public_report
        or artifact_dir / "primeqa_hybrid_train_history_isolation_sharded_stage165.json",
        original_private_report_path=original_private_report
        or artifact_dir / "primeqa_hybrid_train_history_isolation_sharded_stage165_private.json",
        user_confirmed_stage_continuation=user_confirmed_stage_continuation,
        confirmation_note=confirmation_note,
    )
    visual_dir = visualization_dir or (
        artifact_dir
        / "primeqa_hybrid_train_history_isolation_transition_correction_stage165_visuals"
    )
    visualizations = correction.write_stage165_transition_correction_visualizations(
        report=report,
        output_dir=visual_dir,
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_train_history_isolation_transition_correction_stage165.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage165 transition correction: {output_path}")
    if report["decision"]["all_correction_guards_passed"] is not True:
        raise typer.Exit(code=1)


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "correction": report["correction"],
        "metric_integrity": report["metric_integrity"],
        "execution_counts": report["execution_counts"],
        "guard_checks": report["guard_checks"],
        "public_safe_contract": report["public_safe_contract"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
