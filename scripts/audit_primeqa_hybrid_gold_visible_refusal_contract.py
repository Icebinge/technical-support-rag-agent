from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_gold_visible_refusal_contract_correction as correction,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Audit the Stage164 generation-rank contract correction.")


@app.command()
def main(
    original_public_report: Annotated[
        Path | None,
        typer.Option("--original-public-report", help="Immutable original Stage164 report."),
    ] = None,
    original_private_report: Annotated[
        Path | None,
        typer.Option(
            "--original-private-report",
            help="Immutable original Stage164 ignored hashed feature report.",
        ),
    ] = None,
    stage160_hashed_report: Annotated[
        Path | None,
        typer.Option("--stage160-hashed-report", help="Stage160 ignored hashed diagnostics."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage164 contract-correction JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for correction SVGs."),
    ] = None,
    user_confirmed_correction: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-correction/--no-user-confirmed-correction",
            help="Required confirmation that the user selected correction option A.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Correct Stage164 using only immutable hashed artifacts."""

    artifact_dir = ProjectSettings().artifact_dir
    report = correction.run_stage164_contract_correction(
        original_public_report_path=original_public_report
        or artifact_dir / "primeqa_hybrid_gold_visible_refusal_diagnostics_stage164.json",
        original_private_report_path=original_private_report
        or artifact_dir / "primeqa_hybrid_gold_visible_refusal_diagnostics_stage164_private.json",
        stage160_hashed_report_path=stage160_hashed_report
        or artifact_dir
        / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160_private.json",
        user_confirmed_correction=user_confirmed_correction,
        confirmation_note=confirmation_note,
    )
    visual_dir = visualization_dir or (
        artifact_dir / "primeqa_hybrid_gold_visible_refusal_contract_stage164_visuals"
    )
    visualizations = correction.write_stage164_contract_correction_visualizations(
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
        artifact_dir / "primeqa_hybrid_gold_visible_refusal_contract_stage164.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage164 contract-correction audit: {output_path}")
    if report["decision"]["all_correction_guards_passed"] is not True:
        raise typer.Exit(code=1)


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "rank_semantics_correction": report["rank_semantics_correction"],
        "hypothesis_interpretation_correction": report["hypothesis_interpretation_correction"],
        "stable_observed_patterns": report["stable_observed_patterns"],
        "process_correction": report["process_correction"],
        "metric_integrity": report["metric_integrity"],
        "guard_checks": report["guard_checks"],
        "public_safe_contract": report["public_safe_contract"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
