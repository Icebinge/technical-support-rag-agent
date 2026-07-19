from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_untouched_rrf_dev_contract_correction import (
    PrimeQAHybridUntouchedRRFDevCorrectionVisualization,
    run_stage163_contract_correction,
    write_stage163_contract_correction_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Audit the Stage163 Top400/Top200 process-contract correction.")


@app.command()
def main(
    original_report: Annotated[
        Path | None,
        typer.Option("--original-report", help="Immutable original Stage163 public report."),
    ] = None,
    stage160_report: Annotated[
        Path | None,
        typer.Option("--stage160-report", help="Stage160 public diagnostic report."),
    ] = None,
    stage160_hashed_diagnostics: Annotated[
        Path | None,
        typer.Option(
            "--stage160-hashed-diagnostics",
            help="Stage160 local hashed diagnostic report without raw question or document data.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage163 contract-correction JSON."),
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
    """Write a correction audit without loading or reevaluating a data split."""

    artifact_dir = ProjectSettings().artifact_dir
    report = run_stage163_contract_correction(
        original_report_path=original_report
        or artifact_dir / "primeqa_hybrid_untouched_rrf_dev_validation_stage163.json",
        stage160_report_path=stage160_report
        or artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160.json",
        stage160_private_report_path=stage160_hashed_diagnostics
        or artifact_dir
        / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160_private.json",
        user_confirmed_correction=user_confirmed_correction,
        confirmation_note=confirmation_note,
    )
    visual_dir = visualization_dir or (
        artifact_dir / "primeqa_hybrid_untouched_rrf_dev_contract_correction_stage163_visuals"
    )
    visualizations = write_stage163_contract_correction_visualizations(
        report=report,
        output_dir=visual_dir,
    )
    report = {**report, "visualizations": _visualization_rows(visualizations)}
    output_path = output or (
        artifact_dir / "primeqa_hybrid_untouched_rrf_dev_contract_correction_stage163.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage163 contract-correction audit: {output_path}")


def _visualization_rows(
    visualizations: list[PrimeQAHybridUntouchedRRFDevCorrectionVisualization],
) -> list[dict[str, str]]:
    return [{"name": artifact.name, "path": artifact.path} for artifact in visualizations]


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "contract_evidence": report["contract_evidence"],
        "correction": report["correction"],
        "metric_integrity": report["metric_integrity"],
        "policy_result": report["policy_result"],
        "guard_checks": report["guard_checks"],
        "public_safe_contract": report["public_safe_contract"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
