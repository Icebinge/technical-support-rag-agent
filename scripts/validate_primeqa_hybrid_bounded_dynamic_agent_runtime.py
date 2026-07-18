from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime_validation import (
    validate_primeqa_hybrid_bounded_dynamic_agent_runtime,
    write_stage157_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Validate the Stage157 bounded dynamic Agent runtime.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Stage157 public validation JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Stage157 SVG output directory."),
    ] = None,
) -> None:
    settings = ProjectSettings()
    artifact_dir = settings.artifact_dir.resolve()
    report = validate_primeqa_hybrid_bounded_dynamic_agent_runtime(
        stage156_protocol_path=(
            artifact_dir / "primeqa_hybrid_bounded_agent_state_protocol_stage156.json"
        ),
        model_snapshot_path=model_snapshot,
        stage128_protocol_path=(
            artifact_dir / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json"
        ),
        stage125_protocol_path=(
            artifact_dir
            / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json"
        ),
        stage80_report_path=(
            artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json"
        ),
        documents_path=(
            settings.primeqa_raw_dir.resolve()
            / "TechQA"
            / "training_and_dev"
            / "training_dev_technotes.sections.json"
        ),
    )
    visualizations = write_stage157_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_runtime_stage157_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_runtime_stage157.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage157 bounded dynamic Agent validation: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "environment": report.get("environment"),
        "synthetic_runtime_cases": report.get("synthetic_runtime_cases"),
        "real_non_test_runtime_probe": report.get("real_non_test_runtime_probe"),
        "model_runtime": report.get("model_runtime"),
        "guard_checks": report.get("guard_checks"),
        "public_safe_contract": report.get("public_safe_contract"),
        "decision": report.get("decision"),
        "timing_seconds": report.get("timing_seconds"),
        "visualizations": report.get("visualizations"),
    }


if __name__ == "__main__":
    app()
