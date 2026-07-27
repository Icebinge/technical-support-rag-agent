from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_top1_joint_objective_protocol as protocol
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage 202 grouped Top-1 joint-objective protocol.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    report = protocol.freeze_top1_joint_objective_protocol(
        stage201_report_path=artifacts
        / "primeqa_hybrid_joint_risk_winner_failure_attribution_stage201.json",
        stage199_report_path=artifacts / "primeqa_hybrid_joint_risk_winner_stage199.json",
        user_confirmed=True,
        confirmation_note=(
            "User selected route A: a grouped Top-1 objective jointly optimizing strict "
            "capture, unsafe avoidance, and changed-answer precision."
        ),
    )
    visualizations = protocol.write_stage202_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifacts / "primeqa_hybrid_top1_joint_objective_protocol_stage202_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or artifacts / "primeqa_hybrid_top1_joint_objective_protocol_stage202.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage 202 protocol: {output_path}")
    if not report["decision"]["protocol_valid"]:
        raise typer.Exit(code=1)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "evidence_summary": report["evidence_summary"],
        "source_trajectory_contract": frozen["source_trajectory_contract"],
        "candidate_pool_contract": frozen["candidate_pool_contract"],
        "top1_joint_objective": frozen["top1_joint_objective"],
        "objective_factorial": frozen["objective_factorial"],
        "cross_validation": frozen["cross_validation"],
        "resource_contract": frozen["resource_contract"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
