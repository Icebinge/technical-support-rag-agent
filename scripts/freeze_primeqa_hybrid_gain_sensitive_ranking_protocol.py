from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_gain_sensitive_ranking_protocol as protocol
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage 187 gain-sensitive ranking protocol.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    report = protocol.freeze_gain_sensitive_ranking_protocol(
        stage181_report_path=artifacts / "primeqa_hybrid_composition_action_audit_stage181.json",
        stage182_report_path=artifacts / "primeqa_hybrid_composition_dual_target_stage182.json",
        stage183_report_path=artifacts / "primeqa_hybrid_composition_f1_risk_stage183.json",
        stage184_report_path=artifacts
        / "primeqa_hybrid_composition_f1_representation_stage184.json",
        stage185_report_path=artifacts
        / "primeqa_hybrid_joint_constraint_ranking_protocol_stage185.json",
        stage186_report_path=artifacts / "primeqa_hybrid_joint_constraint_ranking_stage186.json",
        user_confirmed=True,
        confirmation_note="User approved proceeding to the next recommended stage.",
    )
    visualizations = protocol.write_stage187_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifacts / "primeqa_hybrid_gain_sensitive_ranking_protocol_stage187_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = (
        output or artifacts / "primeqa_hybrid_gain_sensitive_ranking_protocol_stage187.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage 187 protocol: {output_path}")
    if not report["decision"]["protocol_valid"]:
        raise typer.Exit(code=1)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "evidence_summary": report["evidence_summary"],
        "candidate_grid": report["frozen_protocol"]["candidate_grid"],
        "cross_validation": report["frozen_protocol"]["cross_validation"],
        "inner_selection": report["frozen_protocol"]["inner_selection"],
        "advancement_gates": report["frozen_protocol"]["advancement_gates"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
