from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_joint_risk_winner_protocol as protocol
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage 198 joint risk-signal and winner-rule protocol.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    report = protocol.freeze_joint_risk_winner_protocol(
        stage197_report_path=artifacts / "primeqa_hybrid_surviving_unsafe_winner_stage197.json",
        user_confirmed=True,
        confirmation_note=(
            "User selected route A: jointly cross risk-signal families and risk-aware "
            "winner rules while retaining exact Stage196 controls."
        ),
    )
    visualizations = protocol.write_stage198_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifacts / "primeqa_hybrid_joint_risk_winner_protocol_stage198_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or artifacts / "primeqa_hybrid_joint_risk_winner_protocol_stage198.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage 198 protocol: {output_path}")
    if not report["decision"]["protocol_valid"]:
        raise typer.Exit(code=1)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "evidence_summary": report["evidence_summary"],
        "source_trajectory_contract": frozen["source_trajectory_contract"],
        "risk_signal_factor": frozen["risk_signal_factor"],
        "winner_rule_factor": frozen["winner_rule_factor"],
        "factorial_ablation": frozen["factorial_ablation"],
        "cross_validation": frozen["cross_validation"],
        "resource_contract": frozen["resource_contract"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
