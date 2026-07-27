from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_joint_risk_winner_failure_attribution_protocol as protocol,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage 200 joint risk/winner failure-attribution protocol.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    report = protocol.freeze_joint_risk_winner_failure_attribution_protocol(
        stage199_report_path=artifacts / "primeqa_hybrid_joint_risk_winner_stage199.json",
        user_confirmed=True,
        confirmation_note=(
            "User approved the recommended next stage: freeze strict train-only "
            "attribution of Stage199 inner-eligibility failures before designing another "
            "candidate family."
        ),
    )
    visualizations = protocol.write_stage200_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifacts
        / "primeqa_hybrid_joint_risk_winner_failure_attribution_protocol_stage200_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = (
        output
        or artifacts / "primeqa_hybrid_joint_risk_winner_failure_attribution_protocol_stage200.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage 200 protocol: {output_path}")
    if not report["decision"]["protocol_valid"]:
        raise typer.Exit(code=1)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "evidence_summary": report["evidence_summary"],
        "diagnostic_population": frozen["diagnostic_population"],
        "constraint_attribution": frozen["constraint_attribution"],
        "fold_attribution": frozen["fold_attribution"],
        "question_context_attribution": frozen["question_context_attribution"],
        "execution_budget": frozen["execution_budget"],
        "resource_contract": frozen["resource_contract"],
        "authorization_boundary": frozen["authorization_boundary"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
