from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_safety_constrained_lambdamart_protocol as protocol,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage 193 safety-constrained LambdaMART protocol.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    report = protocol.freeze_safety_constrained_lambdamart_protocol(
        stage192_report_path=(
            artifacts / "primeqa_hybrid_rank_capped_safety_pool_failure_stage192.json"
        ),
        user_confirmed=True,
        confirmation_note=(
            "User explicitly selected route A: LightGBM LambdaMART grouped ranking "
            "with an independent unsafe-risk head."
        ),
    )
    visualizations = protocol.write_stage193_visualizations(
        report=report,
        output_dir=(
            visualization_dir
            or artifacts / "primeqa_hybrid_safety_constrained_lambdamart_protocol_stage193_visuals"
        ),
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = (
        output or artifacts / "primeqa_hybrid_safety_constrained_lambdamart_protocol_stage193.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage 193 protocol: {output_path}")
    if not report["decision"]["protocol_valid"]:
        raise typer.Exit(code=1)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "dependency": frozen["dependency_contract"],
        "evidence_summary": report["evidence_summary"],
        "first_stage_pool": frozen["first_stage_pool"],
        "lambdamart_contract": frozen["lambdamart_contract"],
        "unsafe_risk_contract": frozen["unsafe_risk_contract"],
        "within_pool_selection": frozen["within_pool_selection"],
        "candidate_grid": frozen["candidate_grid"],
        "cross_validation": frozen["cross_validation"],
        "inner_selection": frozen["inner_selection"],
        "advancement_gates": frozen["advancement_gates"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
