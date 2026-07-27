from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_safety_first_frontier_protocol as protocol
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage 195 safety-first frontier protocol.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    report = protocol.freeze_safety_first_frontier_protocol(
        stage194_report_path=(
            artifacts / "primeqa_hybrid_safety_constrained_lambdamart_stage194.json"
        ),
        user_confirmed=True,
        confirmation_note=(
            "User asked to continue with the recommended next stage after Stage 194: "
            "freeze the safety-first cost-sensitive frontier experiment."
        ),
    )
    visualizations = protocol.write_stage195_visualizations(
        report=report,
        output_dir=(
            visualization_dir
            or artifacts / "primeqa_hybrid_safety_first_frontier_protocol_stage195_visuals"
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
        output or artifacts / "primeqa_hybrid_safety_first_frontier_protocol_stage195.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage 195 protocol: {output_path}")
    if not report["decision"]["protocol_valid"]:
        raise typer.Exit(code=1)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    frozen = report["frozen_protocol"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "evidence_summary": report["evidence_summary"],
        "first_stage_pool": frozen["first_stage_pool"],
        "gain_ranker": frozen["gain_ranker"],
        "cost_sensitive_unsafe_head": frozen["cost_sensitive_unsafe_head"],
        "safety_first_frontier": frozen["safety_first_frontier"],
        "candidate_grid": frozen["candidate_grid"],
        "cross_validation": frozen["cross_validation"],
        "resource_contract": frozen["resource_contract"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
