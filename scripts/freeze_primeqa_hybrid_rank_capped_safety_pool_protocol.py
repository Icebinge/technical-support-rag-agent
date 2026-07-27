from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_rank_capped_safety_pool_protocol as protocol,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Freeze the Stage 190 rank-capped safety-pool protocol.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    report = protocol.freeze_rank_capped_safety_pool_protocol(
        stage189_report_path=artifacts / "primeqa_hybrid_gain_sensitive_failure_stage189.json",
        user_confirmed=True,
        confirmation_note=(
            "User approved the recommended next stage; Stage189 then identified "
            "safety-frontier exclusion as the primary bottleneck."
        ),
    )
    visualizations = protocol.write_stage190_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifacts / "primeqa_hybrid_rank_capped_safety_pool_protocol_stage190_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = (
        output or artifacts / "primeqa_hybrid_rank_capped_safety_pool_protocol_stage190.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage 190 protocol: {output_path}")
    if not report["decision"]["protocol_valid"]:
        raise typer.Exit(code=1)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "evidence_summary": report["evidence_summary"],
        "rank_capped_safety_pool": report["frozen_protocol"]["rank_capped_safety_pool"],
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
