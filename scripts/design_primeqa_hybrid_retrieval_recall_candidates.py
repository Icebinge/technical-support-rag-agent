from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_retrieval_recall_candidate_design import (
    design_primeqa_hybrid_retrieval_recall_candidates,
    write_primeqa_hybrid_retrieval_recall_candidate_design_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help=(
        "Run Stage76 train/dev-only retrieval-recall candidate design from the "
        "Stage75 BM25 miss report."
    )
)


@app.command()
def main(
    stage75_report: Annotated[
        Path | None,
        typer.Option("--stage75-report", help="Stage75 BM25 miss report JSON path."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage76 design report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run Stage76 candidate design and write the report."""

    settings = ProjectSettings()
    stage75_report_path = stage75_report or (
        settings.artifact_dir / "primeqa_hybrid_bm25_top10_miss_analysis_stage75.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_recall_candidate_design_stage76.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals"
    )

    report = design_primeqa_hybrid_retrieval_recall_candidates(
        stage75_report_path=stage75_report_path,
    )
    visualizations = write_primeqa_hybrid_retrieval_recall_candidate_design_visualizations(
        report=report,
        output_dir=visualization_output_dir,
    )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid retrieval-recall candidate design: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "stage75_summary": report["stage75_summary"],
        "recommended_execution_order": report["recommended_execution_order"],
        "candidate_summary": [
            {
                "candidate_id": candidate["candidate_id"],
                "status": candidate["status"],
                "risk_level": candidate["risk_level"],
                "priority_score": candidate["priority_score"],
                "target_miss_count": candidate["target_miss_count"],
                "target_miss_count_by_split": candidate["target_miss_count_by_split"],
            }
            for candidate in report["candidate_designs"]
        ],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
