from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_retrieval_recall_exhaustion_summary import (
    summarize_primeqa_hybrid_retrieval_recall_exhaustion,
    write_primeqa_hybrid_retrieval_recall_exhaustion_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage83 PrimeQA hybrid retrieval-recall exhaustion summary."
)


@app.command()
def main(
    stage76_report: Annotated[
        Path | None,
        typer.Option("--stage76-report", help="Stage76 candidate design report JSON."),
    ] = None,
    stage77_report: Annotated[
        Path | None,
        typer.Option("--stage77-report", help="Stage77 query-view report JSON."),
    ] = None,
    stage78_report: Annotated[
        Path | None,
        typer.Option("--stage78-report", help="Stage78 fielded BM25 report JSON."),
    ] = None,
    stage79_report: Annotated[
        Path | None,
        typer.Option("--stage79-report", help="Stage79 section BM25 report JSON."),
    ] = None,
    stage80_report: Annotated[
        Path | None,
        typer.Option("--stage80-report", help="Stage80 dense feasibility report JSON."),
    ] = None,
    stage81_report: Annotated[
        Path | None,
        typer.Option("--stage81-report", help="Stage81 dense+sparse RRF report JSON."),
    ] = None,
    stage82_report: Annotated[
        Path | None,
        typer.Option("--stage82-report", help="Stage82 BM25 grid report JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage83 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Write Stage83 summary from saved Stage76-Stage82 reports."""

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals"
    )
    report = summarize_primeqa_hybrid_retrieval_recall_exhaustion(
        stage76_report_path=stage76_report
        or settings.artifact_dir
        / "primeqa_hybrid_retrieval_recall_candidate_design_stage76.json",
        stage77_report_path=stage77_report
        or settings.artifact_dir
        / "primeqa_hybrid_query_view_ablation_stage77.json",
        stage78_report_path=stage78_report
        or settings.artifact_dir
        / "primeqa_hybrid_fielded_bm25_fusion_stage78.json",
        stage79_report_path=stage79_report
        or settings.artifact_dir
        / "primeqa_hybrid_section_bm25_doc_rollup_stage79.json",
        stage80_report_path=stage80_report
        or settings.artifact_dir
        / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json",
        stage81_report_path=stage81_report
        or settings.artifact_dir
        / "primeqa_hybrid_dense_sparse_rrf_comparison_stage81.json",
        stage82_report_path=stage82_report
        or settings.artifact_dir
        / "primeqa_hybrid_bm25_k1_b_grid_stage82.json",
    )
    visualizations = write_primeqa_hybrid_retrieval_recall_exhaustion_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid retrieval-recall exhaustion report: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "candidate_outcomes": report["candidate_outcomes"],
        "blocked_candidate": report["blocked_candidate"],
        "dev_only_observations": report["dev_only_observations"],
        "aggregate_summary": report["aggregate_summary"],
        "next_route_options": report["next_route_options"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
