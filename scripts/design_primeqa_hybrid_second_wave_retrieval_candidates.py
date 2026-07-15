from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_second_wave_retrieval_candidate_design import (
    design_primeqa_hybrid_second_wave_retrieval_candidates,
    write_primeqa_hybrid_second_wave_retrieval_candidate_design_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage84 second-wave PrimeQA hybrid retrieval candidate design."
)


@app.command()
def main(
    stage75_report: Annotated[
        Path | None,
        typer.Option("--stage75-report", help="Stage75 BM25 miss report JSON."),
    ] = None,
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
        typer.Option("--stage81-report", help="Stage81 dense+sparse report JSON."),
    ] = None,
    stage82_report: Annotated[
        Path | None,
        typer.Option("--stage82-report", help="Stage82 BM25 grid report JSON."),
    ] = None,
    stage83_report: Annotated[
        Path | None,
        typer.Option("--stage83-report", help="Stage83 exhaustion summary JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage84 design report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_route: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-route/--no-user-confirmed-route",
            help="Required confirmation for Stage83's recommended Stage84 route.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage84 second-wave candidate design from saved reports."""

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals"
    )
    report = design_primeqa_hybrid_second_wave_retrieval_candidates(
        stage75_report_path=stage75_report
        or settings.artifact_dir
        / "primeqa_hybrid_bm25_top10_miss_analysis_stage75.json",
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
        stage83_report_path=stage83_report
        or settings.artifact_dir
        / "primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83.json",
        user_confirmed_route=user_confirmed_route,
        confirmation_note=confirmation_note,
    )
    visualizations = (
        write_primeqa_hybrid_second_wave_retrieval_candidate_design_visualizations(
            report=report,
            output_dir=visualization_output_dir,
        )
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
    typer.echo(f"Saved PrimeQA hybrid second-wave retrieval design: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "user_confirmation": report["user_confirmation"],
        "stage75_miss_summary": report["stage75_miss_summary"],
        "prior_route_evidence": report["prior_route_evidence"],
        "recommended_execution_order": report["recommended_execution_order"],
        "candidate_summary": [
            {
                "candidate_id": candidate["candidate_id"],
                "status": candidate["status"],
                "risk_level": candidate["risk_level"],
                "priority_score": candidate["priority_score"],
                "target_miss_count": candidate["target_miss_count"],
                "target_miss_count_by_split": candidate["target_miss_count_by_split"],
                "prior_signal_score": candidate["prior_signal_score"],
            }
            for candidate in report["candidate_designs"]
        ],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
