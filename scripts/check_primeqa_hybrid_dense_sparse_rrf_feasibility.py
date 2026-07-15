from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_dense_sparse_rrf_feasibility import (
    check_primeqa_hybrid_dense_sparse_rrf_feasibility,
    write_primeqa_hybrid_dense_sparse_rrf_feasibility_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage80 local feasibility check for PrimeQA hybrid dense+sparse RRF."
)


@app.command()
def main(
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training_dev_technotes.sections.json."),
    ] = None,
    pyproject: Annotated[
        Path,
        typer.Option("--pyproject", help="Project pyproject.toml path."),
    ] = Path("pyproject.toml"),
    stage76_report: Annotated[
        Path | None,
        typer.Option("--stage76-report", help="Stage76 candidate design report JSON."),
    ] = None,
    stage79_report: Annotated[
        Path | None,
        typer.Option("--stage79-report", help="Stage79 section BM25 report JSON."),
    ] = None,
    dense_cache_dir: Annotated[
        Path | None,
        typer.Option("--dense-cache-dir", help="Local dense embedding cache directory."),
    ] = None,
    huggingface_hub_dir: Annotated[
        Path | None,
        typer.Option("--huggingface-hub-dir", help="Local Hugging Face hub cache dir."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage80 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    legacy_metric: Annotated[
        list[Path] | None,
        typer.Option("--legacy-metric", help="Optional old dense/hybrid metric report."),
    ] = None,
) -> None:
    """Write a no-download dense+sparse RRF feasibility report."""

    settings = ProjectSettings()
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    stage76_report_path = stage76_report or (
        settings.artifact_dir / "primeqa_hybrid_retrieval_recall_candidate_design_stage76.json"
    )
    stage79_report_path = stage79_report or (
        settings.artifact_dir / "primeqa_hybrid_section_bm25_doc_rollup_stage79.json"
    )
    resolved_dense_cache_dir = dense_cache_dir or settings.data_dir / "indexes" / "dense"
    resolved_huggingface_hub_dir = (
        huggingface_hub_dir or Path.home() / ".cache" / "huggingface" / "hub"
    )
    output_path = output or (
        settings.artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals"
    )
    legacy_metric_paths = legacy_metric or _default_legacy_metric_paths(settings)

    report = check_primeqa_hybrid_dense_sparse_rrf_feasibility(
        documents_path=documents_path,
        pyproject_path=pyproject,
        stage76_report_path=stage76_report_path,
        stage79_report_path=stage79_report_path,
        dense_cache_dir=resolved_dense_cache_dir,
        huggingface_hub_dir=resolved_huggingface_hub_dir,
        legacy_metric_paths=legacy_metric_paths,
    )
    visualizations = write_primeqa_hybrid_dense_sparse_rrf_feasibility_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid dense+sparse RRF feasibility report: {output_path}")


def _default_legacy_metric_paths(settings: ProjectSettings) -> list[Path]:
    return [
        settings.artifact_dir / "dense_dev_metrics.json",
        settings.artifact_dir / "hybrid_dev_metrics.json",
        settings.artifact_dir / "dense_e5_small_v2_512_dev_metrics.json",
        settings.artifact_dir / "hybrid_e5_small_v2_512_dev_metrics.json",
    ]


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "loaded_data_summary": report["loaded_data_summary"],
        "dependency_checks": report["dependency_checks"],
        "code_readiness": report["code_readiness"],
        "dense_cache_candidates": report["dense_cache_candidates"],
        "candidate_options": report["candidate_options"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
