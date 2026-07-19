from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_untouched_rrf_dev_validation import (
    PrimeQAHybridUntouchedRRFDevVisualization,
    run_primeqa_hybrid_untouched_rrf_dev_validation,
    write_stage163_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run the frozen Stage163 one-shot development validation.")


@app.command()
def main(
    stage162_report: Annotated[
        Path | None,
        typer.Option("--stage162-report", help="Completed Stage162 public report JSON."),
    ] = None,
    stage160_report: Annotated[
        Path | None,
        typer.Option("--stage160-report", help="Completed Stage160 public report JSON."),
    ] = None,
    stage80_report: Annotated[
        Path | None,
        typer.Option("--stage80-report", help="Stage80 local dense-cache report JSON."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Exact frozen Stage68 development JSONL."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training/dev technote sections JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage163 public aggregate JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage163 SVGs."),
    ] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-validation/--no-user-confirmed-validation",
            help="Required confirmation for the frozen Stage163 development gate.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
    include_dense_channels: Annotated[
        bool,
        typer.Option(
            "--include-dense-channels/--no-include-dense-channels",
            help="Use only the already-authorized local dense caches when enabled.",
        ),
    ] = True,
    encoder_batch_size: Annotated[
        int,
        typer.Option("--encoder-batch-size", help="Dense query encoder batch size."),
    ] = 64,
    encoder_device: Annotated[
        str | None,
        typer.Option("--encoder-device", help="Optional dense query encoder device."),
    ] = None,
) -> None:
    """Write the Stage163 one-shot development report and SVG visualizations."""

    settings = ProjectSettings()
    artifact_dir = settings.artifact_dir
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    report = run_primeqa_hybrid_untouched_rrf_dev_validation(
        stage162_report_path=stage162_report
        or artifact_dir / "primeqa_hybrid_conservative_context_swap_stage162.json",
        stage160_report_path=stage160_report
        or artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160.json",
        stage80_report_path=stage80_report
        or artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json",
        dev_split_path=dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl",
        documents_path=documents
        or settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        include_dense_channels=include_dense_channels,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        progress_sink=_write_progress,
    )
    visual_dir = visualization_dir or (
        artifact_dir / "primeqa_hybrid_untouched_rrf_dev_validation_stage163_visuals"
    )
    visualizations = write_stage163_visualizations(report=report, output_dir=visual_dir)
    report = {**report, "visualizations": _visualization_rows(visualizations)}
    output_path = output or (
        artifact_dir / "primeqa_hybrid_untouched_rrf_dev_validation_stage163.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage163 untouched-RRF development report: {output_path}")


def _visualization_rows(
    visualizations: list[PrimeQAHybridUntouchedRRFDevVisualization],
) -> list[dict[str, str]]:
    return [{"name": artifact.name, "path": artifact.path} for artifact in visualizations]


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "split_contract": report["split_contract"],
        "candidate_pool_summary": report["candidate_pool_summary"],
        "dev_results": report["dev_results"],
        "policy_comparison": report["policy_comparison"],
        "policy_adoption": report["policy_adoption"],
        "guard_checks": report["guard_checks"],
        "public_safe_contract": report["public_safe_contract"],
        "decision": report["decision"],
        "timing_seconds": report["timing_seconds"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
