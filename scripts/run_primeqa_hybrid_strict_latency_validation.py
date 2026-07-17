from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_strict_latency_validation import (
    run_primeqa_hybrid_strict_latency_validation,
    write_primeqa_hybrid_strict_latency_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage142 strict warm single-request latency validation.")


@app.command()
def main(
    stage141_protocol: Annotated[
        Path | None,
        typer.Option("--stage141-protocol", help="Stage141 activation protocol JSON."),
    ] = None,
    stage140_validation: Annotated[
        Path | None,
        typer.Option("--stage140-validation", help="Stage140 performance JSON."),
    ] = None,
    stage128_protocol: Annotated[
        Path | None,
        typer.Option("--stage128-protocol", help="Stage128 candidate-pool protocol JSON."),
    ] = None,
    stage127_review: Annotated[
        Path | None,
        typer.Option("--stage127-review", help="Stage127 selected-config review JSON."),
    ] = None,
    stage125_protocol: Annotated[
        Path | None,
        typer.Option("--stage125-protocol", help="Stage125 candidate protocol JSON."),
    ] = None,
    stage80_report: Annotated[
        Path | None,
        typer.Option("--stage80-report", help="Stage80 dense-cache report JSON."),
    ] = None,
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train JSONL."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Frozen Stage68 dev JSONL."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training/dev sections JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage142 public-safe report JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage142 SVG charts."),
    ] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-validation/--no-user-confirmed-validation",
            help="Required confirmation for Stage142 train-first validation.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
    encoder_batch_size: Annotated[
        int,
        typer.Option("--encoder-batch-size", help="SentenceTransformer query batch size."),
    ] = 64,
    encoder_device: Annotated[
        str | None,
        typer.Option("--encoder-device", help="Optional SentenceTransformer device."),
    ] = None,
) -> None:
    """Write Stage142 aggregate-only strict latency evidence and visualizations."""

    settings = ProjectSettings()
    artifact_dir = settings.artifact_dir
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    report = run_primeqa_hybrid_strict_latency_validation(
        stage141_protocol_path=stage141_protocol
        or artifact_dir / "primeqa_hybrid_nondefault_runtime_activation_protocol_stage141.json",
        stage140_validation_path=stage140_validation
        or artifact_dir
        / "primeqa_hybrid_online_candidate_pool_performance_validation_stage140.json",
        stage128_protocol_path=stage128_protocol
        or artifact_dir / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json",
        stage127_review_path=stage127_review
        or artifact_dir
        / "primeqa_hybrid_prefix_preserving_recall_expansion_selected_config_review_stage127.json",
        stage125_protocol_path=stage125_protocol
        or artifact_dir
        / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json",
        stage80_report_path=stage80_report
        or artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json",
        train_split_path=train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        dev_split_path=dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl",
        documents_path=documents
        or settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
    )
    visualizations = write_primeqa_hybrid_strict_latency_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_strict_latency_validation_stage142_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_strict_latency_validation_stage142.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage142 strict latency validation: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "optimization_contract": report.get("optimization_contract"),
        "strict_latency_slo": report.get("strict_latency_slo"),
        "warmup": report.get("warmup"),
        "train_validation": report.get("train_validation"),
        "dev_report_only_validation": report.get("dev_report_only_validation"),
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
