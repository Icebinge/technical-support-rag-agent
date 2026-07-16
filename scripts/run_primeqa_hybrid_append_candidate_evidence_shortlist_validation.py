from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_append_candidate_evidence_shortlist_validation import (
    run_primeqa_hybrid_append_candidate_evidence_shortlist_validation,
    write_primeqa_hybrid_append_candidate_evidence_shortlist_validation_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage132 PrimeQA hybrid append-candidate evidence shortlist validation."
)


@app.command()
def main(
    stage131_protocol: Annotated[
        Path | None,
        typer.Option("--stage131-protocol", help="Stage131 protocol JSON."),
    ] = None,
    stage128_protocol: Annotated[
        Path | None,
        typer.Option("--stage128-protocol", help="Stage128 protocol JSON."),
    ] = None,
    stage125_protocol: Annotated[
        Path | None,
        typer.Option("--stage125-protocol", help="Stage125 executable protocol JSON."),
    ] = None,
    stage80_report: Annotated[
        Path | None,
        typer.Option("--stage80-report", help="Stage80 dense cache feasibility JSON."),
    ] = None,
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train split JSONL."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Frozen Stage68 dev split JSONL."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training/dev sections JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage132 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-validation/--no-user-confirmed-validation",
            help="Required confirmation for Stage132 train/dev validation.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
    include_dense_channels: Annotated[
        bool,
        typer.Option(
            "--include-dense-channels/--no-include-dense-channels",
            help="Use only existing local dense caches when enabled.",
        ),
    ] = True,
    encoder_batch_size: Annotated[
        int,
        typer.Option("--encoder-batch-size", help="SentenceTransformer query batch size."),
    ] = 64,
    encoder_device: Annotated[
        str | None,
        typer.Option("--encoder-device", help="Optional SentenceTransformer device."),
    ] = None,
) -> None:
    """Write the Stage132 train/dev-only append-shortlist validation report."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    stage131_protocol_path = stage131_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_append_candidate_evidence_shortlist_protocol_stage131.json"
    )
    stage128_protocol_path = stage128_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json"
    )
    stage125_protocol_path = stage125_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json"
    )
    stage80_report_path = stage80_report or (
        settings.artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json"
    )
    train_split_path = train_split or (
        split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    )
    dev_split_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_append_candidate_evidence_shortlist_validation_stage132_visuals"
    )

    report = run_primeqa_hybrid_append_candidate_evidence_shortlist_validation(
        stage131_protocol_path=stage131_protocol_path,
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        include_dense_channels=include_dense_channels,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
    )
    visualizations = (
        write_primeqa_hybrid_append_candidate_evidence_shortlist_validation_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage132 validation: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "candidate_pool_summary": report.get("candidate_pool_summary"),
        "train_selection": report.get("train_selection"),
        "dev_report_observations": report.get("dev_report_observations"),
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
