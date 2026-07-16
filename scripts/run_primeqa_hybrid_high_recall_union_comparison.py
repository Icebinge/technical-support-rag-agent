from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    run_primeqa_hybrid_high_recall_union_comparison,
    write_primeqa_hybrid_high_recall_union_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage116 PrimeQA hybrid high-recall union candidate-pool comparison."
)


@app.command()
def main(
    stage115_report: Annotated[
        Path | None,
        typer.Option("--stage115-report", help="Stage115 stop decision JSON."),
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
        typer.Option("--output", help="Output Stage116 report JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_direction: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-direction/--no-user-confirmed-direction",
            help="Required confirmation for the Stage116 high-recall union direction.",
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
    channel_top_k: Annotated[
        int,
        typer.Option("--channel-top-k", help="Per-channel first-stage retrieval depth."),
    ] = 200,
    encoder_batch_size: Annotated[
        int,
        typer.Option("--encoder-batch-size", help="SentenceTransformer query batch size."),
    ] = 64,
    encoder_device: Annotated[
        str | None,
        typer.Option("--encoder-device", help="Optional SentenceTransformer device."),
    ] = None,
) -> None:
    """Write the Stage116 train/dev-only high-recall union candidate-pool report."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    stage115_report_path = stage115_report or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115.json"
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
        settings.artifact_dir / "primeqa_hybrid_high_recall_union_stage116.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir / "primeqa_hybrid_high_recall_union_stage116_visuals"
    )

    report = run_primeqa_hybrid_high_recall_union_comparison(
        stage115_report_path=stage115_report_path,
        stage80_report_path=stage80_report_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        user_confirmed_direction=user_confirmed_direction,
        confirmation_note=confirmation_note,
        include_dense_channels=include_dense_channels,
        channel_top_k=channel_top_k,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
    )
    visualizations = write_primeqa_hybrid_high_recall_union_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage116 high-recall union report: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "split_contract": report["split_contract"],
        "analysis_config": report["analysis_config"],
        "loaded_data_summary": report["loaded_data_summary"],
        "dense_channel_preflight": {
            "status": report["dense_channel_preflight"]["status"],
            "can_run_without_download": report["dense_channel_preflight"][
                "can_run_without_download"
            ],
            "dense_cache_config_count": len(
                report["dense_channel_preflight"]["dense_cache_configs"]
            ),
        },
        "channel_catalog": report["channel_catalog"],
        "candidate_pool_metrics_by_split": report["candidate_pool_metrics_by_split"],
        "comparisons_to_baseline": report["comparisons_to_baseline"],
        "train_fold_stability": {
            key: value
            for key, value in report.get("train_fold_stability", {}).items()
            if key != "fold_metrics"
        },
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
