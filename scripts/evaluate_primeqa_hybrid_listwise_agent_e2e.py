from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_listwise_agent_e2e as analysis
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage178A train-only listwise tool-Agent OOF E2E evaluation.")


@app.command()
def main(
    model_snapshot: Annotated[Path, typer.Option("--model-snapshot")],
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    private_output: Annotated[Path | None, typer.Option("--private-output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
) -> None:
    """Train OOF/full listwise models and compare two real Agent context paths."""

    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    split_dir = artifacts / "primeqa_hybrid_split_stage68_splits"
    checkpoint_path = checkpoint or artifacts / "stage178a_listwise_full_train_checkpoint"
    report, private_report = analysis.run_stage178a_listwise_agent_e2e(
        stage178_alignment_path=artifacts / "stage178_candidate_alignment_audit.json",
        stage177_report_path=artifacts / "primeqa_hybrid_listwise_reranker_cv_stage177.json",
        stage128_protocol_path=(
            artifacts / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json"
        ),
        stage125_protocol_path=(
            artifacts / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json"
        ),
        stage80_report_path=(
            artifacts / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json"
        ),
        train_split_path=split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        documents_path=settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        model_snapshot_path=model_snapshot,
        checkpoint_path=checkpoint_path,
        encoder_batch_size=encoder_batch_size,
        progress_sink=_write_progress,
    )
    visuals_path = visualization_dir or (
        artifacts / "primeqa_hybrid_listwise_agent_e2e_stage178a_visuals"
    )
    visualizations = analysis.write_stage178a_visualizations(
        report=report,
        output_dir=visuals_path,
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or artifacts / "primeqa_hybrid_listwise_agent_e2e_stage178a.json"
    private_path = private_output or (
        artifacts / "primeqa_hybrid_listwise_agent_e2e_stage178a_private.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    private_path.write_text(
        json.dumps(private_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage178A report: {output_path}")
    typer.echo(f"Saved Stage178A private OOF scores: {private_path}")
    if report["decision"]["status"] == "stage178a_process_invalid":
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "training": report["training"],
        "agent_e2e": report["agent_e2e"],
        "cpu_checkpoint_latency_probe": report["cpu_checkpoint_latency_probe"],
        "cpu_checkpoint_runtime_smoke": report["cpu_checkpoint_runtime_smoke"],
        "resource_consumption": report["resource_consumption"],
        "timing_seconds": report["timing_seconds"],
        "quality_gates": report["quality_gates"],
        "process_guards": report["process_guards"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
