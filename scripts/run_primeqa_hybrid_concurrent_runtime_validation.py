from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_validation import (
    run_primeqa_hybrid_concurrent_runtime_validation,
    write_primeqa_hybrid_concurrent_runtime_validation_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage145 strict practical concurrency-four validation.")


@app.command()
def main(
    stage144_protocol: Annotated[Path | None, typer.Option("--stage144-protocol")] = None,
    stage143_validation: Annotated[Path | None, typer.Option("--stage143-validation")] = None,
    stage128_protocol: Annotated[Path | None, typer.Option("--stage128-protocol")] = None,
    stage125_protocol: Annotated[Path | None, typer.Option("--stage125-protocol")] = None,
    stage80_report: Annotated[Path | None, typer.Option("--stage80-report")] = None,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    dev_split: Annotated[Path | None, typer.Option("--dev-split")] = None,
    documents: Annotated[Path | None, typer.Option("--documents")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir"),
    ] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option("--user-confirmed-validation/--no-user-confirmed-validation"),
    ] = False,
    confirmation_note: Annotated[str, typer.Option("--confirmation-note")] = "not confirmed",
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
    encoder_device: Annotated[str | None, typer.Option("--encoder-device")] = None,
) -> None:
    """Write the aggregate-only Stage145 report and visualizations."""

    settings = ProjectSettings()
    artifact_dir = settings.artifact_dir
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    report = run_primeqa_hybrid_concurrent_runtime_validation(
        stage144_protocol_path=stage144_protocol
        or artifact_dir / "primeqa_hybrid_concurrent_runtime_validation_protocol_stage144.json",
        stage143_validation_path=stage143_validation
        or artifact_dir / "primeqa_hybrid_optional_sidecar_runtime_validation_stage143.json",
        stage128_protocol_path=stage128_protocol
        or artifact_dir / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json",
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
    visualizations = write_primeqa_hybrid_concurrent_runtime_validation_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_concurrent_runtime_validation_stage145_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_concurrent_runtime_validation_stage145.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage145 concurrent validation: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    train = report.get("train_validation") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "resource_summary": report.get("resource_summary"),
        "warmup": report.get("warmup"),
        "overload_probe": report.get("overload_probe"),
        "train_summary": {
            "accepted_request_count": train.get("accepted_request_count"),
            "complete_pass_count": train.get("complete_pass_count"),
            "global_pooled_report": train.get("global_pooled_report"),
            "failed_latency_gate_scopes": train.get("failed_latency_gate_scopes"),
            "cross_request_contamination_count": train.get("cross_request_contamination_count"),
            "runtime_counter_delta": train.get("runtime_counter_delta"),
        },
        "dev_report_only_validation": report.get("dev_report_only_validation"),
        "concurrency_policy_evaluation": report.get("concurrency_policy_evaluation"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
