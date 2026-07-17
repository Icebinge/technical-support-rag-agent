from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_runtime_validation import (
    run_primeqa_hybrid_optional_sidecar_runtime_validation,
    write_primeqa_hybrid_optional_sidecar_runtime_validation_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage143 explicit optional runtime wiring validation.")


@app.command()
def main(
    stage141_protocol: Annotated[Path | None, typer.Option("--stage141-protocol")] = None,
    stage142_validation: Annotated[Path | None, typer.Option("--stage142-validation")] = None,
    stage139_regression: Annotated[Path | None, typer.Option("--stage139-regression")] = None,
    stage128_protocol: Annotated[Path | None, typer.Option("--stage128-protocol")] = None,
    stage125_protocol: Annotated[Path | None, typer.Option("--stage125-protocol")] = None,
    stage80_report: Annotated[Path | None, typer.Option("--stage80-report")] = None,
    train_split: Annotated[Path | None, typer.Option("--train-split")] = None,
    dev_split: Annotated[Path | None, typer.Option("--dev-split")] = None,
    documents: Annotated[Path | None, typer.Option("--documents")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
    user_confirmed_validation: Annotated[
        bool,
        typer.Option("--user-confirmed-validation/--no-user-confirmed-validation"),
    ] = False,
    confirmation_note: Annotated[str, typer.Option("--confirmation-note")] = "not confirmed",
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
    encoder_device: Annotated[str | None, typer.Option("--encoder-device")] = None,
) -> None:
    """Write aggregate-only Stage143 runtime validation and visualizations."""

    settings = ProjectSettings()
    artifact_dir = settings.artifact_dir
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    report = run_primeqa_hybrid_optional_sidecar_runtime_validation(
        stage141_protocol_path=stage141_protocol
        or artifact_dir / "primeqa_hybrid_nondefault_runtime_activation_protocol_stage141.json",
        stage142_validation_path=stage142_validation
        or artifact_dir / "primeqa_hybrid_strict_latency_validation_stage142.json",
        stage139_regression_path=stage139_regression
        or artifact_dir
        / "primeqa_hybrid_optional_sidecar_agent_entrypoint_validation_stage142_regression.json",
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
    visualizations = write_primeqa_hybrid_optional_sidecar_runtime_validation_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifact_dir / "primeqa_hybrid_optional_sidecar_runtime_validation_stage143_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_optional_sidecar_runtime_validation_stage143.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid Stage143 runtime validation: {output_path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "startup_cases": report.get("startup_cases"),
        "resource_summary": report.get("resource_summary"),
        "train_runtime_validation": report.get("train_runtime_validation"),
        "train_fold_reports": report.get("train_fold_reports"),
        "dev_runtime_report_only_validation": report.get("dev_runtime_report_only_validation"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "public_safe_contract": report.get("public_safe_contract"),
        "visualizations": report.get("visualizations"),
        "timing_seconds": report.get("timing_seconds"),
    }


if __name__ == "__main__":
    app()
