from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_train_history_evidence_gate_cv as analysis
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage167 train-only evidence-gate nested CV.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
    user_confirmed_stage167: Annotated[
        bool,
        typer.Option("--user-confirmed-stage167/--no-user-confirmed-stage167"),
    ] = False,
    confirmation_note: Annotated[str, typer.Option("--confirmation-note")] = "not confirmed",
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
    encoder_device: Annotated[str | None, typer.Option("--encoder-device")] = None,
) -> None:
    """Run nested CV without accepting development or test paths."""

    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    split_dir = artifacts / "primeqa_hybrid_split_stage68_splits"
    report = analysis.run_stage167_evidence_gate_nested_cv(
        stage161_report_path=artifacts / "primeqa_hybrid_protected_context_selector_stage161.json",
        stage165_correction_path=artifacts
        / "primeqa_hybrid_train_history_isolation_transition_correction_stage165.json",
        stage165_private_path=artifacts
        / "primeqa_hybrid_train_history_isolation_sharded_stage165_private.json",
        stage166_report_path=artifacts
        / "primeqa_hybrid_train_history_safety_gate_cv_stage166.json",
        stage80_report_path=artifacts / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json",
        train_split_path=split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        documents_path=settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        user_confirmed_stage167=user_confirmed_stage167,
        confirmation_note=confirmation_note,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        progress_sink=_write_progress,
    )
    visual_dir = visualization_dir or (
        artifacts / "primeqa_hybrid_train_history_evidence_gate_cv_stage167_visuals"
    )
    visualizations = analysis.write_stage167_visualizations(report=report, output_dir=visual_dir)
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifacts / "primeqa_hybrid_train_history_evidence_gate_cv_stage167.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage167 evidence-gate nested CV: {output_path}")
    if report["decision"]["all_process_guards_passed"] is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "evidence_provenance": report["evidence_provenance"],
        "case_summary": report["case_summary"],
        "candidate_family": report["candidate_family"],
        "nested_cv": report["nested_cv"],
        "timing_seconds": report["timing_seconds"],
        "guard_checks": report["guard_checks"],
        "public_safe_contract": report["public_safe_contract"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
