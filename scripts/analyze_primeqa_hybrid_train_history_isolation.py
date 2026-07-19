from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_validation as stage165,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage165 full-train paired history-isolation diagnostics.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    user_confirmed_full_train_pairing: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-full-train-pairing",
            help="Required confirmation for option A: 562 rows and 1124 Agent turns.",
        ),
    ] = False,
    stage164_correction: Annotated[
        Path | None,
        typer.Option("--stage164-correction", help="Completed Stage164 correction report."),
    ] = None,
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train JSONL."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Stage165 public aggregate JSON."),
    ] = None,
    private_output: Annotated[
        Path | None,
        typer.Option("--private-output", help="Ignored hashed paired diagnostics JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Stage165 SVG output directory."),
    ] = None,
) -> None:
    if not user_confirmed_full_train_pairing:
        raise typer.BadParameter(
            "Stage165 requires --user-confirmed-full-train-pairing for option A"
        )
    settings = ProjectSettings(
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=model_snapshot,
    )
    artifact_dir = settings.artifact_dir.resolve()
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    run = stage165.validate_primeqa_hybrid_train_history_isolation(
        settings=settings,
        stage164_correction_path=stage164_correction
        or artifact_dir / "primeqa_hybrid_gold_visible_refusal_contract_stage164.json",
        train_split_path=train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        user_confirmed_full_train_pairing=True,
        confirmation_note="user_selected_A_full_562_train_rows_1124_agent_turns",
        progress_sink=_write_progress,
    )

    private_path = private_output or (
        artifact_dir / "primeqa_hybrid_train_history_isolation_stage165_private.json"
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(run.private_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    report = run.public_report
    visualizations = stage165.write_stage165_visualizations(
        report=report,
        output_dir=(
            visualization_dir
            or artifact_dir / "primeqa_hybrid_train_history_isolation_stage165_visuals"
        ),
    )
    report = {
        **report,
        "private_diagnostic_artifact_written": {
            "path": str(private_path),
            "canonical_content_sha256": report["private_diagnostic_artifact_contract"][
                "canonical_content_sha256"
            ],
        },
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (artifact_dir / "primeqa_hybrid_train_history_isolation_stage165.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage165 public diagnostics: {output_path}")
    typer.echo(f"Saved Stage165 private hashed diagnostics: {private_path}")
    if report.get("decision", {}).get("all_process_guards_passed") is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = report.get("paired_diagnostics") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "train_diagnostic_protocol": report.get("train_diagnostic_protocol"),
        "workload_plan": report.get("workload_plan"),
        "grouped_fold_protocol": report.get("grouped_fold_protocol"),
        "runtime": report.get("runtime"),
        "primary_post_first_answerable_effect": diagnostics.get(
            "primary_post_first_answerable_effect"
        ),
        "gold_visible_post_first_answerable_effect": diagnostics.get(
            "gold_visible_post_first_answerable_effect"
        ),
        "unanswerable_post_first_safety_effect": diagnostics.get(
            "unanswerable_post_first_safety_effect"
        ),
        "first_turn_negative_control": diagnostics.get("first_turn_negative_control"),
        "question_alignment": diagnostics.get("question_alignment"),
        "grouped_fold_stability": diagnostics.get("grouped_fold_stability"),
        "timing_seconds": report.get("timing_seconds"),
        "guard_checks": report.get("guard_checks"),
        "public_safe_contract": report.get("public_safe_contract"),
        "decision": report.get("decision"),
        "visualizations": report.get("visualizations"),
    }


if __name__ == "__main__":
    app()
