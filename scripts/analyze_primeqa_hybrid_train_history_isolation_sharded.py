from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_sharded_validation as sharded,
)
from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_validation as stage165,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run the confirmed Stage165 12-process train diagnostic.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    user_confirmed_12_process_sharding: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-12-process-sharding",
            help="Required confirmation for option A: 12 fresh sequential processes.",
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
    shard_artifact_dir: Annotated[
        Path | None,
        typer.Option("--shard-artifact-dir", help="New ignored shard artifact directory."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Stage165 final or failed public JSON."),
    ] = None,
    private_output: Annotated[
        Path | None,
        typer.Option("--private-output", help="Ignored merged content-free diagnostics."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Stage165 merged SVG directory."),
    ] = None,
) -> None:
    if not user_confirmed_12_process_sharding:
        raise typer.BadParameter(
            "Stage165 requires --user-confirmed-12-process-sharding for option A"
        )
    settings = ProjectSettings(
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=model_snapshot,
    )
    artifact_dir = settings.artifact_dir.resolve()
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    run = sharded.validate_primeqa_hybrid_train_history_isolation_sharded(
        settings=settings,
        stage164_correction_path=stage164_correction
        or artifact_dir / "primeqa_hybrid_gold_visible_refusal_contract_stage164.json",
        train_split_path=train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        shard_artifact_dir=shard_artifact_dir
        or artifact_dir / "primeqa_hybrid_train_history_isolation_stage165_shards",
        user_confirmed_12_process_sharding=True,
        progress_sink=_write_progress,
    )
    report = run.public_report
    if run.private_report is not None:
        private_path = private_output or (
            artifact_dir / "primeqa_hybrid_train_history_isolation_sharded_stage165_private.json"
        )
        private_path.parent.mkdir(parents=True, exist_ok=True)
        private_path.write_text(
            json.dumps(run.private_report, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        visuals = stage165.write_stage165_visualizations(
            report=report,
            output_dir=(
                visualization_dir
                or artifact_dir / "primeqa_hybrid_train_history_isolation_sharded_stage165_visuals"
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
            "visualizations": [{"name": visual.name, "path": visual.path} for visual in visuals],
        }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_train_history_isolation_sharded_stage165.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage165 sharded diagnostics: {output_path}")
    if report["decision"].get("all_process_guards_passed") is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = report.get("paired_diagnostics") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "sharding_plan": report.get("sharding_plan"),
        "runtime": report.get("runtime"),
        "execution": report.get("execution"),
        "primary_post_first_answerable_effect": diagnostics.get(
            "primary_post_first_answerable_effect"
        ),
        "unanswerable_post_first_safety_effect": diagnostics.get(
            "unanswerable_post_first_safety_effect"
        ),
        "grouped_fold_stability": diagnostics.get("grouped_fold_stability"),
        "timing_seconds": report.get("timing_seconds"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "visualizations": report.get("visualizations"),
    }


if __name__ == "__main__":
    app()
