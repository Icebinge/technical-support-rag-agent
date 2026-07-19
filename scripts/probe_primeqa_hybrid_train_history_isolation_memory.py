from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_memory_probe as memory_probe,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Probe Stage165 thread 37 CUDA memory without retrying.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    user_confirmed_thread37_probe: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-thread37-probe",
            help="Required confirmation for option A: at most eight train-only turns.",
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
        typer.Option("--output", help="Stage165 memory-probe public aggregate JSON."),
    ] = None,
    private_output: Annotated[
        Path | None,
        typer.Option("--private-output", help="Ignored content-free event aggregate JSON."),
    ] = None,
    private_event_jsonl: Annotated[
        Path | None,
        typer.Option(
            "--private-event-jsonl",
            help="Ignored incrementally flushed content-free event JSONL.",
        ),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Stage165 memory-probe SVG directory."),
    ] = None,
) -> None:
    if not user_confirmed_thread37_probe:
        raise typer.BadParameter("Stage165 memory probe requires --user-confirmed-thread37-probe")
    settings = ProjectSettings(
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=model_snapshot,
    )
    artifact_dir = settings.artifact_dir.resolve()
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    event_path = private_event_jsonl or (
        artifact_dir / "primeqa_hybrid_train_history_isolation_memory_probe_stage165_events.jsonl"
    )
    run = memory_probe.run_stage165_memory_probe(
        settings=settings,
        stage164_correction_path=stage164_correction
        or artifact_dir / "primeqa_hybrid_gold_visible_refusal_contract_stage164.json",
        train_split_path=train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        private_event_jsonl_path=event_path,
        user_confirmed_thread37_probe=True,
    )

    private_path = private_output or (
        artifact_dir / "primeqa_hybrid_train_history_isolation_memory_probe_stage165_private.json"
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(run.private_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    visuals = memory_probe.write_stage165_memory_probe_visualizations(
        private_report=run.private_report,
        output_dir=(
            visualization_dir
            or artifact_dir / "primeqa_hybrid_train_history_isolation_memory_probe_stage165_visuals"
        ),
    )
    report = {
        **run.public_report,
        "private_artifacts_written": {
            "event_jsonl_path": str(event_path),
            "aggregate_json_path": str(private_path),
            "canonical_content_sha256": run.public_report["private_event_artifact_contract"][
                "canonical_content_sha256"
            ],
        },
        "visualizations": [{"name": visual.name, "path": visual.path} for visual in visuals],
    }
    output_path = output or (
        artifact_dir / "primeqa_hybrid_train_history_isolation_memory_probe_stage165.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage165 memory-probe public diagnostics: {output_path}")
    typer.echo(f"Saved Stage165 memory-probe private diagnostics: {private_path}")
    typer.echo(f"Saved Stage165 incremental memory events: {event_path}")
    if report.get("decision", {}).get("all_process_guards_passed") is not True:
        raise typer.Exit(code=1)


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "probe_contract": report.get("probe_contract"),
        "execution": report.get("execution"),
        "memory_diagnostics": report.get("memory_diagnostics"),
        "timing_seconds": report.get("timing_seconds"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
        "visualizations": report.get("visualizations"),
    }


if __name__ == "__main__":
    app()
