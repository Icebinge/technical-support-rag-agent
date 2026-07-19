from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_sharded_validation as sharded,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run one exact Stage165 process-isolated train shard.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    shard_ordinal: Annotated[
        int,
        typer.Option("--shard-ordinal", min=1, max=12, help="Frozen shard ordinal."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Shard public aggregate JSON."),
    ],
    observation_jsonl: Annotated[
        Path,
        typer.Option(
            "--observation-jsonl",
            help="Ignored incrementally flushed content-free observations.",
        ),
    ],
    user_confirmed_stage165_shard: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-stage165-shard",
            help="Required confirmation from the parent 12-shard protocol.",
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
) -> None:
    if not user_confirmed_stage165_shard:
        raise typer.BadParameter("Stage165 shard requires --user-confirmed-stage165-shard")
    settings = ProjectSettings(
        enable_bounded_dynamic_agent_runtime=True,
        enable_bounded_dynamic_agent_http_transport=True,
        bounded_dynamic_agent_model_snapshot=model_snapshot,
    )
    artifact_dir = settings.artifact_dir.resolve()
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    run = sharded.execute_stage165_shard(
        settings=settings,
        stage164_correction_path=stage164_correction
        or artifact_dir / "primeqa_hybrid_gold_visible_refusal_contract_stage164.json",
        train_split_path=train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        shard_ordinal=shard_ordinal,
        observation_jsonl_path=observation_jsonl,
        user_confirmed_shard_execution=True,
        progress_sink=_write_progress,
    )
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run.public_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(run.public_report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved Stage165 shard diagnostics: {output_path}")
    if run.public_report["decision"]["all_process_guards_passed"] is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "shard": report.get("shard"),
        "execution": report.get("execution"),
        "timing_seconds": report.get("timing_seconds"),
        "guard_checks": report.get("guard_checks"),
        "decision": report.get("decision"),
    }


if __name__ == "__main__":
    app()
