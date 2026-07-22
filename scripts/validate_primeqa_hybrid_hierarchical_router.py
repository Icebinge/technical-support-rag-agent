from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_hierarchical_router_validation as validation
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage171 expanded train-only hierarchical-router validation.")


@app.command()
def main(
    model_snapshot: Annotated[Path, typer.Option("--model-snapshot")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
) -> None:
    """Validate the frozen hierarchy without accepting dev or test paths."""

    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    split_dir = artifacts / "primeqa_hybrid_split_stage68_splits"
    report = validation.run_stage171_hierarchical_validation(
        stage170_report_path=artifacts
        / "primeqa_hybrid_iterative_router_prompt_comparison_stage170.json",
        stage80_report_path=artifacts / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json",
        train_split_path=split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        documents_path=settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        model_snapshot_path=model_snapshot,
        encoder_batch_size=encoder_batch_size,
        progress_sink=_write_progress,
    )
    visual_dir = visualization_dir or (
        artifacts / "primeqa_hybrid_hierarchical_router_validation_stage171_visuals"
    )
    visualizations = validation.write_stage171_visualizations(
        report=report,
        output_dir=visual_dir,
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path}
            for visualization in visualizations
        ],
    }
    output_path = output or (
        artifacts / "primeqa_hybrid_hierarchical_router_validation_stage171.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage171 hierarchical validation: {output_path}")
    if report["decision"]["all_process_guards_passed"] is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "synthetic_calibration": {
            key: value for key, value in report["synthetic_calibration"].items() if key != "phases"
        },
        "quality_metrics": report["quality_metrics"],
        "legacy_quality_gates": report["legacy_quality_gates"],
        "hierarchy_quality_gates": report["hierarchy_quality_gates"],
        "fold_stability": report["fold_stability"],
        "resource_consumption": report["resource_consumption"],
        "timing_seconds": report["timing_seconds"],
        "process_guards": report["process_guards"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
