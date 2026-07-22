from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as analysis
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run Stage172 train-only evidence-entailment nested CV.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
) -> None:
    """Cross-validate the frozen classifier without accepting dev or test paths."""

    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    split_dir = artifacts / "primeqa_hybrid_split_stage68_splits"
    report = analysis.run_stage172_evidence_entailment_cv(
        stage171_report_path=artifacts
        / "primeqa_hybrid_hierarchical_router_validation_stage171.json",
        stage80_report_path=artifacts / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json",
        train_split_path=split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        documents_path=settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        encoder_batch_size=encoder_batch_size,
        progress_sink=_write_progress,
    )
    visual_dir = visualization_dir or (
        artifacts / "primeqa_hybrid_evidence_entailment_cv_stage172_visuals"
    )
    visualizations = analysis.write_stage172_visualizations(
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
    output_path = output or (artifacts / "primeqa_hybrid_evidence_entailment_cv_stage172.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage172 evidence-entailment nested CV: {output_path}")
    if report["decision"]["all_process_guards_passed"] is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "case_summary": report["case_summary"],
        "nested_cv": report["nested_cv"],
        "stage171_comparison": report["stage171_comparison"],
        "resource_consumption": report["resource_consumption"],
        "timing_seconds": report["timing_seconds"],
        "process_guards": report["process_guards"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
