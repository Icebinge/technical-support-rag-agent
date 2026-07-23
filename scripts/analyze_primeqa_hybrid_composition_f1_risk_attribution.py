from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import (
    primeqa_hybrid_composition_f1_risk_attribution as analysis,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run the Stage183 train-only composition F1-risk attribution.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    report = analysis.run_stage183_composition_f1_risk_attribution(
        stage182_report_path=artifacts / "primeqa_hybrid_composition_dual_target_stage182.json",
        stage181_report_path=artifacts / "primeqa_hybrid_composition_action_audit_stage181.json",
        stage180_report_path=artifacts / "primeqa_hybrid_citation_aware_composition_stage180.json",
        stage179_report_path=artifacts
        / "primeqa_hybrid_listwise_agent_failure_attribution_stage179.json",
        stage178_public_path=artifacts / "primeqa_hybrid_listwise_agent_e2e_stage178a.json",
        stage178_private_path=artifacts
        / "primeqa_hybrid_listwise_agent_e2e_stage178a_private.json",
        stage178_alignment_path=artifacts / "stage178_candidate_alignment_audit.json",
        stage128_protocol_path=artifacts
        / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json",
        stage125_protocol_path=artifacts
        / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json",
        stage80_report_path=artifacts / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json",
        train_split_path=artifacts
        / "primeqa_hybrid_split_stage68_splits"
        / "primeqa_hybrid_split_stage68_train.jsonl",
        documents_path=settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        encoder_batch_size=encoder_batch_size,
        progress_sink=_progress,
    )
    visuals = analysis.write_stage183_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifacts / "primeqa_hybrid_composition_f1_risk_stage183_visuals",
    )
    report = {
        **report,
        "visualizations": [
            {"name": visualization.name, "path": visualization.path} for visualization in visuals
        ],
    }
    output_path = output or artifacts / "primeqa_hybrid_composition_f1_risk_stage183.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage183 report: {output_path}")
    if not report["decision"]["diagnostic_complete"]:
        raise typer.Exit(code=1)


def _progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    attribution = report["f1_risk_attribution"]
    return {
        "stage": report["stage"],
        "stage182_reproduction": report["stage182_reproduction"],
        "selected_action_summary": attribution["selected_action_summary"],
        "head_metric_ranges": attribution["head_metric_ranges"],
        "risk_calibration": attribution["risk_calibration"],
        "selected_regression_concentration": attribution["selected_regression_concentration"],
        "safe_alternative_headroom": attribution["safe_alternative_headroom"],
        "runtime_feature_separation": attribution["runtime_feature_separation"],
        "no_inner_eligible_fold_attribution": attribution["no_inner_eligible_fold_attribution"],
        "diagnostic_findings": attribution["diagnostic_findings"],
        "resource_consumption": report["resource_consumption"],
        "timing_seconds": report["timing_seconds"],
        "process_guards": report["process_guards"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
