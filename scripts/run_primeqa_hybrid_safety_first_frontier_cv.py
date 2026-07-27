from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_safety_first_frontier_cv as analysis
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Run the Stage 196 train-only safety-first frontier CV.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    visualization_dir: Annotated[Path | None, typer.Option("--visualization-dir")] = None,
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    dependency_dir = artifacts / "stage194_dependency"
    report = analysis.run_stage196_safety_first_frontier_cv(
        stage195_protocol_path=artifacts
        / "primeqa_hybrid_safety_first_frontier_protocol_stage195.json",
        lightgbm_wheel_path=dependency_dir / "lightgbm-4.7.0-py3-none-win_amd64.whl",
        narwhals_wheel_path=dependency_dir / "narwhals-2.24.0-py3-none-any.whl",
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
    visualizations = analysis.write_stage196_visualizations(
        report=report,
        output_dir=visualization_dir
        or artifacts / "primeqa_hybrid_safety_first_frontier_stage196_visuals",
    )
    report = {
        **report,
        "visualizations": [{"name": item.name, "path": item.path} for item in visualizations],
    }
    output_path = output or artifacts / "primeqa_hybrid_safety_first_frontier_stage196.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage 196 report: {output_path}")
    if not report["decision"]["experiment_valid"]:
        raise typer.Exit(code=1)


def _progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    cv = report["safety_first_frontier_nested_cv"]
    return {
        "stage": report["stage"],
        "dependency_preflight": report["dependency_preflight"],
        "resource_preflight": report["resource_preflight"],
        "stage182_reproduction": report["stage182_reproduction"],
        "dataset": cv["dataset"],
        "outer_folds": cv["outer_folds"],
        "aggregate": cv["aggregate"],
        "aggregate_diagnostics": cv["aggregate_diagnostics"],
        "paired_bootstrap": cv["paired_bootstrap"],
        "prediction_metrics": cv["prediction_metrics"],
        "selected_risk_weight_counts": cv["selected_risk_weight_counts"],
        "selected_prefix_counts": cv["selected_prefix_counts"],
        "advancement_gates": cv["advancement_gates"],
        "candidate_family_accepted": cv["candidate_family_accepted"],
        "execution": cv["execution"],
        "resource_consumption": report["resource_consumption"],
        "timing_seconds": report["timing_seconds"],
        "process_guards": report["process_guards"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
