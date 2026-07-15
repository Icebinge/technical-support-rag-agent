from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_validation_failure_pattern_analysis import (
    analyze_primeqa_hybrid_validation_failure_patterns,
    write_primeqa_hybrid_validation_failure_pattern_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage107 PrimeQA hybrid validation-failure pattern analysis."
)


@app.command()
def main(
    stage102_report: Annotated[
        Path | None,
        typer.Option("--stage102-report", help="Stage102 decomposition JSON."),
    ] = None,
    stage105_report: Annotated[
        Path | None,
        typer.Option("--stage105-report", help="Stage105 comparison JSON."),
    ] = None,
    stage106_decision: Annotated[
        Path | None,
        typer.Option("--stage106-decision", help="Stage106 stop decision JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage107 analysis JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_analysis: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-analysis/--no-user-confirmed-analysis",
            help="Required confirmation for Stage107 failure-pattern analysis.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write the Stage107 frozen public-safe validation-failure analysis."""

    settings = ProjectSettings()
    stage102_report_path = stage102_report or (
        settings.artifact_dir
        / "primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json"
    )
    stage105_report_path = stage105_report or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_comparison_stage105.json"
    )
    stage106_decision_path = stage106_decision or (
        settings.artifact_dir
        / "primeqa_hybrid_evidence_answerability_stop_decision_stage106.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_validation_failure_pattern_analysis_stage107.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals"
    )

    report = analyze_primeqa_hybrid_validation_failure_patterns(
        stage102_report_path=stage102_report_path,
        stage105_report_path=stage105_report_path,
        stage106_decision_path=stage106_decision_path,
        user_confirmed_analysis=user_confirmed_analysis,
        confirmation_note=confirmation_note,
    )
    visualizations = write_primeqa_hybrid_validation_failure_pattern_visualizations(
        report=report,
        output_dir=visualization_output_dir,
    )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid validation-failure analysis: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    pattern = report["pattern_summary"]
    return {
        "stage": report["stage"],
        "protocol_id": report["protocol_id"],
        "user_confirmation": report["user_confirmation"],
        "split_contract": report["split_contract"],
        "dev_failure_overview": pattern["dev_failure_overview"],
        "top_dev_failure_buckets": pattern["dev_bucket_failure_profile"][:5],
        "top_dev_route_failures": pattern["dev_route_failure_profile"][:5],
        "dev_retrieval_and_context_profile": pattern[
            "dev_retrieval_and_context_profile"
        ],
        "stage105_candidate_failure_pattern": {
            "selected_config": pattern["stage105_candidate_failure_pattern"][
                "selected_config"
            ],
            "selectable_config_count": pattern[
                "stage105_candidate_failure_pattern"
            ]["selectable_config_count"],
            "dev_better_nonselectable_config_count": pattern[
                "stage105_candidate_failure_pattern"
            ]["dev_better_nonselectable_config_count"],
            "train_guard_failure_reasons": pattern[
                "stage105_candidate_failure_pattern"
            ]["train_guard_failure_reasons"],
            "candidate_behavior_clusters": pattern[
                "stage105_candidate_failure_pattern"
            ]["candidate_behavior_clusters"],
        },
        "observed_failure_rules": pattern["observed_failure_rules"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
