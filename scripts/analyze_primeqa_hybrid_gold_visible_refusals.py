from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application import primeqa_hybrid_gold_visible_refusal_diagnostics as stage164
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Analyze Stage160 gold-document-visible Agent refusals.")


@app.command()
def main(
    stage163_correction: Annotated[
        Path | None,
        typer.Option("--stage163-correction", help="Completed Stage163 correction report."),
    ] = None,
    stage160_report: Annotated[
        Path | None,
        typer.Option("--stage160-report", help="Completed Stage160 public report."),
    ] = None,
    stage160_hashed_report: Annotated[
        Path | None,
        typer.Option("--stage160-hashed-report", help="Stage160 ignored hashed diagnostics."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Exact frozen Stage68 development JSONL."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training/dev technote sections JSON."),
    ] = None,
    router_source: Annotated[
        Path | None,
        typer.Option("--router-source", help="Exact structured-router source used by Stage160."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage164 public aggregate JSON."),
    ] = None,
    private_output: Annotated[
        Path | None,
        typer.Option("--private-output", help="Ignored hashed Stage164 feature JSON."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for Stage164 SVGs."),
    ] = None,
    user_confirmed_diagnostics: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-diagnostics/--no-user-confirmed-diagnostics",
            help="Required confirmation for Stage164 diagnostic-only analysis.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual user-confirmation note."),
    ] = "not confirmed",
) -> None:
    """Write Stage164 aggregate and ignored hashed refusal diagnostics."""

    settings = ProjectSettings()
    artifact_dir = settings.artifact_dir
    split_dir = artifact_dir / "primeqa_hybrid_split_stage68_splits"
    run = stage164.run_primeqa_hybrid_gold_visible_refusal_diagnostics(
        stage163_correction_path=stage163_correction
        or artifact_dir / "primeqa_hybrid_untouched_rrf_dev_contract_correction_stage163.json",
        stage160_report_path=stage160_report
        or artifact_dir / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160.json",
        stage160_hashed_report_path=stage160_hashed_report
        or artifact_dir
        / "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_stage160_private.json",
        dev_split_path=dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl",
        documents_path=documents
        or settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        router_source_path=router_source
        or Path("src/ts_rag_agent/application/primeqa_hybrid_structured_decision_router.py"),
        user_confirmed_diagnostics=user_confirmed_diagnostics,
        confirmation_note=confirmation_note,
        progress_sink=_write_progress,
    )

    private_path = private_output or (
        artifact_dir / "primeqa_hybrid_gold_visible_refusal_diagnostics_stage164_private.json"
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(run.private_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    report = {
        **run.public_report,
        "private_feature_artifact_written": {
            "path": str(private_path),
            "byte_sha256": stage164.private_report_byte_sha256(private_path),
            "canonical_content_sha256": stage164.private_report_canonical_sha256(
                run.private_report
            ),
        },
    }
    visual_dir = visualization_dir or (
        artifact_dir / "primeqa_hybrid_gold_visible_refusal_diagnostics_stage164_visuals"
    )
    visualizations = stage164.write_stage164_visualizations(
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
        artifact_dir / "primeqa_hybrid_gold_visible_refusal_diagnostics_stage164.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage164 public diagnostics: {output_path}")
    typer.echo(f"Saved Stage164 private hashed features: {private_path}")
    if report["decision"]["all_process_guards_passed"] is not True:
        raise typer.Exit(code=1)


def _write_progress(event: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(dict(event), ensure_ascii=True, separators=(",", ":")))


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "split_contract": report["split_contract"],
        "router_prompt_contract": report["router_prompt_contract"],
        "cohort_summary": report["cohort_summary"],
        "answer_visibility_summary": report["answer_visibility_summary"],
        "fixed_binary_associations": report["fixed_binary_associations"],
        "fixed_numeric_associations": report["fixed_numeric_associations"],
        "question_route_summary": report["question_route_summary"],
        "fold_stability": report["fold_stability"],
        "primary_hypothesis_assessment": report["primary_hypothesis_assessment"],
        "guard_checks": report["guard_checks"],
        "public_safe_contract": report["public_safe_contract"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
