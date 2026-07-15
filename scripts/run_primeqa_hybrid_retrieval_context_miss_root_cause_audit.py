from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_retrieval_context_miss_root_cause_audit import (
    run_primeqa_hybrid_retrieval_context_miss_root_cause_audit,
    write_primeqa_hybrid_retrieval_context_miss_root_cause_audit_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage112 PrimeQA hybrid retrieval-context-miss root-cause audit."
)


@app.command()
def main(
    stage111_protocol: Annotated[
        Path | None,
        typer.Option("--stage111-protocol", help="Stage111 frozen protocol JSON."),
    ] = None,
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train JSONL path."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Frozen Stage68 dev JSONL path."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training_dev_technotes.sections.json."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage112 audit JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_audit: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-audit/--no-user-confirmed-audit",
            help="Required confirmation for running the Stage112 audit.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
    sample_limit_per_bucket: Annotated[
        int,
        typer.Option(
            "--sample-limit-per-bucket",
            help="Public-safe sample rows kept per split and root-cause bucket.",
        ),
    ] = 5,
) -> None:
    """Run Stage112 and write the report plus SVG visualizations."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    stage111_protocol_path = stage111_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111.json"
    )
    train_split_path = train_split or split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    dev_split_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals"
    )

    report = run_primeqa_hybrid_retrieval_context_miss_root_cause_audit(
        stage111_protocol_path=stage111_protocol_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        user_confirmed_audit=user_confirmed_audit,
        confirmation_note=confirmation_note,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    visualizations = (
        write_primeqa_hybrid_retrieval_context_miss_root_cause_audit_visualizations(
            report=report,
            output_dir=visualization_output_dir,
        )
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
    typer.echo(f"Saved PrimeQA hybrid Stage112 root-cause audit: {output_path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "user_confirmation": report["user_confirmation"],
        "stage111_summary": report["stage111_summary"],
        "audit_config": report["audit_config"],
        "loaded_data_summary": report["loaded_data_summary"],
        "split_summary": {
            split: {
                "answerable_rows": split_report["answerable_rows"],
                "audit_case_count": split_report["audit_case_count"],
                "audit_case_rate_among_answerable": split_report[
                    "audit_case_rate_among_answerable"
                ],
                "top_primary_root_causes": sorted(
                    split_report["primary_root_cause_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )[:6],
                "gold_doc_rank_bucket_counts": split_report[
                    "gold_doc_rank_bucket_counts"
                ],
                "dimension_high_signal_counts": split_report[
                    "dimension_high_signal_counts"
                ],
            }
            for split, split_report in report["split_reports"].items()
        },
        "cross_split_summary": report["cross_split_summary"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
