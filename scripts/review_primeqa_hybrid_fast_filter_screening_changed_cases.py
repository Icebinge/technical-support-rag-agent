from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_fast_filter_screening_changed_case_review import (
    review_primeqa_hybrid_fast_filter_screening_changed_cases,
    write_primeqa_hybrid_fast_filter_screening_changed_case_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Run Stage122 PrimeQA hybrid fast-filter screening changed-case review."
)


@app.command()
def main(
    stage121_validation: Annotated[
        Path | None,
        typer.Option("--stage121-validation", help="Stage121 validation JSON."),
    ] = None,
    stage120_protocol: Annotated[
        Path | None,
        typer.Option("--stage120-protocol", help="Stage120 frozen protocol JSON."),
    ] = None,
    stage80_report: Annotated[
        Path | None,
        typer.Option("--stage80-report", help="Stage80 dense cache feasibility JSON."),
    ] = None,
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train split JSONL."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Frozen Stage68 dev split JSONL."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training_dev_technotes.sections.json."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage122 review JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
    user_confirmed_review: Annotated[
        bool,
        typer.Option(
            "--user-confirmed-review/--no-user-confirmed-review",
            help="Required confirmation for Stage122 changed-case review.",
        ),
    ] = False,
    confirmation_note: Annotated[
        str,
        typer.Option("--confirmation-note", help="Short factual confirmation note."),
    ] = "not confirmed",
    configs: Annotated[
        str,
        typer.Option("--configs", help="Comma-separated config IDs to review."),
    ] = (
        "special_token_exact_window40_rule_selector_v1,"
        "top10_locked_route_vote_window50_pairwise_logistic_v1"
    ),
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum public-safe samples per split/config."),
    ] = 30,
    include_dense_channels: Annotated[
        bool,
        typer.Option(
            "--include-dense-channels/--no-include-dense-channels",
            help="Use only existing local dense caches when enabled.",
        ),
    ] = True,
    encoder_batch_size: Annotated[
        int,
        typer.Option("--encoder-batch-size", help="SentenceTransformer query batch size."),
    ] = 64,
    encoder_device: Annotated[
        str | None,
        typer.Option("--encoder-device", help="Optional SentenceTransformer device."),
    ] = None,
) -> None:
    """Write the Stage122 train/dev changed-case review report."""

    settings = ProjectSettings()
    split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    stage121_validation_path = stage121_validation or (
        settings.artifact_dir
        / "primeqa_hybrid_fast_filter_screening_validation_stage121.json"
    )
    stage120_protocol_path = stage120_protocol or (
        settings.artifact_dir
        / "primeqa_hybrid_fast_filter_screening_protocol_stage120.json"
    )
    stage80_report_path = stage80_report or (
        settings.artifact_dir / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json"
    )
    train_split_path = train_split or (
        split_dir / "primeqa_hybrid_split_stage68_train.jsonl"
    )
    dev_split_path = dev_split or split_dir / "primeqa_hybrid_split_stage68_dev.jsonl"
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    output_path = output or (
        settings.artifact_dir
        / "primeqa_hybrid_fast_filter_screening_changed_case_review_stage122.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir
        / "primeqa_hybrid_fast_filter_screening_changed_case_review_stage122_visuals"
    )

    report = review_primeqa_hybrid_fast_filter_screening_changed_cases(
        stage121_validation_path=stage121_validation_path,
        stage120_protocol_path=stage120_protocol_path,
        train_split_path=train_split_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        stage80_report_path=stage80_report_path,
        user_confirmed_review=user_confirmed_review,
        confirmation_note=confirmation_note,
        config_ids=_parse_configs(configs),
        sample_limit=sample_limit,
        include_dense_channels=include_dense_channels,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
    )
    visualizations = write_primeqa_hybrid_fast_filter_screening_changed_case_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid Stage122 changed-case review: {output_path}")


def _parse_configs(raw_configs: str) -> tuple[str, ...]:
    config_ids = tuple(config.strip() for config in raw_configs.split(",") if config.strip())
    if not config_ids:
        raise typer.BadParameter("--configs must not be empty.")
    return config_ids


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "analysis_id": report["analysis_id"],
        "review_config_ids": report["analysis_config"]["review_config_ids"],
        "config_reviews": [
            {
                "config_id": review["config_id"],
                "interpretation": review["interpretation"],
                "train_cv": {
                    key: review["split_reviews"]["train_cv"][key]
                    for key in (
                        "changed_case_count",
                        "improved_count",
                        "regressed_count",
                        "hit20_recovery_count",
                        "hit20_regression_count",
                    )
                },
                "dev": {
                    key: review["split_reviews"]["dev"][key]
                    for key in (
                        "changed_case_count",
                        "improved_count",
                        "regressed_count",
                        "hit20_recovery_count",
                        "hit20_regression_count",
                    )
                },
            }
            for review in report["config_reviews"]
        ],
        "cross_config_findings": report["cross_config_findings"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "public_safe_contract": report["public_safe_contract"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


if __name__ == "__main__":
    app()
