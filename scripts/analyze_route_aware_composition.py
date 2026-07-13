from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.route_aware_composition_policy import (
    RouteAwareCompositionPolicy,
    analyze_route_aware_composition_policy,
    route_aware_composition_result_to_dict,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Analyze a route-aware answer-composition policy.")


@app.command()
def main(
    answer_gap_report: Annotated[
        Path,
        typer.Option("--answer-gap-report", help="Input full answer-gap JSON report."),
    ],
    strong_first_score_min: Annotated[
        float,
        typer.Option("--strong-first-score-min", help="Minimum first-candidate score."),
    ] = 100.0,
    strong_first_score_ratio_min: Annotated[
        float,
        typer.Option(
            "--strong-first-score-ratio-min",
            help="Minimum first/second score ratio for top1 routing.",
        ),
    ] = 1.15,
    strong_first_score_margin_min: Annotated[
        float,
        typer.Option(
            "--strong-first-score-margin-min",
            help="Minimum first-minus-second score margin for top1 routing.",
        ),
    ] = 20.0,
    max_top1_retrieval_rank: Annotated[
        int,
        typer.Option("--max-top1-retrieval-rank", help="Worst retrieval rank allowed for top1."),
    ] = 3,
    duplicate_threshold: Annotated[
        float,
        typer.Option("--duplicate-threshold", help="Same-document near-duplicate threshold."),
    ] = 0.96,
    min_average_f1_gain: Annotated[
        float,
        typer.Option("--min-average-f1-gain", help="F1 gain required for acceptance."),
    ] = 0.002,
    max_allowed_citation_loss: Annotated[
        int,
        typer.Option("--max-allowed-citation-loss", help="Maximum allowed citation loss."),
    ] = 2,
    sample_limit_per_bucket: Annotated[
        int,
        typer.Option("--sample-limit-per-bucket", help="Maximum saved cases per bucket."),
    ] = 20,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON report path."),
    ] = None,
) -> None:
    """Evaluate route-aware composition over a full answer-gap report."""

    _validate_options(
        answer_gap_report=answer_gap_report,
        strong_first_score_min=strong_first_score_min,
        strong_first_score_ratio_min=strong_first_score_ratio_min,
        strong_first_score_margin_min=strong_first_score_margin_min,
        max_top1_retrieval_rank=max_top1_retrieval_rank,
        duplicate_threshold=duplicate_threshold,
        min_average_f1_gain=min_average_f1_gain,
        max_allowed_citation_loss=max_allowed_citation_loss,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / f"route_aware_composition_policy_{answer_gap_report.stem}.json"
    )
    report = json.loads(answer_gap_report.read_text(encoding="utf-8"))
    policy = RouteAwareCompositionPolicy(
        strong_first_score_min=strong_first_score_min,
        strong_first_score_ratio_min=strong_first_score_ratio_min,
        strong_first_score_margin_min=strong_first_score_margin_min,
        max_top1_retrieval_rank=max_top1_retrieval_rank,
        duplicate_threshold=duplicate_threshold,
    )
    result = analyze_route_aware_composition_policy(
        answer_gap_report=report,
        policy=policy,
        min_average_f1_gain=min_average_f1_gain,
        max_allowed_citation_loss=max_allowed_citation_loss,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    result_dict = route_aware_composition_result_to_dict(result)
    result_dict["policy_config"] = {
        "strong_first_score_min": strong_first_score_min,
        "strong_first_score_ratio_min": strong_first_score_ratio_min,
        "strong_first_score_margin_min": strong_first_score_margin_min,
        "max_top1_retrieval_rank": max_top1_retrieval_rank,
        "duplicate_threshold": duplicate_threshold,
        "min_average_f1_gain": min_average_f1_gain,
        "max_allowed_citation_loss": max_allowed_citation_loss,
        "sample_limit_per_bucket": sample_limit_per_bucket,
    }
    result_dict["source_report"] = str(answer_gap_report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(json.dumps(result_dict["summary"], ensure_ascii=False, indent=2))
    typer.echo(f"Saved route-aware composition policy report: {output_path}")


def _validate_options(
    answer_gap_report: Path,
    strong_first_score_min: float,
    strong_first_score_ratio_min: float,
    strong_first_score_margin_min: float,
    max_top1_retrieval_rank: int,
    duplicate_threshold: float,
    min_average_f1_gain: float,
    max_allowed_citation_loss: int,
    sample_limit_per_bucket: int,
) -> None:
    if not answer_gap_report.exists():
        raise typer.BadParameter(f"Missing answer gap report: {answer_gap_report}")
    if strong_first_score_min < 0:
        raise typer.BadParameter("--strong-first-score-min must be non-negative.")
    if strong_first_score_ratio_min < 1:
        raise typer.BadParameter("--strong-first-score-ratio-min must be at least 1.")
    if strong_first_score_margin_min < 0:
        raise typer.BadParameter("--strong-first-score-margin-min must be non-negative.")
    if max_top1_retrieval_rank <= 0:
        raise typer.BadParameter("--max-top1-retrieval-rank must be positive.")
    if not 0 <= duplicate_threshold <= 1:
        raise typer.BadParameter("--duplicate-threshold must be between 0 and 1.")
    if min_average_f1_gain < 0:
        raise typer.BadParameter("--min-average-f1-gain must be non-negative.")
    if max_allowed_citation_loss < 0:
        raise typer.BadParameter("--max-allowed-citation-loss must be non-negative.")
    if sample_limit_per_bucket < 0:
        raise typer.BadParameter("--sample-limit-per-bucket must be non-negative.")


if __name__ == "__main__":
    app()
