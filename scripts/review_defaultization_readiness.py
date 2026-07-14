from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.defaultization_readiness_review import (
    SplitReviewSources,
    review_defaultization_readiness,
    write_defaultization_review_visualizations,
)

app = typer.Typer(
    help="Review dev/train readiness before a single held-out defaultization test."
)


@app.command()
def main(
    dev_topk_report: Annotated[
        Path,
        typer.Option("--dev-topk-report", help="Dev top-k baseline report."),
    ],
    dev_rank_contained_report: Annotated[
        Path,
        typer.Option(
            "--dev-rank-contained-report",
            help="Dev rank-contained reranker report.",
        ),
    ],
    dev_candidate_report: Annotated[
        Path,
        typer.Option("--dev-candidate-report", help="Dev Stage 51 candidate report."),
    ],
    dev_candidate_risk_report: Annotated[
        Path,
        typer.Option(
            "--dev-candidate-risk-report",
            help="Dev Stage 51 changed-answer risk report.",
        ),
    ],
    train_topk_report: Annotated[
        Path,
        typer.Option("--train-topk-report", help="Train top-k baseline report."),
    ],
    train_rank_contained_report: Annotated[
        Path,
        typer.Option(
            "--train-rank-contained-report",
            help="Train rank-contained reranker report.",
        ),
    ],
    train_candidate_report: Annotated[
        Path,
        typer.Option("--train-candidate-report", help="Train Stage 51 candidate report."),
    ],
    train_candidate_risk_report: Annotated[
        Path,
        typer.Option(
            "--train-candidate-risk-report",
            help="Train Stage 51 changed-answer risk report.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 52 readiness review JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write the Stage 52 defaultization readiness review."""

    paths = [
        dev_topk_report,
        dev_rank_contained_report,
        dev_candidate_report,
        dev_candidate_risk_report,
        train_topk_report,
        train_rank_contained_report,
        train_candidate_report,
        train_candidate_risk_report,
    ]
    for path in paths:
        _ensure_file_exists(path)

    review = review_defaultization_readiness(
        sources=[
            SplitReviewSources(
                split="dev",
                topk_report=_load_json(dev_topk_report),
                rank_contained_report=_load_json(dev_rank_contained_report),
                candidate_report=_load_json(dev_candidate_report),
                candidate_risk_report=_load_json(dev_candidate_risk_report),
            ),
            SplitReviewSources(
                split="train",
                topk_report=_load_json(train_topk_report),
                rank_contained_report=_load_json(train_rank_contained_report),
                candidate_report=_load_json(train_candidate_report),
                candidate_risk_report=_load_json(train_candidate_risk_report),
            ),
        ]
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_defaultization_review_visualizations(
            review=review,
            output_dir=visualization_dir,
        )

    report = {
        **review,
        "paths": {
            "dev_topk_report": str(dev_topk_report),
            "dev_rank_contained_report": str(dev_rank_contained_report),
            "dev_candidate_report": str(dev_candidate_report),
            "dev_candidate_risk_report": str(dev_candidate_risk_report),
            "train_topk_report": str(train_topk_report),
            "train_rank_contained_report": str(train_rank_contained_report),
            "train_candidate_report": str(train_candidate_report),
            "train_candidate_risk_report": str(train_candidate_risk_report),
        },
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved defaultization readiness review: {output}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_policy": report["candidate_policy"],
        "overall_decision": report["overall_decision"],
        "split_summary": [
            {
                "split": split_review["split"],
                "candidate_vs_topk": split_review["candidate_vs_topk"],
                "candidate_risk_summary": split_review["candidate_risk_summary"],
                "candidate_checks_passed": sum(
                    check["passed"]
                    for check in split_review["candidate_readiness_checks"]
                ),
                "candidate_checks_total": len(
                    split_review["candidate_readiness_checks"]
                ),
                "rank_contained_vs_topk": split_review["rank_contained_vs_topk"],
            }
            for split_review in report["split_reviews"]
        ],
        "heldout_protocol_status": report["heldout_test_protocol"]["status"],
        "visualizations": report["visualizations"],
    }


if __name__ == "__main__":
    app()
