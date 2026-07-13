from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.route_aware_composition_cv import (
    DEFAULT_INSTALL_MARGIN_GRID,
    cross_validate_route_aware_composition_policy,
    route_aware_cv_result_to_dict,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Cross-validate route-aware composition thresholds.")


@app.command()
def main(
    answer_gap_report: Annotated[
        Path,
        typer.Option("--answer-gap-report", help="Input full answer-gap JSON report."),
    ],
    fold_count: Annotated[
        int,
        typer.Option("--fold-count", help="Number of deterministic CV folds."),
    ] = 5,
    install_upgrade_score_margin_grid: Annotated[
        str,
        typer.Option(
            "--install-upgrade-score-margin-grid",
            help="Comma-separated install/upgrade/config margin candidates.",
        ),
    ] = ",".join(str(value) for value in DEFAULT_INSTALL_MARGIN_GRID),
    min_train_average_f1_gain: Annotated[
        float,
        typer.Option(
            "--min-train-average-f1-gain",
            help="Minimum train-fold F1 gain required for threshold selection.",
        ),
    ] = 0.0,
    min_train_citation_delta: Annotated[
        int,
        typer.Option(
            "--min-train-citation-delta",
            help="Minimum train-fold citation delta required for threshold selection.",
        ),
    ] = 0,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON report path."),
    ] = None,
) -> None:
    """Run deterministic k-fold threshold cross-validation."""

    _validate_options(
        answer_gap_report=answer_gap_report,
        fold_count=fold_count,
        min_train_average_f1_gain=min_train_average_f1_gain,
    )
    margins = _parse_margin_grid(install_upgrade_score_margin_grid)

    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir
        / f"route_aware_composition_cv_{answer_gap_report.stem}.json"
    )
    report = json.loads(answer_gap_report.read_text(encoding="utf-8"))
    result = cross_validate_route_aware_composition_policy(
        answer_gap_report=report,
        install_upgrade_score_margin_grid=margins,
        fold_count=fold_count,
        min_train_average_f1_gain=min_train_average_f1_gain,
        min_train_citation_delta=min_train_citation_delta,
    )
    result_dict = route_aware_cv_result_to_dict(result)
    result_dict["source_report"] = str(answer_gap_report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(json.dumps(result_dict["aggregate_validation"], ensure_ascii=False, indent=2))
    typer.echo(f"Saved route-aware composition CV report: {output_path}")


def _parse_margin_grid(raw_grid: str) -> list[float]:
    try:
        margins = [
            float(raw_margin.strip())
            for raw_margin in raw_grid.split(",")
            if raw_margin.strip()
        ]
    except ValueError as exc:
        raise typer.BadParameter(
            "--install-upgrade-score-margin-grid must contain only numbers."
        ) from exc

    if not margins:
        raise typer.BadParameter("--install-upgrade-score-margin-grid must not be empty.")
    if any(margin < 0 for margin in margins):
        raise typer.BadParameter(
            "--install-upgrade-score-margin-grid values must be non-negative."
        )
    return margins


def _validate_options(
    answer_gap_report: Path,
    fold_count: int,
    min_train_average_f1_gain: float,
) -> None:
    if not answer_gap_report.exists():
        raise typer.BadParameter(f"Missing answer gap report: {answer_gap_report}")
    if fold_count < 2:
        raise typer.BadParameter("--fold-count must be at least 2.")
    if min_train_average_f1_gain < 0:
        raise typer.BadParameter("--min-train-average-f1-gain must be non-negative.")


if __name__ == "__main__":
    app()
