from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    DEFAULT_BLOCKED_ROUTE_SETS,
    DEFAULT_MAX_SELECTED_RANK_GRID,
    DEFAULT_MIN_SCORE_MARGIN_GRID,
    DEFAULT_PROTECT_TOP1_CANDIDATE_SCORE_MIN_GRID,
    candidate_reranker_policy_search_to_dict,
    search_candidate_reranker_policies,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Search constrained offline candidate-reranker policies.")

DEFAULT_PROTECT_TOP1_CANDIDATE_SCORE_MIN_GRID_TEXT = ",".join(
    "none" if value is None else str(value)
    for value in DEFAULT_PROTECT_TOP1_CANDIDATE_SCORE_MIN_GRID
)
DEFAULT_BLOCKED_ROUTE_SETS_TEXT = ";".join(
    "none" if not route_set else ",".join(route_set)
    for route_set in DEFAULT_BLOCKED_ROUTE_SETS
)


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Candidate reranker model name."),
    ] = "logistic_best_candidate",
    fold_count: Annotated[
        int,
        typer.Option("--fold-count", help="Number of deterministic question folds."),
    ] = 5,
    max_selected_rank_grid: Annotated[
        str,
        typer.Option("--max-selected-rank-grid", help="Comma-separated max rank values."),
    ] = ",".join(str(value) for value in DEFAULT_MAX_SELECTED_RANK_GRID),
    min_score_margin_grid: Annotated[
        str,
        typer.Option("--min-score-margin-grid", help="Comma-separated model margin values."),
    ] = ",".join(str(value) for value in DEFAULT_MIN_SCORE_MARGIN_GRID),
    protect_top1_candidate_score_min_grid: Annotated[
        str,
        typer.Option(
            "--protect-top1-candidate-score-min-grid",
            help="Comma-separated thresholds or 'none'.",
        ),
    ] = DEFAULT_PROTECT_TOP1_CANDIDATE_SCORE_MIN_GRID_TEXT,
    blocked_route_sets: Annotated[
        str,
        typer.Option(
            "--blocked-route-sets",
            help="Semicolon-separated route sets; routes inside a set are comma-separated.",
        ),
    ] = DEFAULT_BLOCKED_ROUTE_SETS_TEXT,
    deep_rank_min: Annotated[
        int,
        typer.Option("--deep-rank-min", help="Selected rank threshold for deep-rank cases."),
    ] = 6,
    top_policy_limit: Annotated[
        int,
        typer.Option("--top-policy-limit", help="Number of ranked policies retained."),
    ] = 25,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output policy-search JSON path."),
    ] = None,
) -> None:
    """Run constrained offline policy search over grouped-CV reranker selections."""

    _ensure_file_exists(dataset)
    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir / f"candidate_reranker_policy_search_{dataset.stem}.json"
    )

    rows = load_candidate_reranker_rows(dataset)
    result = search_candidate_reranker_policies(
        rows=rows,
        model_name=model,
        fold_count=fold_count,
        max_selected_rank_grid=_parse_int_grid(max_selected_rank_grid),
        min_score_margin_grid=_parse_float_grid(min_score_margin_grid),
        protect_top1_candidate_score_min_grid=_parse_optional_float_grid(
            protect_top1_candidate_score_min_grid
        ),
        blocked_route_sets=_parse_route_sets(blocked_route_sets),
        deep_rank_min=deep_rank_min,
        top_policy_limit=top_policy_limit,
    )
    result_dict = candidate_reranker_policy_search_to_dict(result)
    result_dict["source_paths"] = {"dataset": str(dataset)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_reduction = result.best_regression_reduction_policy
    typer.echo(
        json.dumps(
            {
                "model_name": result.model_name,
                "fold_count": result.fold_count,
                "policy_count": result.policy_count,
                "unconstrained": _metrics_summary(result.unconstrained_metrics),
                "best_average_delta_policy": {
                    "name": result.best_average_delta_policy.config.name,
                    **_metrics_summary(result.best_average_delta_policy.metrics),
                },
                "best_regression_reduction_policy": (
                    {
                        "name": best_reduction.config.name,
                        **_metrics_summary(best_reduction.metrics),
                    }
                    if best_reduction
                    else None
                ),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _metrics_summary(metrics) -> dict:
    return {
        "policy_average_token_f1": metrics.policy_average_token_f1,
        "average_delta_vs_top_candidate": metrics.average_delta_vs_top_candidate,
        "oracle_gap_closed_rate": metrics.oracle_gap_closed_rate,
        "replacement_count": metrics.replacement_count,
        "regressed_count": metrics.regressed_count,
        "regression_reduction_vs_unconstrained": (
            metrics.regression_reduction_vs_unconstrained
        ),
        "final_missed_gold_document_count": metrics.final_missed_gold_document_count,
        "final_deep_rank_count": metrics.final_deep_rank_count,
    }


def _parse_int_grid(raw_grid: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw_grid.split(",") if value.strip()]
    except ValueError as exc:
        raise typer.BadParameter("grid must contain only integers") from exc
    if not values:
        raise typer.BadParameter("grid must not be empty")
    return values


def _parse_float_grid(raw_grid: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw_grid.split(",") if value.strip()]
    except ValueError as exc:
        raise typer.BadParameter("grid must contain only numbers") from exc
    if not values:
        raise typer.BadParameter("grid must not be empty")
    return values


def _parse_optional_float_grid(raw_grid: str) -> list[float | None]:
    values = []
    for raw_value in raw_grid.split(","):
        value = raw_value.strip().lower()
        if not value:
            continue
        if value == "none":
            values.append(None)
            continue
        try:
            values.append(float(value))
        except ValueError as exc:
            raise typer.BadParameter(
                "optional float grid must contain numbers or 'none'"
            ) from exc
    if not values:
        raise typer.BadParameter("optional float grid must not be empty")
    return values


def _parse_route_sets(raw_route_sets: str) -> list[tuple[str, ...]]:
    route_sets = []
    for raw_route_set in raw_route_sets.split(";"):
        routes = tuple(
            route.strip()
            for route in raw_route_set.split(",")
            if route.strip() and route.strip().lower() != "none"
        )
        route_sets.append(routes)
    if not route_sets:
        raise typer.BadParameter("route set grid must not be empty")
    return route_sets


def _format_optional_float_grid(values: tuple[float | None, ...]) -> str:
    return ",".join("none" if value is None else str(value) for value in values)


def _format_route_sets(route_sets: tuple[tuple[str, ...], ...]) -> str:
    return ";".join("none" if not route_set else ",".join(route_set) for route_set in route_sets)


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
