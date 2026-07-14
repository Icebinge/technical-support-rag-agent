from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_baseline_evaluation import (
    evaluate_msqa_topk_baseline,
    write_msqa_topk_baseline_visualizations,
)

app = typer.Typer(help="Evaluate MSQA frozen-split answer-source top-k baselines.")


@app.command()
def main(
    msqa_csv: Annotated[
        Path,
        typer.Option("--msqa-csv", help="Local MSQA data/msqa-32k.csv path."),
    ],
    split_jsonl: Annotated[
        Path,
        typer.Option("--split-jsonl", help="Frozen Stage 57 MSQA split JSONL."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 58 baseline JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
    top_k: Annotated[
        str,
        typer.Option("--top-k", help="Comma-separated top-k values, e.g. 1,3,5,10."),
    ] = "1,3,5,10",
    corpus_modes: Annotated[
        str,
        typer.Option(
            "--corpus-modes",
            help="Comma-separated corpus modes: answer_only,question_answer_page_text.",
        ),
    ] = "answer_only,question_answer_page_text",
    corpus_scope: Annotated[
        str,
        typer.Option(
            "--corpus-scope",
            help="Corpus scope: frozen_split_only or all_contract_rows.",
        ),
    ] = "frozen_split_only",
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum failure samples saved per class."),
    ] = 20,
) -> None:
    """Write Stage 58 MSQA top-k baseline report."""

    _ensure_file_exists(msqa_csv)
    _ensure_file_exists(split_jsonl)
    report = evaluate_msqa_topk_baseline(
        msqa_csv_path=msqa_csv,
        split_jsonl_path=split_jsonl,
        top_k_values=_parse_int_list(top_k, option_name="--top-k"),
        corpus_modes=_parse_str_list(corpus_modes, option_name="--corpus-modes"),
        corpus_scope=corpus_scope,
        sample_limit=sample_limit,
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_msqa_topk_baseline_visualizations(
            report=report,
            output_dir=visualization_dir,
        )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved MSQA top-k baseline report: {output}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": report["data"],
        "baseline_definition": report["baseline_definition"],
        "variants": [
            {
                "corpus_mode": variant["corpus_mode"],
                "retrieval_metrics": variant["retrieval_metrics"],
                "answer_metrics": variant["answer_metrics"],
                "failure_mode_counts": variant["failure_mode_counts"],
                "timing_seconds": variant["timing_seconds"],
            }
            for variant in report["variants"]
        ],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


def _parse_int_list(raw_value: str, option_name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must contain integers.") from exc
    if not values:
        raise typer.BadParameter(f"{option_name} must not be empty.")
    if any(value <= 0 for value in values):
        raise typer.BadParameter(f"{option_name} values must be positive.")
    return values


def _parse_str_list(raw_value: str, option_name: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not values:
        raise typer.BadParameter(f"{option_name} must not be empty.")
    return values


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


if __name__ == "__main__":
    app()
