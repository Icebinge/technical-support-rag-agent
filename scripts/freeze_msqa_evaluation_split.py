from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_evaluation_split import (
    build_msqa_evaluation_split_report,
    write_msqa_evaluation_split_visualizations,
    write_msqa_project_split_jsonl,
)

app = typer.Typer(help="Freeze the Stage 57 project-owned MSQA evaluation split.")


@app.command()
def main(
    msqa_csv: Annotated[
        Path,
        typer.Option("--msqa-csv", help="Local MSQA data/msqa-32k.csv path."),
    ],
    primeqa_train_questions: Annotated[
        Path,
        typer.Option("--primeqa-train-questions", help="PrimeQA training_Q_A.json."),
    ],
    primeqa_dev_questions: Annotated[
        Path,
        typer.Option("--primeqa-dev-questions", help="PrimeQA dev_Q_A.json."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 57 split report JSON path."),
    ],
    split_output: Annotated[
        Path | None,
        typer.Option("--split-output", help="Optional frozen split JSONL output path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
    near_duplicate_threshold: Annotated[
        float,
        typer.Option(
            "--near-duplicate-threshold",
            help="Token Jaccard threshold for non-exact near-duplicate detection.",
        ),
    ] = 0.9,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum leakage samples saved per class."),
    ] = 20,
) -> None:
    """Write Stage 57 MSQA adapter/leakage/split report."""

    for path in [msqa_csv, primeqa_train_questions, primeqa_dev_questions]:
        _ensure_file_exists(path)
    report = build_msqa_evaluation_split_report(
        msqa_csv_path=msqa_csv,
        primeqa_train_questions_path=primeqa_train_questions,
        primeqa_dev_questions_path=primeqa_dev_questions,
        near_duplicate_threshold=near_duplicate_threshold,
        sample_limit=sample_limit,
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_msqa_evaluation_split_visualizations(
            report=report,
            output_dir=visualization_dir,
        )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
        "split_output": str(split_output) if split_output is not None else None,
    }
    if split_output is not None:
        write_msqa_project_split_jsonl(
            report=report,
            msqa_csv_path=msqa_csv,
            output_path=split_output,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved MSQA evaluation split report: {output}")
    if split_output is not None:
        typer.echo(f"Saved frozen MSQA split JSONL: {split_output}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    leakage_counts = report["primeqa_leakage_audit"]["counts"]
    return {
        "adapter_contract": report["adapter_contract"],
        "leakage_counts": leakage_counts,
        "frozen_split": {
            "split_name": report["frozen_split"]["split_name"],
            "source_split_used": report["frozen_split"]["source_split_used"],
            "filter_counts": report["frozen_split"]["filter_counts"],
            "selected_question_ids_sha256": report["frozen_split"][
                "selected_question_ids_sha256"
            ],
            "first_selected_question_ids": report["frozen_split"][
                "first_selected_question_ids"
            ],
            "last_selected_question_ids": report["frozen_split"][
                "last_selected_question_ids"
            ],
        },
        "readiness": report["readiness"],
        "visualizations": report["visualizations"],
        "split_output": report["split_output"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


if __name__ == "__main__":
    app()
