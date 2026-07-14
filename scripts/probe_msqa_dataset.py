from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_schema_probe import (
    probe_msqa_dataset,
    write_msqa_schema_probe_visualizations,
)

app = typer.Typer(help="Probe the local MSQA dataset before held-out metrics.")


@app.command()
def main(
    msqa_csv: Annotated[
        Path,
        typer.Option("--msqa-csv", help="Local MSQA data/msqa-32k.csv path."),
    ],
    test_id_file: Annotated[
        Path,
        typer.Option("--test-id-file", help="Local MSQA data/test_id.txt path."),
    ],
    readme: Annotated[
        Path,
        typer.Option("--readme", help="Local MSQA README.md path."),
    ],
    repo_dir: Annotated[
        Path,
        typer.Option("--repo-dir", help="Local MSQA git clone directory."),
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
        typer.Option("--output", help="Output Stage 56 probe JSON path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum compact MSQA row samples to save."),
    ] = 3,
) -> None:
    """Write the Stage 56 MSQA schema/source-link probe report."""

    for path in [
        msqa_csv,
        test_id_file,
        readme,
        primeqa_train_questions,
        primeqa_dev_questions,
    ]:
        _ensure_file_exists(path)
    _ensure_directory_exists(repo_dir)

    report = probe_msqa_dataset(
        msqa_csv_path=msqa_csv,
        test_id_path=test_id_file,
        readme_path=readme,
        repository_head=_repository_head(repo_dir),
        primeqa_train_questions_path=primeqa_train_questions,
        primeqa_dev_questions_path=primeqa_dev_questions,
        sample_limit=sample_limit,
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_msqa_schema_probe_visualizations(
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
    typer.echo(f"Saved MSQA schema probe report: {output}")


def _repository_head(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": {
            "row_count": report["schema"]["row_count"],
            "readme_row_count_claim": report["schema"]["readme_row_count_claim"],
            "row_count_delta_vs_readme_claim": report["schema"][
                "row_count_delta_vs_readme_claim"
            ],
            "field_count": report["schema"]["field_count"],
            "duplicate_question_id_rows": report["schema"][
                "duplicate_question_id_rows"
            ],
        },
        "source_link_coverage": {
            "rows_with_row_url": report["source_link_coverage"]["rows_with_row_url"],
            "rows_with_answer_text_link": report["source_link_coverage"][
                "rows_with_answer_text_link"
            ],
            "rows_with_processed_answer_link": report["source_link_coverage"][
                "rows_with_processed_answer_link"
            ],
        },
        "test_id_file": report["test_id_file"],
        "primeqa_exact_leakage_precheck": report["primeqa_exact_leakage_precheck"],
        "readiness": report["readiness"],
        "visualizations": report["visualizations"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


def _ensure_directory_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise typer.BadParameter(f"Path is not a directory: {path}")


if __name__ == "__main__":
    app()
