from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_split_freeze import (
    freeze_primeqa_hybrid_split,
    write_primeqa_frozen_split_jsonl,
    write_primeqa_hybrid_split_freeze_visualizations,
)

app = typer.Typer(help="Freeze the Stage68 PrimeQA/TechQA hybrid split.")


@app.command()
def main(
    train_questions: Annotated[
        Path,
        typer.Option("--train-questions", help="PrimeQA training_Q_A.json path."),
    ],
    dev_questions: Annotated[
        Path,
        typer.Option("--dev-questions", help="PrimeQA dev_Q_A.json path."),
    ],
    validation_reference: Annotated[
        Path,
        typer.Option(
            "--validation-reference",
            help="PrimeQA validation_reference.json path.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage68 freeze report JSON path."),
    ],
    split_output_dir: Annotated[
        Path,
        typer.Option("--split-output-dir", help="Output directory for frozen JSONL files."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
    document_disjoint_answer_doc_ratio: Annotated[
        float,
        typer.Option(
            "--document-disjoint-answer-doc-ratio",
            help="Share of unique answer documents selected for strict test isolation.",
        ),
    ] = 0.10,
    remainder_train_ratio: Annotated[
        float,
        typer.Option("--remainder-train-ratio", help="Train ratio after doc isolation."),
    ] = 0.70,
    remainder_dev_ratio: Annotated[
        float,
        typer.Option("--remainder-dev-ratio", help="Dev ratio after doc isolation."),
    ] = 0.15,
    remainder_test_ratio: Annotated[
        float,
        typer.Option("--remainder-test-ratio", help="Random test ratio after doc isolation."),
    ] = 0.15,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic split seed."),
    ] = 20260714,
) -> None:
    """Write the Stage68 frozen hybrid split report and local split artifacts."""

    for path in [train_questions, dev_questions, validation_reference]:
        _ensure_file_exists(path)
    bundle = freeze_primeqa_hybrid_split(
        train_questions_path=train_questions,
        dev_questions_path=dev_questions,
        validation_reference_path=validation_reference,
        document_disjoint_answer_doc_ratio=document_disjoint_answer_doc_ratio,
        remainder_train_ratio=remainder_train_ratio,
        remainder_dev_ratio=remainder_dev_ratio,
        remainder_test_ratio=remainder_test_ratio,
        seed=seed,
    )
    split_artifacts = write_primeqa_frozen_split_jsonl(
        bundle=bundle,
        output_dir=split_output_dir,
    )
    report = {
        **bundle.report,
        "split_artifacts": split_artifacts,
    }
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_primeqa_hybrid_split_freeze_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid split freeze report: {output}")
    typer.echo(f"Saved frozen split JSONL files under: {split_output_dir}")


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "split_name": report["frozen_split"]["split_name"],
        "protocol_version": report["frozen_split"]["protocol_version"],
        "split_summary": report["frozen_split"]["split_summary"],
        "leakage_checks": report["leakage_checks"],
        "freeze_checks": report["freeze_checks"],
        "decision": report["decision"],
        "split_artifacts": report["split_artifacts"],
        "visualizations": report["visualizations"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


MappingForSummary = dict[str, Any]


if __name__ == "__main__":
    app()
