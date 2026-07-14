from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_split_plan import (
    plan_primeqa_hybrid_split,
    write_primeqa_hybrid_split_assignments,
    write_primeqa_hybrid_split_visualizations,
)

app = typer.Typer(help="Plan the Stage67 PrimeQA/TechQA hybrid split dry-run.")


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
        typer.Option("--output", help="Output Stage67 split plan JSON path."),
    ],
    assignments_output: Annotated[
        Path | None,
        typer.Option(
            "--assignments-output",
            help="Optional JSONL row-assignment artifact without raw text.",
        ),
    ] = None,
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
    """Write the Stage67 hybrid split dry-run report."""

    report = plan_primeqa_hybrid_split(
        train_questions_path=train_questions,
        dev_questions_path=dev_questions,
        validation_reference_path=validation_reference,
        document_disjoint_answer_doc_ratio=document_disjoint_answer_doc_ratio,
        remainder_train_ratio=remainder_train_ratio,
        remainder_dev_ratio=remainder_dev_ratio,
        remainder_test_ratio=remainder_test_ratio,
        seed=seed,
    )
    visualizations = []
    if assignments_output is not None:
        write_primeqa_hybrid_split_assignments(
            report=report,
            output_path=assignments_output,
        )
        report = {
            **report,
            "assignments_artifact": {
                "path": str(assignments_output),
                "row_count": len(report["assignments"]),
                "contains_raw_text": False,
            },
        }
    if visualization_dir is not None:
        visualizations = write_primeqa_hybrid_split_visualizations(
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
    typer.echo(f"Saved PrimeQA hybrid split dry-run report: {output}")


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "split_protocol": {
            "seed": report["split_protocol"]["seed"],
            "document_disjoint_answer_doc_ratio": report["split_protocol"][
                "document_disjoint_answer_doc_ratio"
            ],
            "remainder_split_ratios": report["split_protocol"][
                "remainder_split_ratios"
            ],
        },
        "input_summary": {
            "row_count": report["input_summary"]["row_count"],
            "group_count": report["input_summary"]["group_count"],
            "duplicate_group_count": report["input_summary"]["duplicate_group_count"],
            "answerable_count": report["input_summary"]["answerable_count"],
            "unanswerable_count": report["input_summary"]["unanswerable_count"],
        },
        "document_disjoint_summary": report["document_disjoint_summary"],
        "split_summary": report["split_summary"],
        "leakage_checks": report["leakage_checks"],
        "decision": report["decision"],
        "assignments_artifact": report.get("assignments_artifact"),
        "visualizations": report["visualizations"],
    }


MappingForSummary = dict[str, Any]


if __name__ == "__main__":
    app()
