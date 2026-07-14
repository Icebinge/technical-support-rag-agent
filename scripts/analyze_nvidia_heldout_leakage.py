from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.heldout_leakage_analysis import (
    LeakageQuestion,
    analyze_heldout_leakage,
    write_leakage_visualizations,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions
from ts_rag_agent.infrastructure.techqa_loader import load_nvidia_samples

app = typer.Typer(help="Audit NVIDIA TechQA-RAG-Eval leakage against PrimeQA train/dev.")


@app.command()
def main(
    nvidia_samples: Annotated[
        Path,
        typer.Option("--nvidia-samples", help="NVIDIA TechQA-RAG-Eval train.json."),
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
        typer.Option("--output", help="Output leakage report JSON path."),
    ],
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
        typer.Option("--sample-limit", help="Maximum overlap samples saved per class."),
    ] = 20,
) -> None:
    """Write leakage report before any held-out evaluation metrics are run."""

    for path in [nvidia_samples, primeqa_train_questions, primeqa_dev_questions]:
        _ensure_file_exists(path)

    leakage_report = analyze_heldout_leakage(
        heldout_questions=_load_nvidia_questions(nvidia_samples),
        development_questions=(
            _load_primeqa_questions(primeqa_train_questions, split="train")
            + _load_primeqa_questions(primeqa_dev_questions, split="dev")
        ),
        near_duplicate_threshold=near_duplicate_threshold,
        sample_limit=sample_limit,
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_leakage_visualizations(
            leakage_report=leakage_report,
            output_dir=visualization_dir,
        )
    report = {
        **leakage_report,
        "paths": {
            "nvidia_samples": str(nvidia_samples),
            "primeqa_train_questions": str(primeqa_train_questions),
            "primeqa_dev_questions": str(primeqa_dev_questions),
        },
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved NVIDIA held-out leakage report: {output}")


def _load_nvidia_questions(path: Path) -> list[LeakageQuestion]:
    return [
        LeakageQuestion(
            source="nvidia/TechQA-RAG-Eval",
            split="train.json",
            question_id=sample.id,
            question_text=sample.question,
        )
        for sample in load_nvidia_samples(path)
    ]


def _load_primeqa_questions(path: Path, split: str) -> list[LeakageQuestion]:
    return [
        LeakageQuestion(
            source="PrimeQA/TechQA",
            split=split,
            question_id=question.id,
            question_text=question.full_question,
        )
        for question in load_primeqa_questions(path)
    ]


def _console_summary(report: dict) -> dict:
    return {
        "counts": report["counts"],
        "heldout_usable_without_exclusions": report[
            "heldout_usable_without_exclusions"
        ],
        "decision": report["decision"],
        "exact_overlap_sample_count": len(report["exact_overlap_samples"]),
        "near_duplicate_overlap_sample_count": len(
            report["near_duplicate_overlap_samples"]
        ),
        "visualizations": report["visualizations"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


if __name__ == "__main__":
    app()
