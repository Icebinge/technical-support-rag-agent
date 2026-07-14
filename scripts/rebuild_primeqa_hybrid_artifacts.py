from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_split_rebuild import (
    rebuild_primeqa_hybrid_train_dev_artifacts,
    write_primeqa_hybrid_rebuild_candidate_artifacts,
    write_primeqa_hybrid_rebuild_question_artifacts,
    write_primeqa_hybrid_rebuild_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(
    help="Rebuild Stage69 loaders and train/dev candidate artifacts from the frozen split."
)


@app.command()
def main(
    train_split: Annotated[
        Path | None,
        typer.Option("--train-split", help="Frozen Stage68 train JSONL path."),
    ] = None,
    dev_split: Annotated[
        Path | None,
        typer.Option("--dev-split", help="Frozen Stage68 dev JSONL path."),
    ] = None,
    test_split: Annotated[
        Path | None,
        typer.Option("--test-split", help="Frozen Stage68 test JSONL path."),
    ] = None,
    documents: Annotated[
        Path | None,
        typer.Option("--documents", help="PrimeQA training_dev_technotes.sections.json."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output Stage69 rebuild report JSON path."),
    ] = None,
    question_output_dir: Annotated[
        Path | None,
        typer.Option("--question-output-dir", help="Output directory for question JSON files."),
    ] = None,
    candidate_output: Annotated[
        Path | None,
        typer.Option("--candidate-output", help="Output train/dev candidate JSONL path."),
    ] = None,
    candidate_summary_output: Annotated[
        Path | None,
        typer.Option("--candidate-summary-output", help="Output candidate summary JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
    candidate_splits: Annotated[
        str,
        typer.Option(
            "--candidate-splits",
            help="Comma-separated candidate artifact splits. Test is intentionally rejected.",
        ),
    ] = "train,dev",
    retrieval_top_k: Annotated[
        int,
        typer.Option("--retrieval-top-k", help="Number of retrieved documents."),
    ] = 5,
    evidence_selector: Annotated[
        str,
        typer.Option("--evidence-selector", help="Evidence selector for candidate rows."),
    ] = "hybrid-routing",
    max_candidates_per_document: Annotated[
        int,
        typer.Option(
            "--max-candidates-per-document",
            help="Maximum selector candidates retained from one document.",
        ),
    ] = 3,
    candidate_limit: Annotated[
        int,
        typer.Option("--candidate-limit", help="Maximum rows retained per question."),
    ] = 25,
    min_candidate_score: Annotated[
        float,
        typer.Option("--min-candidate-score", help="Minimum retained candidate score."),
    ] = 2.0,
) -> None:
    """Write Stage69 rebuild report and local ignored artifacts."""

    settings = ProjectSettings()
    default_split_dir = settings.artifact_dir / "primeqa_hybrid_split_stage68_splits"
    default_output_dir = settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_questions"
    split_paths = {
        "train": train_split
        or default_split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        "dev": dev_split
        or default_split_dir / "primeqa_hybrid_split_stage68_dev.jsonl",
        "test": test_split
        or default_split_dir / "primeqa_hybrid_split_stage68_test.jsonl",
    }
    documents_path = documents or (
        settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json"
    )
    output_path = output or settings.artifact_dir / "primeqa_hybrid_rebuild_stage69.json"
    question_dir = question_output_dir or default_output_dir
    candidate_path = candidate_output or (
        settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_candidates.jsonl"
    )
    candidate_summary_path = candidate_summary_output or (
        settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_candidates.summary.json"
    )
    visualization_output_dir = visualization_dir or (
        settings.artifact_dir / "primeqa_hybrid_rebuild_stage69_visuals"
    )

    for path in [*split_paths.values(), documents_path]:
        _ensure_file_exists(path)
    bundle = rebuild_primeqa_hybrid_train_dev_artifacts(
        split_paths=split_paths,
        documents_path=documents_path,
        candidate_splits=_parse_candidate_splits(candidate_splits),
        retrieval_top_k=retrieval_top_k,
        evidence_selector_name=evidence_selector,
        max_candidates_per_document=max_candidates_per_document,
        candidate_limit=candidate_limit,
        min_candidate_score=min_candidate_score,
    )
    question_artifacts = write_primeqa_hybrid_rebuild_question_artifacts(
        bundle=bundle,
        output_dir=question_dir,
    )
    candidate_artifact = write_primeqa_hybrid_rebuild_candidate_artifacts(
        bundle=bundle,
        dataset_output=candidate_path,
        summary_output=candidate_summary_path,
    )
    report = {
        **bundle.report,
        "question_artifacts": question_artifacts,
        "candidate_artifact": candidate_artifact,
    }
    visualizations = write_primeqa_hybrid_rebuild_visualizations(
        report=report,
        output_dir=visualization_output_dir,
    )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved PrimeQA hybrid rebuild report: {output_path}")


def _parse_candidate_splits(raw_splits: str) -> list[str]:
    split_names = [split.strip().lower() for split in raw_splits.split(",") if split.strip()]
    if not split_names:
        raise typer.BadParameter("--candidate-splits must not be empty.")
    return split_names


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "split_contract": report["split_contract"],
        "loaded_split_summary": report["loaded_split_summary"],
        "candidate_summary": report["candidate_build_summary"]["summary"],
        "guard_checks": report["guard_checks"],
        "decision": report["decision"],
        "question_artifacts": report["question_artifacts"],
        "candidate_artifact": report["candidate_artifact"],
        "visualizations": report["visualizations"],
        "timing_seconds": report["timing_seconds"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


MappingForSummary = dict[str, Any]


if __name__ == "__main__":
    app()
