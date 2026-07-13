from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset import (
    build_candidate_reranker_dataset,
    candidate_reranker_dataset_build_to_dict,
    candidate_reranker_row_to_dict,
)
from ts_rag_agent.application.evidence_selection import create_sentence_evidence_selector
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(
    help="Build an offline feature dataset for candidate reranker experiments."
)


@app.command()
def main(
    splits: Annotated[
        str,
        typer.Option("--splits", help="Comma-separated question splits: dev,train."),
    ] = "dev,train",
    retrieval_top_k: Annotated[
        int,
        typer.Option("--retrieval-top-k", help="Number of retrieved documents."),
    ] = 5,
    evidence_selector: Annotated[
        str,
        typer.Option(
            "--evidence-selector",
            help="Candidate pool selector, for example hybrid-routing.",
        ),
    ] = "hybrid-routing",
    max_candidates_per_document: Annotated[
        int,
        typer.Option(
            "--max-candidates-per-document",
            help="Maximum selector candidates retained from the same document.",
        ),
    ] = 3,
    candidate_limit: Annotated[
        int,
        typer.Option("--candidate-limit", help="Maximum rows retained per question."),
    ] = 25,
    min_candidate_score: Annotated[
        float,
        typer.Option(
            "--min-candidate-score",
            help="Minimum candidate score retained in the dataset.",
        ),
    ] = 2.0,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output JSONL dataset path."),
    ] = None,
    summary_output: Annotated[
        Path | None,
        typer.Option("--summary-output", help="Output JSON summary path."),
    ] = None,
) -> None:
    """Build candidate rows with runtime features separated from gold labels."""

    split_names = _parse_splits(splits)
    _validate_options(
        retrieval_top_k=retrieval_top_k,
        max_candidates_per_document=max_candidates_per_document,
        candidate_limit=candidate_limit,
        min_candidate_score=min_candidate_score,
    )

    settings = ProjectSettings()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    selector_slug = evidence_selector.strip().lower().replace("-", "_")
    split_slug = "_".join(split_names)
    output_path = output or (
        settings.artifact_dir
        / f"candidate_reranker_dataset_{split_slug}_{selector_slug}.jsonl"
    )
    summary_path = summary_output or output_path.with_suffix(".summary.json")

    _ensure_file_exists(documents_path)
    started_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    loaded_at = time.perf_counter()

    retriever = BM25Retriever()
    retriever.fit(documents)
    indexed_at = time.perf_counter()

    split_questions = {}
    split_paths = {}
    for split in split_names:
        questions_path = _resolve_questions_path(training_dir, split)
        _ensure_file_exists(questions_path)
        split_paths[split] = str(questions_path)
        split_questions[split] = load_primeqa_questions(questions_path)

    selector = create_sentence_evidence_selector(
        selector_name=evidence_selector,
        max_candidates_per_document=max_candidates_per_document,
    )
    build = build_candidate_reranker_dataset(
        split_questions=split_questions,
        search_fn=lambda question, top_k: retriever.search(
            question.full_question,
            top_k=top_k,
        ),
        evidence_selector=selector,
        retrieval_top_k=retrieval_top_k,
        candidate_limit=candidate_limit,
        min_candidate_score=min_candidate_score,
    )
    built_at = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dataset_file:
        for row in build.rows:
            dataset_file.write(
                json.dumps(candidate_reranker_row_to_dict(row), ensure_ascii=False)
                + "\n"
            )

    summary = {
        "dataset": "PrimeQA/TechQA",
        "paths": {
            "documents": str(documents_path),
            "questions": split_paths,
            "dataset_output": str(output_path),
        },
        "build_config": {
            "splits": split_names,
            "retrieval_top_k": retrieval_top_k,
            "evidence_selector": selector.name,
            "max_candidates_per_document": max_candidates_per_document,
            "candidate_limit": candidate_limit,
            "min_candidate_score": min_candidate_score,
        },
        "data": {
            "documents": len(documents),
            "questions_by_split": {
                split: len(questions)
                for split, questions in split_questions.items()
            },
        },
        **candidate_reranker_dataset_build_to_dict(build),
        "timing_seconds": {
            "load_data": round(loaded_at - started_at, 3),
            "bm25_index": round(indexed_at - loaded_at, 3),
            "dataset_build": round(built_at - indexed_at, 3),
            "total": round(built_at - started_at, 3),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(
        json.dumps(
            {
                "summary": asdict(build.summary),
                "output": str(output_path),
                "summary_output": str(summary_path),
                "timing_seconds": summary["timing_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_splits(raw_splits: str) -> list[str]:
    split_names = [split.strip().lower() for split in raw_splits.split(",") if split.strip()]
    if not split_names:
        raise typer.BadParameter("--splits must not be empty.")
    allowed = {"dev", "train"}
    invalid = sorted(set(split_names) - allowed)
    if invalid:
        raise typer.BadParameter(f"Unsupported split(s): {', '.join(invalid)}")
    return split_names


def _validate_options(
    retrieval_top_k: int,
    max_candidates_per_document: int,
    candidate_limit: int,
    min_candidate_score: float,
) -> None:
    if retrieval_top_k <= 0:
        raise typer.BadParameter("--retrieval-top-k must be positive.")
    if max_candidates_per_document <= 0:
        raise typer.BadParameter("--max-candidates-per-document must be positive.")
    if candidate_limit <= 0:
        raise typer.BadParameter("--candidate-limit must be positive.")
    if min_candidate_score < 0:
        raise typer.BadParameter("--min-candidate-score must be non-negative.")


def _resolve_questions_path(training_dir: Path, split: str) -> Path:
    if split == "dev":
        return training_dir / "dev_Q_A.json"
    if split == "train":
        return training_dir / "training_Q_A.json"
    raise typer.BadParameter("--splits must contain only dev and train.")


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
