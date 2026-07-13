from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.evidence_selection import (
    classify_question_route,
    create_sentence_evidence_selector,
)
from ts_rag_agent.application.local_window_gate_search import (
    DEFAULT_LOCAL_WINDOW_GATE_CONFIGS,
    evaluate_local_window_gate_cases,
    local_window_gate_search_analysis_to_dict,
    summarize_local_window_gate_search,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(
    help="Search safe gates for local-window rerank without changing runtime defaults."
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
    max_sentences: Annotated[
        int,
        typer.Option("--max-sentences", help="Maximum evidence sentences per answer."),
    ] = 3,
    min_sentence_score: Annotated[
        float,
        typer.Option("--min-sentence-score", help="Minimum candidate score to answer."),
    ] = 2.0,
    max_candidates_per_document: Annotated[
        int,
        typer.Option(
            "--max-candidates-per-document",
            help="Maximum candidates retained from the same document.",
        ),
    ] = 3,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON report path."),
    ] = None,
) -> None:
    """Evaluate local-window gate configs on full selected-candidate text."""

    split_names = _parse_splits(splits)
    _validate_options(
        retrieval_top_k=retrieval_top_k,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        max_candidates_per_document=max_candidates_per_document,
    )

    settings = ProjectSettings()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    output_path = output or (
        settings.artifact_dir / "local_window_gate_search_analysis.json"
    )
    _ensure_file_exists(documents_path)

    started_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    loaded_at = time.perf_counter()

    retriever = BM25Retriever()
    retriever.fit(documents)
    indexed_at = time.perf_counter()

    cases_by_gate_by_source = {}
    split_summaries = []
    for split in split_names:
        questions_path = _resolve_questions_path(training_dir, split)
        _ensure_file_exists(questions_path)
        questions = load_primeqa_questions(questions_path)
        split_started_at = time.perf_counter()
        split_cases = _evaluate_split(
            split=split,
            questions=questions,
            retriever=retriever,
            retrieval_top_k=retrieval_top_k,
            max_sentences=max_sentences,
            min_sentence_score=min_sentence_score,
            max_candidates_per_document=max_candidates_per_document,
        )
        split_finished_at = time.perf_counter()
        cases_by_gate_by_source[split] = split_cases
        split_summaries.append(
            {
                "split": split,
                "questions": len(questions),
                "answerable_questions": sum(question.answerable for question in questions),
                "timing_seconds": round(split_finished_at - split_started_at, 3),
            }
        )

    analyzed_at = time.perf_counter()
    analysis = summarize_local_window_gate_search(cases_by_gate_by_source)
    report = {
        "dataset": "PrimeQA/TechQA",
        "analysis_config": {
            "splits": split_names,
            "retrieval_top_k": retrieval_top_k,
            "baseline_selector": "hybrid-routing",
            "forced_local_selector": "local-window-rerank",
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "max_candidates_per_document": max_candidates_per_document,
            "gate_configs": [asdict(config) for config in DEFAULT_LOCAL_WINDOW_GATE_CONFIGS],
        },
        "paths": {
            "documents": str(documents_path),
        },
        "data": {
            "documents": len(documents),
            "splits": split_summaries,
        },
        **local_window_gate_search_analysis_to_dict(analysis),
        "timing_seconds": {
            "load_data": round(loaded_at - started_at, 3),
            "bm25_index": round(indexed_at - loaded_at, 3),
            "gate_search": round(analyzed_at - indexed_at, 3),
            "total": round(analyzed_at - started_at, 3),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(
        json.dumps(
            {
                "total_cases": report["total_cases"],
                "baseline_average_f1_by_source": report[
                    "baseline_average_f1_by_source"
                ],
                "forced_local_average_f1_by_source": report[
                    "forced_local_average_f1_by_source"
                ],
                "stable_gate_candidates": report["stable_gate_candidates"],
                "top_summary": report["top_summary"],
                "top_gate_summaries": report["gate_summaries"][:8],
                "timing_seconds": report["timing_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    typer.echo(f"Saved local-window gate search analysis: {output_path}")


def _evaluate_split(
    split: str,
    questions,
    retriever: BM25Retriever,
    retrieval_top_k: int,
    max_sentences: int,
    min_sentence_score: float,
    max_candidates_per_document: int,
) -> dict:
    baseline_generator = ExtractiveAnswerGenerator(
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        evidence_selector=create_sentence_evidence_selector(
            selector_name="hybrid-routing",
            max_candidates_per_document=max_candidates_per_document,
        ),
    )
    forced_local_generator = ExtractiveAnswerGenerator(
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        evidence_selector=create_sentence_evidence_selector(
            selector_name="local-window-rerank",
            max_candidates_per_document=max_candidates_per_document,
        ),
    )

    baseline_candidates_by_question_id = {}
    forced_local_candidates_by_question_id = {}
    question_route_by_id = {}
    for question in questions:
        if not question.answerable:
            continue
        retrieval_results = retriever.search(
            question.full_question,
            top_k=retrieval_top_k,
        )
        baseline_candidates_by_question_id[question.id] = (
            baseline_generator.select_answer_candidates(question, retrieval_results)
        )
        forced_local_candidates_by_question_id[question.id] = (
            forced_local_generator.select_answer_candidates(question, retrieval_results)
        )
        question_route_by_id[question.id] = classify_question_route(question)

    return evaluate_local_window_gate_cases(
        source_label=split,
        questions=questions,
        baseline_candidates_by_question_id=baseline_candidates_by_question_id,
        forced_local_candidates_by_question_id=forced_local_candidates_by_question_id,
        question_route_by_id=question_route_by_id,
        gate_configs=DEFAULT_LOCAL_WINDOW_GATE_CONFIGS,
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
    max_sentences: int,
    min_sentence_score: float,
    max_candidates_per_document: int,
) -> None:
    if retrieval_top_k <= 0:
        raise typer.BadParameter("--retrieval-top-k must be positive.")
    if max_sentences <= 0:
        raise typer.BadParameter("--max-sentences must be positive.")
    if min_sentence_score < 0:
        raise typer.BadParameter("--min-sentence-score must be non-negative.")
    if max_candidates_per_document <= 0:
        raise typer.BadParameter("--max-candidates-per-document must be positive.")


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
