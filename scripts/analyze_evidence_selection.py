from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceSelector,
    create_sentence_evidence_selector,
)
from ts_rag_agent.application.evidence_selection_analysis import (
    EVIDENCE_SELECTION_BUCKET_DEFINITIONS,
    EvidenceSelectionAnalysisResult,
    EvidenceSelectionAnalyzer,
    case_to_dict,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_documents,
    load_primeqa_questions,
)

app = typer.Typer(
    help="Analyze why gold documents in retrieved context are not cited."
)


@app.command()
def main(
    split: Annotated[
        str,
        typer.Option("--split", help="Question split to analyze: dev or train."),
    ] = "dev",
    retrieval_top_k_values: Annotated[
        str,
        typer.Option("--retrieval-top-k-values", help="Comma-separated top-k values."),
    ] = "5,10,20",
    base_top_k: Annotated[
        int,
        typer.Option("--base-top-k", help="Base top-k for newly available gold docs."),
    ] = 5,
    max_sentences: Annotated[
        int,
        typer.Option("--max-sentences", help="Maximum evidence sentences per answer."),
    ] = 3,
    min_sentence_score: Annotated[
        float,
        typer.Option("--min-sentence-score", help="Minimum candidate score to answer."),
    ] = 2.0,
    evidence_selector: Annotated[
        str,
        typer.Option(
            "--evidence-selector",
            help="Sentence selector: overlap, bm25-sentence, or answer-aware.",
        ),
    ] = "overlap",
    max_candidates_per_document: Annotated[
        int,
        typer.Option(
            "--max-candidates-per-document",
            help="Maximum candidates retained from the same document.",
        ),
    ] = 1,
    sample_limit_per_top_k: Annotated[
        int,
        typer.Option("--sample-limit-per-top-k", help="Maximum failure cases per top-k."),
    ] = 20,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON report path."),
    ] = None,
) -> None:
    """Run evidence selection analysis and save a JSON report."""

    parsed_top_k_values = _parse_int_values(
        retrieval_top_k_values,
        option_name="--retrieval-top-k-values",
    )
    _validate_options(
        retrieval_top_k_values=parsed_top_k_values,
        base_top_k=base_top_k,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        max_candidates_per_document=max_candidates_per_document,
        sample_limit_per_top_k=sample_limit_per_top_k,
    )

    settings = ProjectSettings()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    documents_path = training_dir / "training_dev_technotes.sections.json"
    questions_path = _resolve_questions_path(training_dir, split)
    selector_slug = _selector_slug(evidence_selector)
    output_path = output or (
        settings.artifact_dir / f"evidence_selection_analysis_{split}_{selector_slug}.json"
    )

    _ensure_file_exists(documents_path)
    _ensure_file_exists(questions_path)

    started_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    questions = load_primeqa_questions(questions_path)
    loaded_at = time.perf_counter()

    retriever = BM25Retriever()
    retriever.fit(documents)
    indexed_at = time.perf_counter()

    sentence_selector = _create_selector(
        evidence_selector=evidence_selector,
        max_candidates_per_document=max_candidates_per_document,
    )
    analyzer = EvidenceSelectionAnalyzer(
        retriever=retriever,
        answer_generator=ExtractiveAnswerGenerator(
            max_sentences=max_sentences,
            min_sentence_score=min_sentence_score,
            evidence_selector=sentence_selector,
        ),
        base_top_k=base_top_k,
    )
    analysis = analyzer.analyze(
        questions=questions,
        retrieval_top_k_values=parsed_top_k_values,
        sample_limit_per_top_k=sample_limit_per_top_k,
    )
    analyzed_at = time.perf_counter()

    report = _build_report(
        split=split,
        documents_path=documents_path,
        questions_path=questions_path,
        document_count=len(documents),
        question_count=len(questions),
        retrieval_top_k_values=parsed_top_k_values,
        base_top_k=base_top_k,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        evidence_selector_name=sentence_selector.name,
        max_candidates_per_document=max_candidates_per_document,
        sample_limit_per_top_k=sample_limit_per_top_k,
        analysis=analysis,
        timing_seconds={
            "load_data": round(loaded_at - started_at, 3),
            "bm25_index": round(indexed_at - loaded_at, 3),
            "evidence_selection_analysis": round(analyzed_at - indexed_at, 3),
            "total": round(analyzed_at - started_at, 3),
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(_summarize_report(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved evidence selection analysis: {output_path}")


def _build_report(
    split: str,
    documents_path: Path,
    questions_path: Path,
    document_count: int,
    question_count: int,
    retrieval_top_k_values: list[int],
    base_top_k: int,
    max_sentences: int,
    min_sentence_score: float,
    evidence_selector_name: str,
    max_candidates_per_document: int,
    sample_limit_per_top_k: int,
    analysis: EvidenceSelectionAnalysisResult,
    timing_seconds: dict[str, float],
) -> dict:
    return {
        "dataset": "PrimeQA/TechQA",
        "split": split,
        "paths": {
            "documents": str(documents_path),
            "questions": str(questions_path),
        },
        "analysis_config": {
            "retrieval_top_k_values": retrieval_top_k_values,
            "base_top_k": base_top_k,
            "answer_generator": "extractive_sentence_baseline",
            "evidence_selector": evidence_selector_name,
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "max_candidates_per_document": max_candidates_per_document,
            "sample_limit_per_top_k": sample_limit_per_top_k,
        },
        "data": {
            "documents": document_count,
            "questions": question_count,
        },
        "bucket_definitions": EVIDENCE_SELECTION_BUCKET_DEFINITIONS,
        "summaries": [asdict(summary) for summary in analysis.summaries],
        "cases_by_top_k": {
            str(top_k): [case_to_dict(case) for case in cases]
            for top_k, cases in analysis.cases_by_top_k.items()
        },
        "timing_seconds": timing_seconds,
    }


def _summarize_report(report: dict) -> dict:
    return {
        "dataset": report["dataset"],
        "split": report["split"],
        "analysis_config": report["analysis_config"],
        "data": report["data"],
        "summaries": report["summaries"],
        "timing_seconds": report["timing_seconds"],
    }


def _parse_int_values(raw_value: str, option_name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must contain comma-separated integers.") from exc
    if not values:
        raise typer.BadParameter(f"{option_name} must not be empty.")
    return values


def _validate_options(
    retrieval_top_k_values: list[int],
    base_top_k: int,
    max_sentences: int,
    min_sentence_score: float,
    max_candidates_per_document: int,
    sample_limit_per_top_k: int,
) -> None:
    if any(value <= 0 for value in retrieval_top_k_values):
        raise typer.BadParameter("--retrieval-top-k-values must contain positive integers.")
    if base_top_k <= 0:
        raise typer.BadParameter("--base-top-k must be positive.")
    if max_sentences <= 0:
        raise typer.BadParameter("--max-sentences must be positive.")
    if min_sentence_score < 0:
        raise typer.BadParameter("--min-sentence-score must be non-negative.")
    if max_candidates_per_document <= 0:
        raise typer.BadParameter("--max-candidates-per-document must be positive.")
    if sample_limit_per_top_k < 0:
        raise typer.BadParameter("--sample-limit-per-top-k must be non-negative.")


def _create_selector(
    evidence_selector: str,
    max_candidates_per_document: int,
) -> SentenceEvidenceSelector:
    try:
        return create_sentence_evidence_selector(
            selector_name=evidence_selector,
            max_candidates_per_document=max_candidates_per_document,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _selector_slug(evidence_selector: str) -> str:
    return evidence_selector.strip().lower().replace("-", "_")


def _resolve_questions_path(training_dir: Path, split: str) -> Path:
    normalized = split.strip().lower()
    if normalized == "dev":
        return training_dir / "dev_Q_A.json"
    if normalized == "train":
        return training_dir / "training_Q_A.json"
    raise typer.BadParameter("--split must be either dev or train.")


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
