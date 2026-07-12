from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from ts_rag_agent.domain.dataset import DatasetStats, TechQASample


def load_nvidia_samples(train_json_path: Path) -> list[TechQASample]:
    """Load NVIDIA TechQA-RAG-Eval rows into typed samples."""

    rows = json.loads(train_json_path.read_text(encoding="utf-8"))
    return [TechQASample.model_validate(row) for row in rows]


def list_corpus_filenames(corpus_zip_path: Path) -> set[str]:
    """Return document basenames available in the compressed corpus."""

    with ZipFile(corpus_zip_path) as archive:
        return {Path(info.filename).name for info in archive.infolist() if not info.is_dir()}


def compute_dataset_stats(samples: list[TechQASample], corpus_filenames: set[str]) -> DatasetStats:
    referenced = {
        context.filename
        for sample in samples
        for context in sample.contexts
        if context.filename
    }
    context_counts = [len(sample.contexts) for sample in samples]
    missing = referenced - corpus_filenames
    answerable = sum(1 for sample in samples if sample.is_answerable)

    return DatasetStats(
        total_rows=len(samples),
        answerable_rows=answerable,
        impossible_rows=len(samples) - answerable,
        unique_referenced_files=len(referenced),
        missing_referenced_files=len(missing),
        corpus_files=len(corpus_filenames),
        min_contexts=min(context_counts) if context_counts else 0,
        max_contexts=max(context_counts) if context_counts else 0,
        avg_contexts=round(sum(context_counts) / len(context_counts), 3)
        if context_counts
        else 0.0,
    )
