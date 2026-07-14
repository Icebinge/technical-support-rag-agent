from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.domain.dataset import PrimeQAQuestion


@dataclass(frozen=True)
class PrimeQAHybridSplitSample:
    """One row from the frozen project-owned PrimeQA hybrid split."""

    split_name: str
    protocol_version: str
    assigned_split: str
    split_subtype: str
    source_split: str
    sample_id: str
    question_id: str
    question_title: str
    question_text: str
    answerable: bool
    answer: str
    answer_doc_id: str | None
    candidate_doc_ids: tuple[str, ...]
    start_offset: int | None
    end_offset: int | None

    def to_primeqa_question(self) -> PrimeQAQuestion:
        """Convert to the existing question model with a collision-safe sample id."""

        return PrimeQAQuestion(
            id=self.sample_id,
            title=self.question_title,
            text=self.question_text,
            answer=self.answer,
            answerable=self.answerable,
            answer_doc_id=self.answer_doc_id,
            doc_ids=list(self.candidate_doc_ids),
            start_offset=self.start_offset,
            end_offset=self.end_offset,
        )

    def to_primeqa_compatible_row(self) -> dict[str, Any]:
        """Write a PrimeQA-like row while preserving frozen split metadata."""

        return {
            "QUESTION_ID": self.sample_id,
            "QUESTION_TITLE": self.question_title,
            "QUESTION_TEXT": self.question_text,
            "DOCUMENT": self.answer_doc_id or "",
            "ANSWER": self.answer if self.answerable else "",
            "START_OFFSET": self.start_offset if self.start_offset is not None else "-",
            "END_OFFSET": self.end_offset if self.end_offset is not None else "-",
            "ANSWERABLE": "Y" if self.answerable else "N",
            "DOC_IDS": list(self.candidate_doc_ids),
            "FROZEN_SPLIT_NAME": self.split_name,
            "FROZEN_PROTOCOL_VERSION": self.protocol_version,
            "ASSIGNED_SPLIT": self.assigned_split,
            "SPLIT_SUBTYPE": self.split_subtype,
            "SOURCE_SPLIT": self.source_split,
            "SOURCE_QUESTION_ID": self.question_id,
        }


def load_primeqa_hybrid_split_samples(path: Path) -> list[PrimeQAHybridSplitSample]:
    """Load one frozen Stage68 split JSONL file."""

    _ensure_file(path)
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} in {path}")
            samples.append(_sample_from_row(row=row, path=path, line_number=line_number))
    _validate_unique_sample_ids(samples, path)
    return samples


def load_primeqa_hybrid_split_questions(
    split_paths: Mapping[str, Path],
) -> dict[str, list[PrimeQAQuestion]]:
    """Load frozen split samples as existing PrimeQAQuestion objects."""

    return {
        split: [
            sample.to_primeqa_question()
            for sample in load_primeqa_hybrid_split_samples(path)
        ]
        for split, path in split_paths.items()
    }


def write_primeqa_compatible_question_files(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Write PrimeQA-like JSON arrays for tools that expect QUESTION_* fields."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for split, samples in sorted(split_samples.items()):
        output_path = output_dir / f"primeqa_hybrid_stage69_{split}_Q_A.json"
        rows = [sample.to_primeqa_compatible_row() for sample in samples]
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts.append(
            {
                "split": split,
                "path": str(output_path),
                "row_count": len(rows),
                "contains_raw_question_and_answer_text": True,
                "git_policy": "ignored_artifact",
            }
        )
    return artifacts


def summarize_primeqa_hybrid_split_samples(
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
) -> dict[str, Any]:
    """Summarize loaded frozen split samples without raw text."""

    return {
        split: _sample_summary(samples)
        for split, samples in sorted(split_samples.items())
    }


def _sample_from_row(
    *,
    row: Mapping[str, Any],
    path: Path,
    line_number: int,
) -> PrimeQAHybridSplitSample:
    for field in (
        "split_name",
        "protocol_version",
        "assigned_split",
        "split_subtype",
        "source_split",
        "sample_id",
        "question_id",
        "question_title",
        "question_text",
        "answerable",
        "answer",
        "candidate_doc_ids",
        "answer_span",
    ):
        if field not in row:
            raise ValueError(f"Missing {field!r} on line {line_number} in {path}")
    answer_span = row["answer_span"]
    if not isinstance(answer_span, dict):
        raise ValueError(f"Expected answer_span object on line {line_number} in {path}")
    return PrimeQAHybridSplitSample(
        split_name=str(row["split_name"]),
        protocol_version=str(row["protocol_version"]),
        assigned_split=str(row["assigned_split"]),
        split_subtype=str(row["split_subtype"]),
        source_split=str(row["source_split"]),
        sample_id=str(row["sample_id"]),
        question_id=str(row["question_id"]),
        question_title=str(row["question_title"]),
        question_text=str(row["question_text"]),
        answerable=bool(row["answerable"]),
        answer=str(row.get("answer") or ""),
        answer_doc_id=str(row["answer_doc_id"]) if row.get("answer_doc_id") else None,
        candidate_doc_ids=tuple(str(doc_id) for doc_id in row["candidate_doc_ids"]),
        start_offset=_to_optional_int(answer_span.get("start_offset")),
        end_offset=_to_optional_int(answer_span.get("end_offset")),
    )


def _sample_summary(samples: Sequence[PrimeQAHybridSplitSample]) -> dict[str, Any]:
    answerable_count = sum(sample.answerable for sample in samples)
    return {
        "row_count": len(samples),
        "answerable_count": answerable_count,
        "unanswerable_count": len(samples) - answerable_count,
        "source_split_counts": _counter_dict(sample.source_split for sample in samples),
        "subtype_counts": _counter_dict(sample.split_subtype for sample in samples),
        "unique_answer_doc_ids": len(
            {
                sample.answer_doc_id
                for sample in samples
                if sample.answerable and sample.answer_doc_id is not None
            }
        ),
        "unique_candidate_doc_ids": len(
            {doc_id for sample in samples for doc_id in sample.candidate_doc_ids}
        ),
    }


def _validate_unique_sample_ids(
    samples: Sequence[PrimeQAHybridSplitSample],
    path: Path,
) -> None:
    duplicates = [
        sample_id
        for sample_id, count in Counter(sample.sample_id for sample in samples).items()
        if count > 1
    ]
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Duplicate sample_id values in {path}: {sample}")


def _to_optional_int(value: object) -> int | None:
    if value in (None, "", "-"):
        return None
    return int(value)


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
