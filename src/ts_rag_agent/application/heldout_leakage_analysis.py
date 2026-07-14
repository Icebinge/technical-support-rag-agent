from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg


@dataclass(frozen=True)
class LeakageQuestion:
    """One question row used by held-out leakage analysis."""

    source: str
    split: str
    question_id: str
    question_text: str


@dataclass(frozen=True)
class LeakageVisualization:
    """One generated leakage visualization artifact."""

    name: str
    path: str


def analyze_heldout_leakage(
    heldout_questions: Sequence[LeakageQuestion],
    development_questions: Sequence[LeakageQuestion],
    near_duplicate_threshold: float = 0.9,
    sample_limit: int = 20,
) -> dict:
    """Compare held-out questions against development questions."""

    if not heldout_questions:
        raise ValueError("heldout_questions must not be empty")
    if not development_questions:
        raise ValueError("development_questions must not be empty")
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    heldout_rows = [_normalized_row(question) for question in heldout_questions]
    development_rows = [_normalized_row(question) for question in development_questions]
    development_by_normalized: dict[str, list[dict]] = {}
    for row in development_rows:
        development_by_normalized.setdefault(row["normalized_question"], []).append(row)

    exact_overlaps = _exact_overlaps(
        heldout_rows=heldout_rows,
        development_by_normalized=development_by_normalized,
    )
    exact_heldout_ids = {
        overlap["heldout_question_id"] for overlap in exact_overlaps
    }
    near_duplicate_overlaps = _near_duplicate_overlaps(
        heldout_rows=[
            row for row in heldout_rows if row["question_id"] not in exact_heldout_ids
        ],
        development_rows=development_rows,
        threshold=near_duplicate_threshold,
    )
    near_duplicate_heldout_ids = {
        overlap["heldout_question_id"] for overlap in near_duplicate_overlaps
    }
    unhandled_heldout_ids = exact_heldout_ids | near_duplicate_heldout_ids
    source_counts = _source_counts(development_rows)
    return {
        "analysis_scope": (
            "Held-out leakage audit only. This report compares question text from "
            "the proposed held-out source against development sources. It does "
            "not evaluate answer quality, tune runtime parameters, or change the "
            "default runtime policy."
        ),
        "normalization": "lowercase_ascii_alnum_whitespace",
        "near_duplicate_threshold": near_duplicate_threshold,
        "counts": {
            "heldout_questions": len(heldout_rows),
            "development_questions": len(development_rows),
            "development_source_counts": source_counts,
            "exact_overlap_count": len(exact_heldout_ids),
            "exact_overlap_pair_count": len(exact_overlaps),
            "near_duplicate_overlap_count": len(near_duplicate_heldout_ids),
            "near_duplicate_overlap_pair_count": len(near_duplicate_overlaps),
            "unhandled_overlap_count": len(unhandled_heldout_ids),
            "heldout_questions_without_detected_overlap": (
                len(heldout_rows) - len(unhandled_heldout_ids)
            ),
        },
        "heldout_usable_without_exclusions": not unhandled_heldout_ids,
        "decision": _decision(len(unhandled_heldout_ids)),
        "exact_overlap_samples": exact_overlaps[:sample_limit],
        "near_duplicate_overlap_samples": near_duplicate_overlaps[:sample_limit],
    }


def write_leakage_visualizations(
    leakage_report: Mapping,
    output_dir: Path,
) -> list[LeakageVisualization]:
    """Write SVG charts for held-out leakage analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = leakage_report["counts"]
    charts = {
        "stage53_heldout_overlap_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage 53 proposed held-out overlap counts",
            bars=[
                BarDatum(
                    label="heldout questions",
                    value=float(counts["heldout_questions"]),
                    value_label=str(counts["heldout_questions"]),
                ),
                BarDatum(
                    label="exact overlaps",
                    value=float(counts["exact_overlap_count"]),
                    value_label=str(counts["exact_overlap_count"]),
                ),
                BarDatum(
                    label="near duplicate overlaps",
                    value=float(counts["near_duplicate_overlap_count"]),
                    value_label=str(counts["near_duplicate_overlap_count"]),
                ),
                BarDatum(
                    label="without detected overlap",
                    value=float(counts["heldout_questions_without_detected_overlap"]),
                    value_label=str(counts["heldout_questions_without_detected_overlap"]),
                ),
            ],
            x_label="question count",
            margin_left=260,
        ),
        "stage53_development_source_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage 53 development source counts",
            bars=[
                BarDatum(label=source, value=float(count), value_label=str(count))
                for source, count in sorted(
                    counts["development_source_counts"].items()
                )
            ],
            x_label="question count",
            margin_left=220,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(LeakageVisualization(name=filename, path=str(path)))
    return artifacts


def normalize_question_text(text: str) -> str:
    """Normalize question text for deterministic exact-match leakage checks."""

    lower = text.lower()
    alnum = re.sub(r"[^a-z0-9]+", " ", lower)
    return re.sub(r"\s+", " ", alnum).strip()


def _normalized_row(question: LeakageQuestion) -> dict:
    normalized = normalize_question_text(question.question_text)
    return {
        "source": question.source,
        "split": question.split,
        "question_id": question.question_id,
        "normalized_question": normalized,
        "tokens": set(normalized.split()),
    }


def _exact_overlaps(
    heldout_rows: Sequence[Mapping],
    development_by_normalized: Mapping[str, Sequence[Mapping]],
) -> list[dict]:
    overlaps = []
    for heldout_row in heldout_rows:
        matches = development_by_normalized.get(
            str(heldout_row["normalized_question"]),
            [],
        )
        for match in matches:
            overlaps.append(_overlap_row(heldout_row, match, similarity=1.0))
    return overlaps


def _near_duplicate_overlaps(
    heldout_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
    threshold: float,
) -> list[dict]:
    overlaps = []
    for heldout_row in heldout_rows:
        best_match = None
        best_similarity = 0.0
        for development_row in development_rows:
            similarity = _jaccard_similarity(
                heldout_row["tokens"],
                development_row["tokens"],
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = development_row
        if best_match is not None and best_similarity >= threshold:
            overlaps.append(
                _overlap_row(
                    heldout_row,
                    best_match,
                    similarity=round(best_similarity, 4),
                )
            )
    return overlaps


def _overlap_row(
    heldout_row: Mapping,
    development_row: Mapping,
    similarity: float,
) -> dict:
    return {
        "heldout_source": heldout_row["source"],
        "heldout_split": heldout_row["split"],
        "heldout_question_id": heldout_row["question_id"],
        "development_source": development_row["source"],
        "development_split": development_row["split"],
        "development_question_id": development_row["question_id"],
        "similarity": similarity,
    }


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _source_counts(rows: Sequence[Mapping]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['source']}::{row['split']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _decision(unhandled_overlap_count: int) -> str:
    if unhandled_overlap_count:
        return (
            "blocked: proposed held-out source overlaps with development data; "
            "do not run held-out evaluation metrics."
        )
    return "passed: proposed held-out source has no detected development overlap."
