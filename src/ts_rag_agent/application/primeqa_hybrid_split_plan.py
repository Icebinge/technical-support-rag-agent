from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 67"
_CREATED_AT = "2026-07-14"
_DEFAULT_SEED = 20260714
_DEFAULT_DOCUMENT_DISJOINT_RATIO = 0.10
_DEFAULT_REMAINDER_RATIOS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class PrimeQAHybridSplitVisualization:
    """One generated Stage67 visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _SplitRow:
    source_split: str
    question_id: str
    normalized_question: str
    answerable: bool
    answer_doc_id: str | None
    candidate_doc_ids: tuple[str, ...]

    @property
    def group_key(self) -> str:
        answer_key = self.answer_doc_id if self.answerable else "UNANSWERABLE"
        return f"{self.normalized_question}::{answer_key}"

    @property
    def group_hash(self) -> str:
        return _hash_text(self.group_key)


@dataclass(frozen=True)
class _GroupAssignment:
    split: str
    subtype: str
    reason: str


def plan_primeqa_hybrid_split(
    *,
    train_questions_path: Path,
    dev_questions_path: Path,
    validation_reference_path: Path,
    document_disjoint_answer_doc_ratio: float = _DEFAULT_DOCUMENT_DISJOINT_RATIO,
    remainder_train_ratio: float = _DEFAULT_REMAINDER_RATIOS[0],
    remainder_dev_ratio: float = _DEFAULT_REMAINDER_RATIOS[1],
    remainder_test_ratio: float = _DEFAULT_REMAINDER_RATIOS[2],
    seed: int = _DEFAULT_SEED,
) -> dict[str, Any]:
    """Plan a leak-aware hybrid split without rewriting raw PrimeQA data."""

    _validate_options(
        document_disjoint_answer_doc_ratio=document_disjoint_answer_doc_ratio,
        remainder_ratios=(
            remainder_train_ratio,
            remainder_dev_ratio,
            remainder_test_ratio,
        ),
    )
    rows = _load_all_rows(
        train_questions_path=train_questions_path,
        dev_questions_path=dev_questions_path,
        validation_reference_path=validation_reference_path,
    )
    groups = _groups_by_key(rows)
    answer_doc_ids = sorted(
        {
            row.answer_doc_id
            for row in rows
            if row.answerable and row.answer_doc_id is not None
        }
    )
    selected_document_ids = _select_document_disjoint_answer_docs(
        answer_doc_ids=answer_doc_ids,
        ratio=document_disjoint_answer_doc_ratio,
        seed=seed,
    )
    assignments = _assign_groups(
        groups=groups,
        selected_document_ids=selected_document_ids,
        remainder_ratios=(
            remainder_train_ratio,
            remainder_dev_ratio,
            remainder_test_ratio,
        ),
        seed=seed,
    )
    assignment_rows = _assignment_rows(groups=groups, assignments=assignments)
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "PrimeQA/TechQA hybrid split dry-run only. This stage includes the "
            "original train, dev, and validation_reference rows in one planning "
            "pool; strictly isolates a deterministic 10% sample of answer "
            "documents; then group-random splits the remaining rows by "
            "normalized question plus answer document. It does not rewrite raw "
            "data, does not rebuild retrieval indexes, does not run metrics, "
            "and does not change the default runtime."
        ),
        "input_files": {
            "train_questions": _fingerprint(train_questions_path),
            "dev_questions": _fingerprint(dev_questions_path),
            "validation_reference": _fingerprint(validation_reference_path),
        },
        "split_protocol": {
            "seed": seed,
            "pool_sources": [
                "PrimeQA/TechQA training_Q_A.json",
                "PrimeQA/TechQA dev_Q_A.json",
                "PrimeQA/TechQA validation_reference.json",
            ],
            "validation_handling": (
                "included in the planning pool; duplicate normalized-question "
                "groups stay in one split"
            ),
            "document_disjoint_answer_doc_ratio": document_disjoint_answer_doc_ratio,
            "document_disjoint_definition": (
                "Selected answer documents are fully isolated: any group whose "
                "candidate DOC_IDS contain a selected document is assigned to "
                "document_disjoint_test, so selected documents do not appear in "
                "train/dev/random-test candidate sets."
            ),
            "group_key": "normalized_question + answer_doc_id, or UNANSWERABLE",
            "remainder_split_ratios": {
                "train": remainder_train_ratio,
                "dev": remainder_dev_ratio,
                "random_test": remainder_test_ratio,
            },
        },
        "input_summary": _input_summary(rows=rows, groups=groups),
        "document_disjoint_summary": _document_disjoint_summary(
            groups=groups,
            assignments=assignments,
            selected_document_ids=selected_document_ids,
        ),
        "split_summary": _split_summary(assignment_rows),
        "leakage_checks": _leakage_checks(
            groups=groups,
            assignments=assignments,
            assignment_rows=assignment_rows,
            selected_document_ids=selected_document_ids,
        ),
        "assignments": [row for row in assignment_rows],
        "decision": _decision(),
    }


def write_primeqa_hybrid_split_assignments(
    report: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Write row-level split assignments without raw question or answer text."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = report["assignments"]
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_primeqa_hybrid_split_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSplitVisualization]:
    """Write compact SVG charts for the Stage67 hybrid split dry-run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage67_primeqa_split_rows.svg": render_horizontal_bar_chart_svg(
            title="Stage67 PrimeQA split rows",
            bars=_split_metric_bars(report, "row_count"),
            x_label="row count",
            margin_left=240,
        ),
        "stage67_primeqa_answerable_rows.svg": render_horizontal_bar_chart_svg(
            title="Stage67 PrimeQA answerable rows",
            bars=_split_metric_bars(report, "answerable_count"),
            x_label="answerable row count",
            margin_left=240,
        ),
        "stage67_primeqa_test_subtypes.svg": render_horizontal_bar_chart_svg(
            title="Stage67 PrimeQA test subtypes",
            bars=_test_subtype_bars(report),
            x_label="test row count",
            margin_left=260,
        ),
        "stage67_primeqa_source_rows.svg": render_horizontal_bar_chart_svg(
            title="Stage67 PrimeQA input source rows",
            bars=_source_bars(report),
            x_label="row count",
            margin_left=260,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(PrimeQAHybridSplitVisualization(name=filename, path=str(path)))
    return artifacts


def _load_all_rows(
    *,
    train_questions_path: Path,
    dev_questions_path: Path,
    validation_reference_path: Path,
) -> list[_SplitRow]:
    sources = [
        ("primeqa_train", train_questions_path),
        ("primeqa_dev", dev_questions_path),
        ("primeqa_validation", validation_reference_path),
    ]
    rows = []
    for source_split, path in sources:
        _ensure_file(path)
        raw_rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list):
            raise ValueError(f"Expected list rows in {path}")
        rows.extend(_split_row(source_split=source_split, row=row) for row in raw_rows)
    return rows


def _split_row(*, source_split: str, row: Mapping[str, Any]) -> _SplitRow:
    answerable = _parse_answerable(row["ANSWERABLE"])
    answer_doc_id = str(row["DOCUMENT"]) if answerable else None
    candidate_doc_ids = tuple(str(doc_id) for doc_id in row.get("DOC_IDS", ()))
    return _SplitRow(
        source_split=source_split,
        question_id=str(row["QUESTION_ID"]),
        normalized_question=_normalize_question(row),
        answerable=answerable,
        answer_doc_id=answer_doc_id,
        candidate_doc_ids=candidate_doc_ids,
    )


def _groups_by_key(rows: Sequence[_SplitRow]) -> dict[str, list[_SplitRow]]:
    groups: dict[str, list[_SplitRow]] = defaultdict(list)
    for row in rows:
        groups[row.group_key].append(row)
    return dict(groups)


def _select_document_disjoint_answer_docs(
    *,
    answer_doc_ids: Sequence[str],
    ratio: float,
    seed: int,
) -> set[str]:
    shuffled = list(answer_doc_ids)
    random.Random(seed).shuffle(shuffled)
    selected_count = math.ceil(len(shuffled) * ratio)
    return set(shuffled[:selected_count])


def _assign_groups(
    *,
    groups: Mapping[str, Sequence[_SplitRow]],
    selected_document_ids: set[str],
    remainder_ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, _GroupAssignment]:
    assignments: dict[str, _GroupAssignment] = {}
    remainder_groups: dict[str, Sequence[_SplitRow]] = {}
    for group_key, group_rows in groups.items():
        candidate_doc_ids = set().union(*(set(row.candidate_doc_ids) for row in group_rows))
        if candidate_doc_ids & selected_document_ids:
            assignments[group_key] = _GroupAssignment(
                split="test",
                subtype="document_disjoint",
                reason="candidate_doc_intersects_selected_document",
            )
        else:
            remainder_groups[group_key] = group_rows

    remainder_assignments = _assign_remainder_groups(
        groups=remainder_groups,
        ratios=remainder_ratios,
        seed=seed,
    )
    assignments.update(remainder_assignments)
    return assignments


def _assign_remainder_groups(
    *,
    groups: Mapping[str, Sequence[_SplitRow]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, _GroupAssignment]:
    split_names = ("train", "dev", "test")
    subtypes = {
        "train": "group_random_train",
        "dev": "group_random_dev",
        "test": "group_random_test",
    }
    total_rows = sum(len(rows) for rows in groups.values())
    targets = {
        name: max(total_rows * ratio, 1.0)
        for name, ratio in zip(split_names, ratios, strict=True)
    }
    current = {name: 0 for name in split_names}
    ordered_group_keys = sorted(groups)
    random.Random(seed).shuffle(ordered_group_keys)
    assignments = {}
    for group_key in ordered_group_keys:
        split = min(
            split_names,
            key=lambda name: (
                current[name] / targets[name],
                current[name],
                name,
            ),
        )
        current[split] += len(groups[group_key])
        assignments[group_key] = _GroupAssignment(
            split=split,
            subtype=subtypes[split],
            reason="group_random_remainder_split",
        )
    return assignments


def _assignment_rows(
    *,
    groups: Mapping[str, Sequence[_SplitRow]],
    assignments: Mapping[str, _GroupAssignment],
) -> list[dict[str, Any]]:
    rows = []
    for group_key, group_rows in groups.items():
        assignment = assignments[group_key]
        for row in group_rows:
            rows.append(
                {
                    "assigned_split": assignment.split,
                    "split_subtype": assignment.subtype,
                    "assignment_reason": assignment.reason,
                    "source_split": row.source_split,
                    "question_id": row.question_id,
                    "group_hash": row.group_hash,
                    "answerable": row.answerable,
                    "answer_doc_id": row.answer_doc_id,
                    "candidate_doc_count": len(row.candidate_doc_ids),
                    "candidate_doc_hash": _hash_text("\n".join(row.candidate_doc_ids)),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["assigned_split"],
            row["split_subtype"],
            row["source_split"],
            row["question_id"],
        ),
    )


def _input_summary(
    *,
    rows: Sequence[_SplitRow],
    groups: Mapping[str, Sequence[_SplitRow]],
) -> dict[str, Any]:
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    return {
        "row_count": len(rows),
        "group_count": len(groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_row_count": sum(len(group) for group in duplicate_groups),
        "source_split_counts": _counter_dict(row.source_split for row in rows),
        "answerable_count": sum(row.answerable for row in rows),
        "unanswerable_count": sum(not row.answerable for row in rows),
        "answerable_rate": _ratio(sum(row.answerable for row in rows), len(rows)),
        "unique_answer_doc_ids": len(
            {
                row.answer_doc_id
                for row in rows
                if row.answerable and row.answer_doc_id is not None
            }
        ),
        "unique_candidate_doc_ids": len(
            {doc_id for row in rows for doc_id in row.candidate_doc_ids}
        ),
        "duplicate_group_source_patterns": _counter_dict(
            "+".join(sorted({row.source_split for row in group}))
            for group in duplicate_groups
        ),
    }


def _document_disjoint_summary(
    *,
    groups: Mapping[str, Sequence[_SplitRow]],
    assignments: Mapping[str, _GroupAssignment],
    selected_document_ids: set[str],
) -> dict[str, Any]:
    document_groups = [
        group
        for group_key, group in groups.items()
        if assignments[group_key].subtype == "document_disjoint"
    ]
    answer_doc_selected_groups = [
        group
        for group in document_groups
        if any(row.answer_doc_id in selected_document_ids for row in group)
    ]
    candidate_only_groups = [
        group
        for group in document_groups
        if not any(row.answer_doc_id in selected_document_ids for row in group)
    ]
    rows = [row for group in document_groups for row in group]
    return {
        "unique_answer_doc_ids": len(
            {
                row.answer_doc_id
                for group in groups.values()
                for row in group
                if row.answerable and row.answer_doc_id is not None
            }
        ),
        "selected_answer_doc_count": len(selected_document_ids),
        "selected_answer_doc_ids_sha256": _hash_text(
            "\n".join(sorted(selected_document_ids))
        ),
        "selected_answer_doc_sample": sorted(selected_document_ids)[:20],
        "document_disjoint_group_count": len(document_groups),
        "document_disjoint_row_count": len(rows),
        "answer_doc_selected_group_count": len(answer_doc_selected_groups),
        "candidate_doc_intersection_only_group_count": len(candidate_only_groups),
        "answerable_count": sum(row.answerable for row in rows),
        "unanswerable_count": sum(not row.answerable for row in rows),
        "source_split_counts": _counter_dict(row.source_split for row in rows),
    }


def _split_summary(assignments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_split[str(row["assigned_split"])].append(row)
    return {
        split: _split_row_summary(rows)
        for split, rows in sorted(by_split.items())
    }


def _split_row_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answerable_count = sum(bool(row["answerable"]) for row in rows)
    return {
        "row_count": len(rows),
        "group_count": len({str(row["group_hash"]) for row in rows}),
        "answerable_count": answerable_count,
        "unanswerable_count": len(rows) - answerable_count,
        "answerable_rate": _ratio(answerable_count, len(rows)),
        "source_split_counts": _counter_dict(str(row["source_split"]) for row in rows),
        "subtype_counts": _counter_dict(str(row["split_subtype"]) for row in rows),
        "unique_answer_doc_ids": len(
            {
                str(row["answer_doc_id"])
                for row in rows
                if row["answerable"] and row["answer_doc_id"] is not None
            }
        ),
    }


def _leakage_checks(
    *,
    groups: Mapping[str, Sequence[_SplitRow]],
    assignments: Mapping[str, _GroupAssignment],
    assignment_rows: Sequence[Mapping[str, Any]],
    selected_document_ids: set[str],
) -> list[dict[str, Any]]:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    non_document_test_selected_answer_docs = []
    for row in assignment_rows:
        splits_by_group[str(row["group_hash"])].add(str(row["assigned_split"]))
        if (
            row["split_subtype"] != "document_disjoint"
            and row["answer_doc_id"] in selected_document_ids
        ):
            non_document_test_selected_answer_docs.append(row)
    groups_crossing_splits = {
        group_hash: splits
        for group_hash, splits in splits_by_group.items()
        if len(splits) > 1
    }
    non_document_selected_candidate_groups = [
        group_key
        for group_key, group_rows in groups.items()
        if assignments[group_key].subtype != "document_disjoint"
        and any(
            set(row.candidate_doc_ids).intersection(selected_document_ids)
            for row in group_rows
        )
    ]
    return [
        _check(
            name="normalized_question_answer_doc_groups_do_not_cross_splits",
            passed=not groups_crossing_splits,
            observed=len(groups_crossing_splits),
            expected=0,
        ),
        _check(
            name="selected_document_answer_docs_only_in_document_disjoint_test",
            passed=not non_document_test_selected_answer_docs,
            observed=len(non_document_test_selected_answer_docs),
            expected=0,
        ),
        _check(
            name="selected_document_candidate_doc_ids_only_in_document_disjoint_test",
            passed=not non_document_selected_candidate_groups,
            observed=len(non_document_selected_candidate_groups),
            expected=0,
        ),
    ]


def _decision() -> dict[str, Any]:
    return {
        "status": "primeqa_hybrid_split_dry_run_ready_for_review",
        "split_files_finalized": False,
        "can_run_final_metrics_now": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 68: review the Stage67 dry-run distribution, then confirm "
            "whether to freeze this hybrid split and rebuild training/dev/test "
            "artifacts from the new split boundary"
        ),
        "reason": (
            "The dry-run preserves document-style RAG by using PrimeQA/TechQA "
            "questions and technotes, but old Stage31-66 tuning evidence must "
            "not be treated as valid final-test evidence after a new split."
        ),
    }


def _split_metric_bars(report: Mapping[str, Any], metric_name: str) -> list[BarDatum]:
    return [
        BarDatum(split, float(summary[metric_name]), str(summary[metric_name]))
        for split, summary in report["split_summary"].items()
    ]


def _test_subtype_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    test_summary = report["split_summary"].get("test", {})
    subtype_counts = test_summary.get("subtype_counts", {})
    return [
        BarDatum(subtype, float(count), str(count))
        for subtype, count in subtype_counts.items()
    ]


def _source_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(source, float(count), str(count))
        for source, count in report["input_summary"]["source_split_counts"].items()
    ]


def _validate_options(
    *,
    document_disjoint_answer_doc_ratio: float,
    remainder_ratios: tuple[float, float, float],
) -> None:
    if not 0 < document_disjoint_answer_doc_ratio < 1:
        raise ValueError("document_disjoint_answer_doc_ratio must be between 0 and 1")
    if any(ratio <= 0 for ratio in remainder_ratios):
        raise ValueError("remainder split ratios must be positive")
    if not math.isclose(sum(remainder_ratios), 1.0, abs_tol=0.000001):
        raise ValueError("remainder split ratios must sum to 1.0")


def _normalize_question(row: Mapping[str, Any]) -> str:
    text = f"{row.get('QUESTION_TITLE') or ''} {row.get('QUESTION_TEXT') or ''}"
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _parse_answerable(value: Any) -> bool:
    normalized = str(value).strip().upper()
    if normalized == "Y":
        return True
    if normalized == "N":
        return False
    raise ValueError(f"Unsupported ANSWERABLE value: {value!r}")


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "expected": expected,
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
