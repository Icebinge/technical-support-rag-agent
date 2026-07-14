from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_split_plan import (
    plan_primeqa_hybrid_split,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 68"
_CREATED_AT = "2026-07-14"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"


@dataclass(frozen=True)
class PrimeQAHybridSplitFreezeBundle:
    """Stage68 report plus local raw split samples."""

    report: dict[str, Any]
    samples: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PrimeQAHybridSplitFreezeVisualization:
    """One generated Stage68 freeze visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _RawPrimeQARow:
    source_split: str
    source_row_index: int
    payload: Mapping[str, Any]

    @property
    def question_id(self) -> str:
        return str(self.payload["QUESTION_ID"])

    @property
    def identity(self) -> str:
        return f"{self.source_split}:{self.question_id}"


def freeze_primeqa_hybrid_split(
    *,
    train_questions_path: Path,
    dev_questions_path: Path,
    validation_reference_path: Path,
    document_disjoint_answer_doc_ratio: float = 0.10,
    remainder_train_ratio: float = 0.70,
    remainder_dev_ratio: float = 0.15,
    remainder_test_ratio: float = 0.15,
    seed: int = 20260714,
) -> PrimeQAHybridSplitFreezeBundle:
    """Freeze the Stage68 PrimeQA/TechQA hybrid split from the Stage67 protocol."""

    plan = plan_primeqa_hybrid_split(
        train_questions_path=train_questions_path,
        dev_questions_path=dev_questions_path,
        validation_reference_path=validation_reference_path,
        document_disjoint_answer_doc_ratio=document_disjoint_answer_doc_ratio,
        remainder_train_ratio=remainder_train_ratio,
        remainder_dev_ratio=remainder_dev_ratio,
        remainder_test_ratio=remainder_test_ratio,
        seed=seed,
    )
    raw_rows = _load_raw_rows(
        train_questions_path=train_questions_path,
        dev_questions_path=dev_questions_path,
        validation_reference_path=validation_reference_path,
    )
    assignments = _assignments_by_identity(plan["assignments"])
    samples = tuple(
        _frozen_sample(raw_row=row, assignment=assignments[row.identity])
        for row in raw_rows
    )
    freeze_checks = _freeze_checks(
        plan=plan,
        raw_rows=raw_rows,
        assignments=assignments,
        samples=samples,
    )
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "PrimeQA/TechQA hybrid split freeze. This stage materializes "
            "train/dev/test JSONL artifacts from the Stage67 dry-run boundary. "
            "It does not rebuild indexes, does not train or tune models, does "
            "not run final metrics, and does not change the default runtime."
        ),
        "source_files": plan["input_files"],
        "split_protocol": {
            **plan["split_protocol"],
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "source_stage": "Stage 67",
            "freeze_stage": _STAGE,
        },
        "stage67_plan_summary": {
            "input_summary": plan["input_summary"],
            "document_disjoint_summary": plan["document_disjoint_summary"],
            "split_summary": plan["split_summary"],
            "leakage_checks": plan["leakage_checks"],
            "decision": plan["decision"],
        },
        "frozen_split": _frozen_split_summary(samples),
        "source_summary": _source_summary(samples),
        "leakage_checks": plan["leakage_checks"],
        "freeze_checks": freeze_checks,
        "decision": _decision(freeze_checks),
    }
    return PrimeQAHybridSplitFreezeBundle(report=report, samples=samples)


def write_primeqa_frozen_split_jsonl(
    bundle: PrimeQAHybridSplitFreezeBundle,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Write train/dev/test JSONL files for the frozen Stage68 split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for split in ("train", "dev", "test"):
        split_samples = _samples_for_split(bundle.samples, split)
        output_path = output_dir / f"primeqa_hybrid_split_stage68_{split}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in split_samples:
                handle.write(_json_dumps(sample))
                handle.write("\n")
        artifacts.append(
            {
                "split": split,
                "path": str(output_path),
                "row_count": len(split_samples),
                "sha256": _file_sha256(output_path),
                "contains_raw_question_and_answer_text": True,
                "git_policy": "ignored_artifact",
            }
        )
    return artifacts


def write_primeqa_hybrid_split_freeze_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSplitFreezeVisualization]:
    """Write compact SVG charts for the frozen Stage68 split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage68_primeqa_frozen_split_rows.svg": render_horizontal_bar_chart_svg(
            title="Stage68 frozen split rows",
            bars=_split_metric_bars(report, "row_count"),
            x_label="row count",
            margin_left=240,
        ),
        "stage68_primeqa_frozen_answerable_rows.svg": render_horizontal_bar_chart_svg(
            title="Stage68 frozen answerable rows",
            bars=_split_metric_bars(report, "answerable_count"),
            x_label="answerable row count",
            margin_left=240,
        ),
        "stage68_primeqa_frozen_test_subtypes.svg": render_horizontal_bar_chart_svg(
            title="Stage68 frozen test subtypes",
            bars=_test_subtype_bars(report),
            x_label="test row count",
            margin_left=260,
        ),
        "stage68_primeqa_frozen_source_rows.svg": render_horizontal_bar_chart_svg(
            title="Stage68 frozen source rows",
            bars=_source_bars(report),
            x_label="row count",
            margin_left=260,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSplitFreezeVisualization(name=filename, path=str(path))
        )
    return artifacts


def _load_raw_rows(
    *,
    train_questions_path: Path,
    dev_questions_path: Path,
    validation_reference_path: Path,
) -> list[_RawPrimeQARow]:
    rows = []
    for source_split, path in (
        ("primeqa_train", train_questions_path),
        ("primeqa_dev", dev_questions_path),
        ("primeqa_validation", validation_reference_path),
    ):
        payload = _load_json_list(path)
        for index, raw_row in enumerate(payload, start=1):
            rows.append(
                _RawPrimeQARow(
                    source_split=source_split,
                    source_row_index=index,
                    payload=raw_row,
                )
            )
    _validate_unique_row_identities(rows)
    return rows


def _load_json_list(path: Path) -> list[Mapping[str, Any]]:
    _ensure_file(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list rows in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected object rows in {path}")
    return rows


def _assignments_by_identity(
    assignment_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    assignments = {}
    duplicate_identities = []
    for row in assignment_rows:
        identity = f"{row['source_split']}:{row['question_id']}"
        if identity in assignments:
            duplicate_identities.append(identity)
        assignments[identity] = row
    if duplicate_identities:
        sample = ", ".join(sorted(duplicate_identities)[:5])
        raise ValueError(f"Duplicate assignment identities: {sample}")
    return assignments


def _frozen_sample(
    *,
    raw_row: _RawPrimeQARow,
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    row = raw_row.payload
    answerable = _parse_answerable(row["ANSWERABLE"])
    return {
        "dataset": "primeqa_techqa",
        "split_name": _SPLIT_NAME,
        "protocol_version": _PROTOCOL_VERSION,
        "assigned_split": assignment["assigned_split"],
        "split_subtype": assignment["split_subtype"],
        "assignment_reason": assignment["assignment_reason"],
        "source_split": raw_row.source_split,
        "source_row_index": raw_row.source_row_index,
        "sample_id": raw_row.identity,
        "question_id": raw_row.question_id,
        "question_title": str(row.get("QUESTION_TITLE") or ""),
        "question_text": str(row.get("QUESTION_TEXT") or ""),
        "question": _full_question(row),
        "answerable": answerable,
        "answer": str(row.get("ANSWER") or "") if answerable else "",
        "answer_doc_id": str(row.get("DOCUMENT") or "") if answerable else None,
        "candidate_doc_ids": [str(doc_id) for doc_id in row.get("DOC_IDS", [])],
        "answer_span": {
            "start_offset": _to_optional_int(row.get("START_OFFSET")),
            "end_offset": _to_optional_int(row.get("END_OFFSET")),
        },
        "metadata": {
            "group_hash": assignment["group_hash"],
            "candidate_doc_count": assignment["candidate_doc_count"],
            "candidate_doc_hash": assignment["candidate_doc_hash"],
        },
    }


def _freeze_checks(
    *,
    plan: Mapping[str, Any],
    raw_rows: Sequence[_RawPrimeQARow],
    assignments: Mapping[str, Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    missing_assignment_count = sum(
        1 for row in raw_rows if row.identity not in assignments
    )
    split_counts = Counter(str(sample["assigned_split"]) for sample in samples)
    empty_required_splits = [
        split for split in ("train", "dev", "test") if split_counts[split] == 0
    ]
    failed_stage67_leakage_checks = [
        check for check in plan["leakage_checks"] if not check["passed"]
    ]
    document_disjoint_rows = sum(
        1 for sample in samples if sample["split_subtype"] == "document_disjoint"
    )
    return [
        _check(
            name="all_stage67_leakage_checks_passed",
            passed=not failed_stage67_leakage_checks,
            observed=len(failed_stage67_leakage_checks),
            expected=0,
        ),
        _check(
            name="all_raw_rows_have_assignments",
            passed=missing_assignment_count == 0,
            observed=missing_assignment_count,
            expected=0,
        ),
        _check(
            name="frozen_row_count_matches_input",
            passed=len(samples) == plan["input_summary"]["row_count"],
            observed=len(samples),
            expected=plan["input_summary"]["row_count"],
        ),
        _check(
            name="train_dev_test_are_nonempty",
            passed=not empty_required_splits,
            observed=len(empty_required_splits),
            expected=0,
        ),
        _check(
            name="document_disjoint_test_rows_materialized",
            passed=document_disjoint_rows
            == plan["document_disjoint_summary"]["document_disjoint_row_count"],
            observed=document_disjoint_rows,
            expected=plan["document_disjoint_summary"]["document_disjoint_row_count"],
        ),
    ]


def _frozen_split_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_split[str(sample["assigned_split"])].append(sample)
    split_summary = {
        split: _sample_summary(rows)
        for split, rows in sorted(samples_by_split.items())
    }
    return {
        "split_name": _SPLIT_NAME,
        "protocol_version": _PROTOCOL_VERSION,
        "row_count": len(samples),
        "sample_id_sha256": _sha256_lines(_sample_ids(samples)),
        "split_summary": split_summary,
        "test_subtype_counts": split_summary.get("test", {}).get("subtype_counts", {}),
    }


def _sample_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answerable_count = sum(bool(row["answerable"]) for row in rows)
    return {
        "row_count": len(rows),
        "group_count": len({str(row["metadata"]["group_hash"]) for row in rows}),
        "answerable_count": answerable_count,
        "unanswerable_count": len(rows) - answerable_count,
        "answerable_rate": _ratio(answerable_count, len(rows)),
        "source_split_counts": _counter_dict(str(row["source_split"]) for row in rows),
        "subtype_counts": _counter_dict(str(row["split_subtype"]) for row in rows),
        "sample_id_sha256": _sha256_lines(_sample_ids(rows)),
        "unique_answer_doc_ids": len(
            {
                str(row["answer_doc_id"])
                for row in rows
                if row["answerable"] and row["answer_doc_id"] is not None
            }
        ),
    }


def _source_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source_split_counts": _counter_dict(str(row["source_split"]) for row in samples),
        "source_by_assigned_split": {
            split: _counter_dict(
                str(row["source_split"])
                for row in samples
                if row["assigned_split"] == split
            )
            for split in ("train", "dev", "test")
        },
    }


def _decision(freeze_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = all(bool(check["passed"]) for check in freeze_checks)
    return {
        "status": (
            "primeqa_hybrid_split_frozen_for_rebuild"
            if passed
            else "primeqa_hybrid_split_freeze_blocked"
        ),
        "split_files_finalized": passed,
        "can_run_final_metrics_now": False,
        "can_rebuild_training_and_dev_artifacts_next": passed,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 69: rebuild PrimeQA train/dev/test data loaders and derived "
            "candidate artifacts from the frozen hybrid split, without using "
            "test rows for tuning"
        ),
        "reason": (
            "The split boundary is frozen for local artifact rebuilds, but no "
            "training, retrieval, answer-quality, or final-test metric has been "
            "run in this stage."
        ),
    }


def _samples_for_split(
    samples: Sequence[Mapping[str, Any]],
    split: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        [sample for sample in samples if sample["assigned_split"] == split],
        key=lambda sample: (
            str(sample["split_subtype"]),
            str(sample["source_split"]),
            int(sample["source_row_index"]),
            str(sample["question_id"]),
        ),
    )


def _split_metric_bars(report: Mapping[str, Any], metric_name: str) -> list[BarDatum]:
    return [
        BarDatum(split, float(summary[metric_name]), str(summary[metric_name]))
        for split, summary in report["frozen_split"]["split_summary"].items()
    ]


def _test_subtype_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    subtype_counts = report["frozen_split"].get("test_subtype_counts", {})
    return [
        BarDatum(subtype, float(count), str(count))
        for subtype, count in subtype_counts.items()
    ]


def _source_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(source, float(count), str(count))
        for source, count in report["source_summary"]["source_split_counts"].items()
    ]


def _validate_unique_row_identities(rows: Sequence[_RawPrimeQARow]) -> None:
    duplicate_identities = [
        identity for identity, count in Counter(row.identity for row in rows).items()
        if count > 1
    ]
    if duplicate_identities:
        sample = ", ".join(sorted(duplicate_identities)[:5])
        raise ValueError(f"Duplicate raw row identities: {sample}")


def _full_question(row: Mapping[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(row.get("QUESTION_TITLE") or "").strip(),
            str(row.get("QUESTION_TEXT") or "").strip(),
        )
        if part
    )


def _parse_answerable(value: Any) -> bool:
    normalized = str(value).strip().upper()
    if normalized == "Y":
        return True
    if normalized == "N":
        return False
    raise ValueError(f"Unsupported ANSWERABLE value: {value!r}")


def _to_optional_int(value: object) -> int | None:
    if value in (None, "", "-"):
        return None
    return int(value)


def _sample_ids(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(sample["sample_id"]) for sample in samples]


def _sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dumps(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


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


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
