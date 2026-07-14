from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.heldout_leakage_analysis import normalize_question_text
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

_CSV_FIELD_SIZE_LIMIT = 2_147_483_647
_README_ROW_COUNT_CLAIM = 32_252
_MSQA_REPOSITORY_URL = "https://github.com/microsoft/Microsoft-Q-A-MSQA-"

_REQUIRED_FIELDS = (
    "QuestionId",
    "AnswerId",
    "QuestionText",
    "AnswerText",
    "ProcessedAnswerText",
    "Url",
    "Split",
)
_ANSWER_FIELD_CANDIDATES = (
    "AnswerText",
    "ProcessedAnswerText",
    "DoubleProcessedAnswerText",
)


@dataclass(frozen=True)
class FileFingerprint:
    """Size and checksum for a local source file."""

    path: str
    bytes: int
    sha256: str

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class MsqaRowSample:
    """A compact public-data sample from the MSQA CSV."""

    question_id: str
    split: str
    is_azure: str
    is_m365: str
    is_other: str
    is_short: str
    is_long: str
    url: str
    question_preview: str
    answer_preview: str

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "split": self.split,
            "is_azure": self.is_azure,
            "is_m365": self.is_m365,
            "is_other": self.is_other,
            "is_short": self.is_short,
            "is_long": self.is_long,
            "url": self.url,
            "question_preview": self.question_preview,
            "answer_preview": self.answer_preview,
        }


@dataclass(frozen=True)
class MsqaCsvScan:
    """Structured scan results from the local MSQA CSV."""

    fieldnames: tuple[str, ...]
    row_count: int
    unique_question_ids: int
    duplicate_question_id_rows: int
    duplicate_normalized_question_rows: int
    malformed_row_count: int
    missing_by_field: Mapping[str, int]
    split_counts: Mapping[str, int]
    azure_counts: Mapping[str, int]
    m365_counts: Mapping[str, int]
    other_counts: Mapping[str, int]
    is_short_counts: Mapping[str, int]
    is_long_counts: Mapping[str, int]
    rows_with_any_url: int
    rows_with_row_url: int
    rows_with_learn_answers_url: int
    rows_with_question_text_link: int
    rows_with_answer_text_link: int
    rows_with_processed_answer_link: int
    rows_with_double_processed_answer_link: int
    rows_with_processed_answer_learn_link: int
    rows_with_processed_answer_azure_docish_link: int
    question_ids: frozenset[str]
    split_by_question_id: Mapping[str, str]
    normalized_question_ids: Mapping[str, tuple[str, ...]]
    samples: tuple[MsqaRowSample, ...]

    def to_schema_report(self, readme_row_count_claim: int) -> dict[str, Any]:
        return {
            "field_count": len(self.fieldnames),
            "fields": list(self.fieldnames),
            "required_fields": list(_REQUIRED_FIELDS),
            "required_field_missing_counts": {
                field: int(self.missing_by_field.get(field, 0))
                for field in _REQUIRED_FIELDS
            },
            "answer_field_candidates": {
                field: {
                    "missing_count": int(self.missing_by_field.get(field, 0)),
                    "available_count": self.row_count
                    - int(self.missing_by_field.get(field, 0)),
                }
                for field in _ANSWER_FIELD_CANDIDATES
            },
            "row_count": self.row_count,
            "readme_row_count_claim": readme_row_count_claim,
            "row_count_delta_vs_readme_claim": self.row_count
            - readme_row_count_claim,
            "unique_question_ids": self.unique_question_ids,
            "duplicate_question_id_rows": self.duplicate_question_id_rows,
            "duplicate_normalized_question_rows": self.duplicate_normalized_question_rows,
            "malformed_row_count": self.malformed_row_count,
            "missing_by_field": dict(sorted(self.missing_by_field.items())),
        }

    def to_distribution_report(self) -> dict[str, Any]:
        return {
            "split_counts": dict(sorted(self.split_counts.items())),
            "azure_counts": dict(sorted(self.azure_counts.items())),
            "m365_counts": dict(sorted(self.m365_counts.items())),
            "other_counts": dict(sorted(self.other_counts.items())),
            "is_short_counts": dict(sorted(self.is_short_counts.items())),
            "is_long_counts": dict(sorted(self.is_long_counts.items())),
        }

    def to_link_report(self) -> dict[str, Any]:
        return {
            "rows_with_any_url": _count_and_percent(
                self.rows_with_any_url,
                self.row_count,
            ),
            "rows_with_row_url": _count_and_percent(
                self.rows_with_row_url,
                self.row_count,
            ),
            "rows_with_learn_answers_url": _count_and_percent(
                self.rows_with_learn_answers_url,
                self.row_count,
            ),
            "rows_with_question_text_link": _count_and_percent(
                self.rows_with_question_text_link,
                self.row_count,
            ),
            "rows_with_answer_text_link": _count_and_percent(
                self.rows_with_answer_text_link,
                self.row_count,
            ),
            "rows_with_processed_answer_link": _count_and_percent(
                self.rows_with_processed_answer_link,
                self.row_count,
            ),
            "rows_with_double_processed_answer_link": _count_and_percent(
                self.rows_with_double_processed_answer_link,
                self.row_count,
            ),
            "rows_with_processed_answer_learn_link": _count_and_percent(
                self.rows_with_processed_answer_learn_link,
                self.row_count,
            ),
            "rows_with_processed_answer_azure_docish_link": _count_and_percent(
                self.rows_with_processed_answer_azure_docish_link,
                self.row_count,
            ),
        }


@dataclass(frozen=True)
class MsqaSchemaProbeVisualization:
    """One generated Stage 56 MSQA probe visualization."""

    name: str
    path: str


def probe_msqa_dataset(
    *,
    msqa_csv_path: Path,
    test_id_path: Path,
    readme_path: Path,
    repository_head: str,
    primeqa_train_questions_path: Path,
    primeqa_dev_questions_path: Path,
    repository_url: str = _MSQA_REPOSITORY_URL,
    readme_row_count_claim: int = _README_ROW_COUNT_CLAIM,
    sample_limit: int = 3,
) -> dict[str, Any]:
    """Probe the local MSQA CSV before any held-out metrics are attempted."""

    for path in [
        msqa_csv_path,
        test_id_path,
        readme_path,
        primeqa_train_questions_path,
        primeqa_dev_questions_path,
    ]:
        _ensure_file(path)
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    scan = _scan_msqa_csv(msqa_csv_path=msqa_csv_path, sample_limit=sample_limit)
    test_id_report = _build_test_id_report(
        test_id_path=test_id_path,
        scan=scan,
    )
    exact_leakage_report = _primeqa_exact_leakage_precheck(
        scan=scan,
        primeqa_train_questions_path=primeqa_train_questions_path,
        primeqa_dev_questions_path=primeqa_dev_questions_path,
    )
    return {
        "stage": "Stage 56",
        "created_at": "2026-07-14",
        "analysis_scope": (
            "MSQA local schema probe, source-link coverage audit, and PrimeQA "
            "exact-overlap precheck. This report does not run RAG answer-quality "
            "metrics, does not freeze a final MSQA evaluation split, does not run "
            "near-duplicate leakage search, and does not change runtime defaults."
        ),
        "source": {
            "repository_url": repository_url,
            "repository_head": repository_head,
            "download_method": (
                "git clone --depth 1 --filter=blob:none "
                "https://github.com/microsoft/Microsoft-Q-A-MSQA-.git "
                "data/raw/msqa_repo"
            ),
            "source_files": {
                "msqa_csv": _fingerprint(msqa_csv_path).to_report_dict(),
                "test_id": _fingerprint(test_id_path).to_report_dict(),
                "readme": _fingerprint(readme_path).to_report_dict(),
            },
            "git_policy": (
                "MSQA raw data is stored under data/raw/msqa_repo, which is ignored "
                "by repository policy and is not committed."
            ),
        },
        "schema": scan.to_schema_report(readme_row_count_claim),
        "distributions": scan.to_distribution_report(),
        "source_link_coverage": scan.to_link_report(),
        "test_id_file": test_id_report,
        "primeqa_exact_leakage_precheck": exact_leakage_report,
        "readiness": _readiness(
            scan,
            test_id_report,
            exact_leakage_report,
            readme_row_count_claim,
        ),
        "samples": [sample.to_report_dict() for sample in scan.samples],
        "next_stage": (
            "Stage 57: MSQA adapter contract, near-duplicate leakage audit, and "
            "project-owned evaluation split freeze"
        ),
    }


def write_msqa_schema_probe_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[MsqaSchemaProbeVisualization]:
    """Write SVG charts for the Stage 56 MSQA schema probe."""

    output_dir.mkdir(parents=True, exist_ok=True)
    schema = report["schema"]
    distributions = report["distributions"]
    link_coverage = report["source_link_coverage"]
    test_id_file = report["test_id_file"]
    leakage = report["primeqa_exact_leakage_precheck"]
    charts = {
        "stage56_msqa_split_distribution.svg": render_horizontal_bar_chart_svg(
            title="Stage 56 MSQA split distribution",
            bars=[
                BarDatum(label=str(split), value=float(count), value_label=str(count))
                for split, count in distributions["split_counts"].items()
            ],
            x_label="row count",
            margin_left=220,
        ),
        "stage56_msqa_source_link_coverage.svg": render_horizontal_bar_chart_svg(
            title="Stage 56 MSQA source-link coverage",
            bars=[
                _coverage_bar("row Url", link_coverage["rows_with_row_url"]),
                _coverage_bar(
                    "learn answers Url",
                    link_coverage["rows_with_learn_answers_url"],
                ),
                _coverage_bar(
                    "answer text link",
                    link_coverage["rows_with_answer_text_link"],
                ),
                _coverage_bar(
                    "processed answer link",
                    link_coverage["rows_with_processed_answer_link"],
                ),
                _coverage_bar(
                    "processed answer learn link",
                    link_coverage["rows_with_processed_answer_learn_link"],
                ),
            ],
            x_label="row count",
            margin_left=260,
        ),
        "stage56_msqa_domain_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage 56 MSQA domain flags",
            bars=[
                BarDatum(
                    label="IsAzure=True",
                    value=float(distributions["azure_counts"].get("True", 0)),
                    value_label=str(distributions["azure_counts"].get("True", 0)),
                ),
                BarDatum(
                    label="IsM365=True",
                    value=float(distributions["m365_counts"].get("True", 0)),
                    value_label=str(distributions["m365_counts"].get("True", 0)),
                ),
                BarDatum(
                    label="IsOther=True",
                    value=float(distributions["other_counts"].get("True", 0)),
                    value_label=str(distributions["other_counts"].get("True", 0)),
                ),
            ],
            x_label="row count",
            margin_left=220,
        ),
        "stage56_msqa_test_id_coverage.svg": render_horizontal_bar_chart_svg(
            title="Stage 56 MSQA test_id coverage",
            bars=[
                BarDatum(
                    label="test_id rows",
                    value=float(test_id_file["test_id_count"]),
                    value_label=str(test_id_file["test_id_count"]),
                ),
                BarDatum(
                    label="found in CSV",
                    value=float(test_id_file["test_ids_found_in_csv"]),
                    value_label=str(test_id_file["test_ids_found_in_csv"]),
                ),
                BarDatum(
                    label="missing from CSV",
                    value=float(test_id_file["test_ids_missing_from_csv_count"]),
                    value_label=str(test_id_file["test_ids_missing_from_csv_count"]),
                ),
            ],
            x_label="ID count",
            margin_left=220,
        ),
        "stage56_msqa_primeqa_exact_overlap.svg": render_horizontal_bar_chart_svg(
            title="Stage 56 MSQA PrimeQA exact-overlap precheck",
            bars=[
                BarDatum(
                    label="MSQA rows",
                    value=float(schema["row_count"]),
                    value_label=str(schema["row_count"]),
                ),
                BarDatum(
                    label="PrimeQA train/dev rows",
                    value=float(leakage["development_question_count"]),
                    value_label=str(leakage["development_question_count"]),
                ),
                BarDatum(
                    label="exact overlaps",
                    value=float(leakage["exact_overlap_msqa_question_count"]),
                    value_label=str(leakage["exact_overlap_msqa_question_count"]),
                ),
            ],
            x_label="question count",
            margin_left=260,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(MsqaSchemaProbeVisualization(name=filename, path=str(path)))
    return artifacts


def _scan_msqa_csv(*, msqa_csv_path: Path, sample_limit: int) -> MsqaCsvScan:
    csv.field_size_limit(_CSV_FIELD_SIZE_LIMIT)
    missing_by_field: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    azure_counts: Counter[str] = Counter()
    m365_counts: Counter[str] = Counter()
    other_counts: Counter[str] = Counter()
    is_short_counts: Counter[str] = Counter()
    is_long_counts: Counter[str] = Counter()
    question_ids: set[str] = set()
    duplicate_question_id_rows = 0
    malformed_row_count = 0
    split_by_question_id: dict[str, str] = {}
    normalized_question_ids: dict[str, list[str]] = {}
    samples: list[MsqaRowSample] = []
    rows_with_any_url = 0
    rows_with_row_url = 0
    rows_with_learn_answers_url = 0
    rows_with_question_text_link = 0
    rows_with_answer_text_link = 0
    rows_with_processed_answer_link = 0
    rows_with_double_processed_answer_link = 0
    rows_with_processed_answer_learn_link = 0
    rows_with_processed_answer_azure_docish_link = 0

    with msqa_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        row_count = 0
        for row in reader:
            row_count += 1
            if None in row:
                malformed_row_count += 1
            for field in fieldnames:
                if not str(row.get(field) or "").strip():
                    missing_by_field[field] += 1

            question_id = str(row.get("QuestionId") or "").strip()
            if question_id in question_ids:
                duplicate_question_id_rows += 1
            question_ids.add(question_id)
            split_by_question_id[question_id] = str(row.get("Split") or "").strip()
            normalized_question = normalize_question_text(str(row.get("QuestionText") or ""))
            normalized_question_ids.setdefault(normalized_question, []).append(question_id)

            split_counts[str(row.get("Split") or "").strip()] += 1
            azure_counts[str(row.get("IsAzure") or "").strip()] += 1
            m365_counts[str(row.get("IsM365") or "").strip()] += 1
            other_counts[str(row.get("IsOther") or "").strip()] += 1
            is_short_counts[str(row.get("isShort") or "").strip()] += 1
            is_long_counts[str(row.get("isLong") or "").strip()] += 1

            row_url = str(row.get("Url") or "").strip().lower()
            question_text = str(row.get("QuestionText") or "")
            answer_text = str(row.get("AnswerText") or "")
            processed_answer = str(row.get("ProcessedAnswerText") or "")
            double_processed_answer = str(row.get("DoubleProcessedAnswerText") or "")
            all_text = " ".join([row_url, question_text, answer_text, processed_answer])

            if _contains_url(all_text):
                rows_with_any_url += 1
            if row_url:
                rows_with_row_url += 1
            if row_url.startswith("https://learn.microsoft.com/en-us/answers/questions/"):
                rows_with_learn_answers_url += 1
            if _contains_url(question_text):
                rows_with_question_text_link += 1
            if _contains_url(answer_text):
                rows_with_answer_text_link += 1
            if _contains_url(processed_answer):
                rows_with_processed_answer_link += 1
            if _contains_url(double_processed_answer):
                rows_with_double_processed_answer_link += 1
            if "https://learn.microsoft.com/en-us/" in processed_answer.lower():
                rows_with_processed_answer_learn_link += 1
            if "/azure/" in processed_answer.lower():
                rows_with_processed_answer_azure_docish_link += 1
            if len(samples) < sample_limit:
                samples.append(_sample_from_row(row))

    duplicate_normalized_question_rows = sum(
        max(0, len(ids) - 1) for ids in normalized_question_ids.values()
    )
    return MsqaCsvScan(
        fieldnames=fieldnames,
        row_count=row_count,
        unique_question_ids=len(question_ids),
        duplicate_question_id_rows=duplicate_question_id_rows,
        duplicate_normalized_question_rows=duplicate_normalized_question_rows,
        malformed_row_count=malformed_row_count,
        missing_by_field=dict(missing_by_field),
        split_counts=dict(split_counts),
        azure_counts=dict(azure_counts),
        m365_counts=dict(m365_counts),
        other_counts=dict(other_counts),
        is_short_counts=dict(is_short_counts),
        is_long_counts=dict(is_long_counts),
        rows_with_any_url=rows_with_any_url,
        rows_with_row_url=rows_with_row_url,
        rows_with_learn_answers_url=rows_with_learn_answers_url,
        rows_with_question_text_link=rows_with_question_text_link,
        rows_with_answer_text_link=rows_with_answer_text_link,
        rows_with_processed_answer_link=rows_with_processed_answer_link,
        rows_with_double_processed_answer_link=rows_with_double_processed_answer_link,
        rows_with_processed_answer_learn_link=rows_with_processed_answer_learn_link,
        rows_with_processed_answer_azure_docish_link=rows_with_processed_answer_azure_docish_link,
        question_ids=frozenset(question_ids),
        split_by_question_id=split_by_question_id,
        normalized_question_ids={
            normalized: tuple(ids) for normalized, ids in normalized_question_ids.items()
        },
        samples=tuple(samples),
    )


def _build_test_id_report(*, test_id_path: Path, scan: MsqaCsvScan) -> dict[str, Any]:
    test_ids = [
        line.strip()
        for line in test_id_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique_test_ids = set(test_ids)
    duplicate_count = len(test_ids) - len(unique_test_ids)
    found_ids = unique_test_ids & scan.question_ids
    missing_ids = unique_test_ids - scan.question_ids
    found_split_counts = Counter(
        scan.split_by_question_id[question_id] for question_id in found_ids
    )
    return {
        "test_id_count": len(test_ids),
        "unique_test_id_count": len(unique_test_ids),
        "duplicate_test_id_count": duplicate_count,
        "test_ids_found_in_csv": len(found_ids),
        "test_ids_missing_from_csv_count": len(missing_ids),
        "test_ids_missing_from_csv": _sort_numeric_strings(missing_ids),
        "found_test_id_split_counts": dict(sorted(found_split_counts.items())),
    }


def _primeqa_exact_leakage_precheck(
    *,
    scan: MsqaCsvScan,
    primeqa_train_questions_path: Path,
    primeqa_dev_questions_path: Path,
) -> dict[str, Any]:
    overlap_pairs = []
    source_counts = {}
    for split, path in (
        ("train", primeqa_train_questions_path),
        ("dev", primeqa_dev_questions_path),
    ):
        questions = load_primeqa_questions(path)
        source_counts[f"PrimeQA/TechQA::{split}"] = len(questions)
        for question in questions:
            normalized_question = normalize_question_text(question.full_question)
            for msqa_question_id in scan.normalized_question_ids.get(
                normalized_question,
                (),
            ):
                overlap_pairs.append(
                    {
                        "msqa_question_id": msqa_question_id,
                        "primeqa_split": split,
                        "primeqa_question_id": question.id,
                        "similarity": 1.0,
                    }
                )
    exact_msqa_ids = {pair["msqa_question_id"] for pair in overlap_pairs}
    return {
        "method": "exact_normalized_question_text_overlap_only",
        "normalization": "lowercase_ascii_alnum_whitespace",
        "development_question_count": sum(source_counts.values()),
        "development_source_counts": dict(sorted(source_counts.items())),
        "exact_overlap_pair_count": len(overlap_pairs),
        "exact_overlap_msqa_question_count": len(exact_msqa_ids),
        "passed_exact_precheck": not overlap_pairs,
        "samples": overlap_pairs[:20],
        "not_run_in_stage56": [
            "near_duplicate_token_jaccard_search",
            "semantic_duplicate_search",
            "answer_or_document_overlap_search",
        ],
    }


def _readiness(
    scan: MsqaCsvScan,
    test_id_report: Mapping[str, Any],
    exact_leakage_report: Mapping[str, Any],
    readme_row_count_claim: int,
) -> dict[str, Any]:
    passed_checks = [
        "local_msqa_csv_downloaded_and_parseable",
        "all_rows_have_unique_question_ids",
        "all_rows_have_source_url",
        "primeqa_exact_question_overlap_precheck_passed",
    ]
    blockers = [
        "near_duplicate_leakage_audit_not_run",
        "project_owned_msqa_evaluation_split_not_frozen",
        "msqa_adapter_contract_not_implemented",
        "no_native_unanswerable_rows_for_refusal_evaluation",
    ]
    if scan.row_count != readme_row_count_claim:
        blockers.append("local_row_count_differs_from_readme_claim")
    if test_id_report["test_ids_missing_from_csv_count"]:
        blockers.append("test_id_file_contains_ids_missing_from_local_csv")
    if scan.missing_by_field.get("DoubleProcessedAnswerText", 0):
        blockers.append("double_processed_answer_text_has_missing_rows")
    if exact_leakage_report["exact_overlap_pair_count"]:
        blockers.append("primeqa_exact_question_overlap_detected")
    return {
        "status": "schema_probe_passed_but_metrics_blocked",
        "can_run_final_metrics_now": False,
        "default_runtime_policy": "unchanged",
        "passed_checks": passed_checks,
        "blocking_issues_before_metrics": blockers,
        "decision": (
            "MSQA remains the recommended external candidate, but Stage 56 only "
            "approves continued adapter/leakage work. It does not approve final "
            "held-out metrics."
        ),
    }


def _fingerprint(path: Path) -> FileFingerprint:
    data = path.read_bytes()
    return FileFingerprint(
        path=str(path),
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def _sample_from_row(row: Mapping[str, Any]) -> MsqaRowSample:
    return MsqaRowSample(
        question_id=str(row.get("QuestionId") or ""),
        split=str(row.get("Split") or ""),
        is_azure=str(row.get("IsAzure") or ""),
        is_m365=str(row.get("IsM365") or ""),
        is_other=str(row.get("IsOther") or ""),
        is_short=str(row.get("isShort") or ""),
        is_long=str(row.get("isLong") or ""),
        url=str(row.get("Url") or ""),
        question_preview=_preview(str(row.get("QuestionText") or "")),
        answer_preview=_preview(str(row.get("ProcessedAnswerText") or "")),
    )


def _preview(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _contains_url(text: str) -> bool:
    lowered = text.lower()
    return "http://" in lowered or "https://" in lowered


def _count_and_percent(count: int, total: int) -> dict[str, Any]:
    return {
        "count": count,
        "percent": round((count / total * 100), 3) if total else 0.0,
    }


def _coverage_bar(label: str, coverage: Mapping[str, Any]) -> BarDatum:
    return BarDatum(
        label=label,
        value=float(coverage["count"]),
        value_label=f'{coverage["count"]} ({coverage["percent"]}%)',
    )


def _sort_numeric_strings(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: int(value) if value.isdigit() else value)
