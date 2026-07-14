from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.heldout_leakage_analysis import (
    LeakageQuestion,
    normalize_question_text,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

_CSV_FIELD_SIZE_LIMIT = 2_147_483_647
_ADAPTER_CONTRACT_VERSION = "msqa_eval_adapter_v1"
_SPLIT_PROTOCOL_VERSION = "msqa_project_eval_split_v1"
_EVALUATION_SPLIT_NAME = "msqa_stage57_project_eval_v1"
_DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.9
_SOURCE_SPLIT_FOR_EVALUATION = "test"
_SOURCE_URL_PREFIX = "https://learn.microsoft.com/en-us/answers/questions/"
_CONTRACT_REQUIRED_FIELDS = (
    "QuestionId",
    "AnswerId",
    "QuestionText",
    "ProcessedAnswerText",
    "Url",
    "Split",
)


@dataclass(frozen=True)
class MsqaEvaluationRow:
    """One MSQA row after applying the Stage 57 adapter contract."""

    question_id: str
    answer_id: str
    source_split: str
    question: str
    answer: str
    source_url: str
    tags: str
    is_azure: str
    is_m365: str
    is_other: str
    is_short: str
    is_long: str
    normalized_question: str
    source_row_index: int

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(self.normalized_question.split())

    def to_split_sample(self) -> dict[str, Any]:
        return {
            "dataset": "microsoft_msqa",
            "split": _EVALUATION_SPLIT_NAME,
            "adapter_contract_version": _ADAPTER_CONTRACT_VERSION,
            "question_id": self.question_id,
            "answer_id": self.answer_id,
            "question": self.question,
            "answer": self.answer,
            "source_url": self.source_url,
            "metadata": {
                "source_split": self.source_split,
                "tags": self.tags,
                "is_azure": self.is_azure,
                "is_m365": self.is_m365,
                "is_other": self.is_other,
                "is_short": self.is_short,
                "is_long": self.is_long,
                "source_row_index": self.source_row_index,
            },
        }


@dataclass(frozen=True)
class MsqaLeakageMatch:
    """One exact or near-duplicate leakage match."""

    msqa_question_id: str
    msqa_source_split: str
    development_source: str
    development_split: str
    development_question_id: str
    similarity: float

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "msqa_question_id": self.msqa_question_id,
            "msqa_source_split": self.msqa_source_split,
            "development_source": self.development_source,
            "development_split": self.development_split,
            "development_question_id": self.development_question_id,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class MsqaEvaluationSplitVisualization:
    """One generated Stage 57 MSQA split visualization."""

    name: str
    path: str


def build_msqa_evaluation_split_report(
    *,
    msqa_csv_path: Path,
    primeqa_train_questions_path: Path,
    primeqa_dev_questions_path: Path,
    near_duplicate_threshold: float = _DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Build the Stage 57 MSQA adapter contract, leakage audit, and split report."""

    for path in [msqa_csv_path, primeqa_train_questions_path, primeqa_dev_questions_path]:
        _ensure_file(path)
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    rows, rejected_contract_rows = _load_msqa_rows(msqa_csv_path)
    development_questions = _load_primeqa_leakage_questions(
        train_path=primeqa_train_questions_path,
        dev_path=primeqa_dev_questions_path,
    )
    leakage = _audit_primeqa_leakage(
        msqa_rows=rows,
        development_questions=development_questions,
        near_duplicate_threshold=near_duplicate_threshold,
        sample_limit=sample_limit,
    )
    split = _freeze_project_split(
        rows=rows,
        leaked_question_ids=set(leakage["leaked_question_ids"]),
        rejected_contract_rows=rejected_contract_rows,
    )
    return {
        "stage": "Stage 57",
        "created_at": "2026-07-14",
        "analysis_scope": (
            "MSQA adapter contract, near-duplicate leakage audit, and project-owned "
            "evaluation split freeze. This report does not run RAG answer-quality "
            "metrics, does not tune Stage 51, and does not change runtime defaults."
        ),
        "source_files": {
            "msqa_csv": _fingerprint(msqa_csv_path),
            "primeqa_train_questions": _fingerprint(primeqa_train_questions_path),
            "primeqa_dev_questions": _fingerprint(primeqa_dev_questions_path),
        },
        "adapter_contract": _adapter_contract(),
        "row_loading": {
            "loaded_rows": len(rows),
            "rejected_contract_rows": rejected_contract_rows,
        },
        "primeqa_leakage_audit": _leakage_report_without_ids(leakage),
        "frozen_split": split,
        "readiness": _readiness(leakage=leakage, split=split),
        "next_stage": (
            "Stage 58: run MSQA top-k baseline only on the frozen project split, "
            "without changing the default runtime"
        ),
    }


def write_msqa_project_split_jsonl(
    *,
    report: Mapping[str, Any],
    msqa_csv_path: Path,
    output_path: Path,
) -> None:
    """Write the frozen Stage 57 MSQA project split as local ignored JSONL."""

    selected_ids = set(report["frozen_split"]["selected_question_ids"])
    rows, _ = _load_msqa_rows(msqa_csv_path)
    selected_rows = [row for row in rows if row.question_id in selected_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected_rows:
            handle.write(_json_dumps(row.to_split_sample()))
            handle.write("\n")


def write_msqa_evaluation_split_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[MsqaEvaluationSplitVisualization]:
    """Write SVG charts for the Stage 57 MSQA split freeze."""

    output_dir.mkdir(parents=True, exist_ok=True)
    leakage_counts = report["primeqa_leakage_audit"]["counts"]
    split = report["frozen_split"]
    filter_counts = split["filter_counts"]
    charts = {
        "stage57_msqa_leakage_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage 57 MSQA PrimeQA leakage audit",
            bars=[
                BarDatum(
                    label="MSQA rows",
                    value=float(leakage_counts["msqa_questions"]),
                    value_label=str(leakage_counts["msqa_questions"]),
                ),
                BarDatum(
                    label="exact overlaps",
                    value=float(leakage_counts["exact_overlap_count"]),
                    value_label=str(leakage_counts["exact_overlap_count"]),
                ),
                BarDatum(
                    label="near duplicates",
                    value=float(leakage_counts["near_duplicate_overlap_count"]),
                    value_label=str(leakage_counts["near_duplicate_overlap_count"]),
                ),
                BarDatum(
                    label="without detected overlap",
                    value=float(
                        leakage_counts["msqa_questions_without_detected_overlap"]
                    ),
                    value_label=str(
                        leakage_counts["msqa_questions_without_detected_overlap"]
                    ),
                ),
            ],
            x_label="question count",
            margin_left=280,
        ),
        "stage57_msqa_split_filter_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage 57 MSQA project split filters",
            bars=[
                BarDatum(
                    label="source split candidates",
                    value=float(filter_counts["source_split_candidates"]),
                    value_label=str(filter_counts["source_split_candidates"]),
                ),
                BarDatum(
                    label="invalid source URL",
                    value=float(filter_counts["excluded_invalid_source_url"]),
                    value_label=str(filter_counts["excluded_invalid_source_url"]),
                ),
                BarDatum(
                    label="internal duplicates",
                    value=float(
                        filter_counts["excluded_internal_normalized_duplicates"]
                    ),
                    value_label=str(
                        filter_counts["excluded_internal_normalized_duplicates"]
                    ),
                ),
                BarDatum(
                    label="PrimeQA leakage",
                    value=float(filter_counts["excluded_primeqa_leakage"]),
                    value_label=str(filter_counts["excluded_primeqa_leakage"]),
                ),
                BarDatum(
                    label="selected",
                    value=float(filter_counts["selected_question_count"]),
                    value_label=str(filter_counts["selected_question_count"]),
                ),
            ],
            x_label="row count",
            margin_left=300,
        ),
        "stage57_msqa_selected_domain_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage 57 selected MSQA domain flags",
            bars=[
                BarDatum(
                    label="IsAzure=True",
                    value=float(split["selected_domain_counts"]["is_azure_true"]),
                    value_label=str(split["selected_domain_counts"]["is_azure_true"]),
                ),
                BarDatum(
                    label="IsM365=True",
                    value=float(split["selected_domain_counts"]["is_m365_true"]),
                    value_label=str(split["selected_domain_counts"]["is_m365_true"]),
                ),
                BarDatum(
                    label="IsOther=True",
                    value=float(split["selected_domain_counts"]["is_other_true"]),
                    value_label=str(split["selected_domain_counts"]["is_other_true"]),
                ),
            ],
            x_label="selected row count",
            margin_left=220,
        ),
        "stage57_msqa_adapter_field_coverage.svg": render_horizontal_bar_chart_svg(
            title="Stage 57 MSQA adapter field coverage",
            bars=[
                BarDatum(
                    label="QuestionText",
                    value=float(report["row_loading"]["loaded_rows"]),
                    value_label=str(report["row_loading"]["loaded_rows"]),
                ),
                BarDatum(
                    label="ProcessedAnswerText",
                    value=float(report["row_loading"]["loaded_rows"]),
                    value_label=str(report["row_loading"]["loaded_rows"]),
                ),
                BarDatum(
                    label="Url",
                    value=float(report["row_loading"]["loaded_rows"]),
                    value_label=str(report["row_loading"]["loaded_rows"]),
                ),
            ],
            x_label="loaded row count",
            margin_left=260,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(MsqaEvaluationSplitVisualization(name=filename, path=str(path)))
    return artifacts


def _load_msqa_rows(msqa_csv_path: Path) -> tuple[list[MsqaEvaluationRow], dict[str, int]]:
    csv.field_size_limit(_CSV_FIELD_SIZE_LIMIT)
    rows: list[MsqaEvaluationRow] = []
    rejected = Counter()
    with msqa_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            missing_fields = [
                field
                for field in _CONTRACT_REQUIRED_FIELDS
                if not str(row.get(field) or "").strip()
            ]
            if missing_fields:
                for field in missing_fields:
                    rejected[f"missing_{field}"] += 1
                continue
            question = str(row["QuestionText"]).strip()
            normalized_question = normalize_question_text(question)
            if not normalized_question:
                rejected["empty_normalized_question"] += 1
                continue
            rows.append(
                MsqaEvaluationRow(
                    question_id=str(row["QuestionId"]).strip(),
                    answer_id=str(row["AnswerId"]).strip(),
                    source_split=str(row["Split"]).strip(),
                    question=question,
                    answer=str(row["ProcessedAnswerText"]).strip(),
                    source_url=str(row["Url"]).strip(),
                    tags=str(row.get("Tags") or "").strip(),
                    is_azure=str(row.get("IsAzure") or "").strip(),
                    is_m365=str(row.get("IsM365") or "").strip(),
                    is_other=str(row.get("IsOther") or "").strip(),
                    is_short=str(row.get("isShort") or "").strip(),
                    is_long=str(row.get("isLong") or "").strip(),
                    normalized_question=normalized_question,
                    source_row_index=index,
                )
            )
    return rows, dict(sorted(rejected.items()))


def _load_primeqa_leakage_questions(
    *,
    train_path: Path,
    dev_path: Path,
) -> list[LeakageQuestion]:
    questions = []
    for split, path in (("train", train_path), ("dev", dev_path)):
        for question in load_primeqa_questions(path):
            questions.append(
                LeakageQuestion(
                    source="PrimeQA/TechQA",
                    split=split,
                    question_id=question.id,
                    question_text=question.full_question,
                )
            )
    return questions


def _audit_primeqa_leakage(
    *,
    msqa_rows: Sequence[MsqaEvaluationRow],
    development_questions: Sequence[LeakageQuestion],
    near_duplicate_threshold: float,
    sample_limit: int,
) -> dict[str, Any]:
    development_rows = [_normalized_development_row(question) for question in development_questions]
    development_by_normalized: dict[str, list[dict[str, Any]]] = {}
    for row in development_rows:
        development_by_normalized.setdefault(row["normalized_question"], []).append(row)
    exact_matches: list[MsqaLeakageMatch] = []
    exact_question_ids: set[str] = set()
    for msqa_row in msqa_rows:
        for development_row in development_by_normalized.get(msqa_row.normalized_question, []):
            exact_matches.append(
                _match(
                    msqa_row=msqa_row,
                    development_row=development_row,
                    similarity=1.0,
                )
            )
            exact_question_ids.add(msqa_row.question_id)

    near_matches = _near_duplicate_matches(
        msqa_rows=[
            row for row in msqa_rows if row.question_id not in exact_question_ids
        ],
        development_rows=development_rows,
        threshold=near_duplicate_threshold,
    )
    near_question_ids = {match.msqa_question_id for match in near_matches}
    leaked_question_ids = exact_question_ids | near_question_ids
    return {
        "near_duplicate_method": "token_jaccard_with_development_token_inverted_index",
        "normalization": "lowercase_ascii_alnum_whitespace",
        "near_duplicate_threshold": near_duplicate_threshold,
        "development_source_counts": _development_source_counts(development_rows),
        "counts": {
            "msqa_questions": len(msqa_rows),
            "development_questions": len(development_rows),
            "exact_overlap_count": len(exact_question_ids),
            "exact_overlap_pair_count": len(exact_matches),
            "near_duplicate_overlap_count": len(near_question_ids),
            "near_duplicate_overlap_pair_count": len(near_matches),
            "unhandled_overlap_count": len(leaked_question_ids),
            "msqa_questions_without_detected_overlap": (
                len(msqa_rows) - len(leaked_question_ids)
            ),
        },
        "leaked_question_ids": sorted(leaked_question_ids, key=_numeric_sort_key),
        "exact_overlap_samples": [
            match.to_report_dict() for match in exact_matches[:sample_limit]
        ],
        "near_duplicate_overlap_samples": [
            match.to_report_dict() for match in near_matches[:sample_limit]
        ],
    }


def _near_duplicate_matches(
    *,
    msqa_rows: Sequence[MsqaEvaluationRow],
    development_rows: Sequence[dict[str, Any]],
    threshold: float,
) -> list[MsqaLeakageMatch]:
    inverted_index: dict[str, list[int]] = {}
    for index, row in enumerate(development_rows):
        for token in row["tokens"]:
            inverted_index.setdefault(token, []).append(index)

    matches: list[MsqaLeakageMatch] = []
    for msqa_row in msqa_rows:
        left_count = len(msqa_row.tokens)
        min_right_count = math_ceil(threshold * left_count)
        max_right_count = math_floor(left_count / threshold) if threshold else left_count
        candidate_intersections: Counter[int] = Counter()
        for token in msqa_row.tokens:
            for index in inverted_index.get(token, []):
                right_count = len(development_rows[index]["tokens"])
                if right_count < min_right_count or right_count > max_right_count:
                    continue
                candidate_intersections[index] += 1
        best_row = None
        best_similarity = 0.0
        for index, intersection_count in candidate_intersections.items():
            development_row = development_rows[index]
            right_count = len(development_row["tokens"])
            min_intersection = math_ceil(
                threshold * (left_count + right_count) / (1 + threshold)
            )
            if intersection_count < min_intersection:
                continue
            similarity = _jaccard_from_intersection(
                left_count=left_count,
                right_count=right_count,
                intersection_count=intersection_count,
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_row = development_row
        if best_row is not None and best_similarity >= threshold:
            matches.append(
                _match(
                    msqa_row=msqa_row,
                    development_row=best_row,
                    similarity=round(best_similarity, 4),
                )
            )
    return matches


def _freeze_project_split(
    *,
    rows: Sequence[MsqaEvaluationRow],
    leaked_question_ids: set[str],
    rejected_contract_rows: Mapping[str, int],
) -> dict[str, Any]:
    source_split_rows = [
        row for row in rows if row.source_split.lower() == _SOURCE_SPLIT_FOR_EVALUATION
    ]
    duplicate_normalized_questions = _duplicate_normalized_question_ids(source_split_rows)
    selected_rows = []
    excluded_invalid_url = 0
    excluded_internal_duplicates = 0
    excluded_leakage = 0
    for row in source_split_rows:
        if not row.source_url.startswith(_SOURCE_URL_PREFIX):
            excluded_invalid_url += 1
            continue
        if row.question_id in duplicate_normalized_questions:
            excluded_internal_duplicates += 1
            continue
        if row.question_id in leaked_question_ids:
            excluded_leakage += 1
            continue
        selected_rows.append(row)

    selected_ids = [row.question_id for row in selected_rows]
    selected_id_hash = _sha256_lines(selected_ids)
    return {
        "split_name": _EVALUATION_SPLIT_NAME,
        "split_protocol_version": _SPLIT_PROTOCOL_VERSION,
        "source_split_used": _SOURCE_SPLIT_FOR_EVALUATION,
        "selection_rule": (
            "Use MSQA rows whose CSV Split is 'test', then exclude rows with invalid "
            "row-level Microsoft Learn Q&A URLs, internal normalized-question "
            "duplicates, or detected PrimeQA exact/near-duplicate leakage."
        ),
        "filter_counts": {
            "loaded_contract_rows": len(rows),
            "source_split_candidates": len(source_split_rows),
            "excluded_invalid_source_url": excluded_invalid_url,
            "excluded_internal_normalized_duplicates": excluded_internal_duplicates,
            "excluded_primeqa_leakage": excluded_leakage,
            "rejected_contract_rows": dict(rejected_contract_rows),
            "selected_question_count": len(selected_rows),
        },
        "selected_domain_counts": _selected_domain_counts(selected_rows),
        "selected_question_ids": selected_ids,
        "selected_question_ids_sha256": selected_id_hash,
        "first_selected_question_ids": selected_ids[:10],
        "last_selected_question_ids": selected_ids[-10:],
    }


def _adapter_contract() -> dict[str, Any]:
    return {
        "contract_version": _ADAPTER_CONTRACT_VERSION,
        "dataset": "Microsoft Q&A (MSQA)",
        "sample_id_field": "QuestionId",
        "question_field": "QuestionText",
        "answer_field": "ProcessedAnswerText",
        "source_url_field": "Url",
        "source_split_field": "Split",
        "metadata_fields": [
            "AnswerId",
            "Tags",
            "IsAzure",
            "IsM365",
            "IsOther",
            "isShort",
            "isLong",
        ],
        "no_fallback_policy": (
            "Do not fall back to AnswerText or DoubleProcessedAnswerText. "
            "ProcessedAnswerText is the only answer field approved by this "
            "contract because Stage 56 found it present for every local row."
        ),
        "citation_boundary": (
            "Use the row-level Microsoft Learn Q&A Url as the source URL. Do not "
            "claim complete documentation-span citation coverage from answer links."
        ),
        "unsupported_evaluation_modes": [
            "native_unanswerable_refusal_evaluation",
            "document_span_exact_citation_evaluation",
        ],
    }


def _readiness(*, leakage: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    selected_count = split["filter_counts"]["selected_question_count"]
    leakage_count = leakage["counts"]["unhandled_overlap_count"]
    blockers = []
    if selected_count <= 0:
        blockers.append("project_owned_split_is_empty")
    return {
        "status": "msqa_split_frozen_for_baseline_evaluation"
        if not blockers
        else "blocked_before_msqa_baseline",
        "can_run_msqa_topk_baseline_next": not blockers,
        "can_run_stage51_comparison_now": False,
        "can_defaultize_runtime_now": False,
        "default_runtime_policy": "unchanged",
        "passed_checks": [
            "adapter_contract_defined_without_field_fallback",
            "near_duplicate_leakage_audit_completed",
            "project_owned_evaluation_split_frozen",
        ],
        "remaining_before_stage51_comparison": [
            "run_topk_baseline_on_frozen_msqa_split",
            "review_msqa_baseline_failure_modes",
            "then_run_stage51_candidate_once_against_the_same_frozen_split",
        ],
        "blocking_issues": blockers,
        "decision": (
            f"Stage 57 detected {leakage_count} MSQA rows with PrimeQA exact or "
            f"near-duplicate overlap and froze {selected_count} safe source-split "
            "rows for the next top-k baseline. No runtime policy changes are "
            "approved in this stage."
        ),
    }


def _leakage_report_without_ids(leakage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in leakage.items()
        if key != "leaked_question_ids"
    } | {
        "leaked_question_id_count": len(leakage["leaked_question_ids"]),
        "first_leaked_question_ids": leakage["leaked_question_ids"][:10],
    }


def _normalized_development_row(question: LeakageQuestion) -> dict[str, Any]:
    normalized = normalize_question_text(question.question_text)
    return {
        "source": question.source,
        "split": question.split,
        "question_id": question.question_id,
        "normalized_question": normalized,
        "tokens": frozenset(normalized.split()),
    }


def _match(
    *,
    msqa_row: MsqaEvaluationRow,
    development_row: Mapping[str, Any],
    similarity: float,
) -> MsqaLeakageMatch:
    return MsqaLeakageMatch(
        msqa_question_id=msqa_row.question_id,
        msqa_source_split=msqa_row.source_split,
        development_source=str(development_row["source"]),
        development_split=str(development_row["split"]),
        development_question_id=str(development_row["question_id"]),
        similarity=similarity,
    )


def _development_source_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[f"{row['source']}::{row['split']}"] += 1
    return dict(sorted(counts.items()))


def _duplicate_normalized_question_ids(
    rows: Sequence[MsqaEvaluationRow],
) -> set[str]:
    ids_by_normalized: dict[str, list[str]] = {}
    for row in rows:
        ids_by_normalized.setdefault(row.normalized_question, []).append(row.question_id)
    duplicate_ids = set()
    for question_ids in ids_by_normalized.values():
        if len(question_ids) > 1:
            duplicate_ids.update(question_ids)
    return duplicate_ids


def _selected_domain_counts(rows: Sequence[MsqaEvaluationRow]) -> dict[str, int]:
    return {
        "is_azure_true": sum(1 for row in rows if row.is_azure == "True"),
        "is_m365_true": sum(1 for row in rows if row.is_m365 == "True"),
        "is_other_true": sum(1 for row in rows if row.is_other == "True"),
        "is_short_true": sum(1 for row in rows if row.is_short == "True"),
        "is_long_true": sum(1 for row in rows if row.is_long == "True"),
    }


def _jaccard_from_intersection(
    *,
    left_count: int,
    right_count: int,
    intersection_count: int,
) -> float:
    union_count = left_count + right_count - intersection_count
    if union_count == 0:
        return 1.0
    return intersection_count / union_count


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _sha256_lines(values: Sequence[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def _numeric_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _json_dumps(value: Mapping[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def math_ceil(value: float) -> int:
    return int(-(-value // 1))


def math_floor(value: float) -> int:
    return int(value // 1)
