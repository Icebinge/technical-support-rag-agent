from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class CandidateRerankerQuestionAudit:
    """Question-level audit values for one candidate pool."""

    split: str
    question_id: str
    question_route: str
    candidate_count: int
    gold_document_candidate_count: int
    top_candidate_token_f1: float
    best_candidate_token_f1: float
    best_candidate_rank: int | None
    oracle_gain_vs_top_candidate: float


@dataclass(frozen=True)
class F1DistributionBucket:
    """Binned F1 label distribution."""

    label: str
    lower_bound: float
    upper_bound: float
    count: int
    rate: float


@dataclass(frozen=True)
class RankDistributionBucket:
    """Binned best-candidate rank distribution."""

    label: str
    count: int
    rate: float


@dataclass(frozen=True)
class RouteOracleGainSummary:
    """Route-level oracle gain audit."""

    question_route: str
    question_count: int
    average_candidate_count: float
    average_top_candidate_token_f1: float
    average_best_candidate_token_f1: float
    average_oracle_gain_vs_top_candidate: float
    median_oracle_gain_vs_top_candidate: float
    positive_oracle_gain_question_count: int
    positive_oracle_gain_rate: float
    gold_document_candidate_question_count: int
    gold_document_candidate_rate: float


@dataclass(frozen=True)
class SplitAuditSummary:
    """Split-level distribution and oracle gain audit."""

    split: str
    question_count: int
    row_count: int
    average_candidate_count: float
    average_top_candidate_token_f1: float
    average_best_candidate_token_f1: float
    average_oracle_gain_vs_top_candidate: float
    positive_oracle_gain_question_count: int
    positive_oracle_gain_rate: float
    best_rank_1_question_count: int
    best_rank_1_rate: float
    gold_document_candidate_question_count: int
    gold_document_candidate_rate: float


@dataclass(frozen=True)
class FeatureLeakageAudit:
    """Key-based audit for accidental gold-label leakage into runtime features."""

    runtime_feature_keys: list[str]
    gold_label_keys: list[str]
    metadata_keys: list[str]
    suspicious_runtime_feature_keys: list[str]
    text_like_runtime_feature_keys: list[str]
    non_scalar_runtime_feature_keys: list[str]
    label_leakage_detected_from_keys: bool
    audit_scope: str


@dataclass(frozen=True)
class DatasetConsistencyAudit:
    """Consistency checks between row JSONL and Stage 31 summary JSON."""

    summary_total_rows: int
    actual_total_rows: int
    total_rows_match: bool
    summary_total_questions: int
    actual_question_summary_count: int
    row_question_count: int
    total_questions_match: bool
    row_questions_without_summary_count: int
    summary_questions_without_rows_count: int
    row_questions_without_summary: list[str]
    summary_questions_without_rows: list[str]
    rows_by_split_match: bool
    rows_by_route_match: bool


@dataclass(frozen=True)
class CandidateRerankerDatasetAudit:
    """Full candidate-reranker dataset audit result."""

    total_rows: int
    total_questions: int
    rows_by_split: dict[str, int]
    rows_by_route: dict[str, int]
    questions_by_split: dict[str, int]
    questions_by_route: dict[str, int]
    candidate_token_f1_distribution: list[F1DistributionBucket]
    best_candidate_token_f1_distribution: list[F1DistributionBucket]
    best_candidate_rank_distribution: list[RankDistributionBucket]
    route_oracle_gain: list[RouteOracleGainSummary]
    split_summaries: list[SplitAuditSummary]
    feature_leakage_audit: FeatureLeakageAudit
    consistency_audit: DatasetConsistencyAudit


@dataclass(frozen=True)
class VisualizationArtifact:
    """One generated visualization file."""

    name: str
    path: str


F1_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.00-0.05", 0.0, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    ("0.20-0.40", 0.20, 0.40),
    ("0.40-0.60", 0.40, 0.60),
    ("0.60-0.80", 0.60, 0.80),
    ("0.80-1.00", 0.80, 1.01),
)
RANK_BUCKET_LABELS = (
    "rank_1",
    "rank_2",
    "rank_3",
    "rank_4_5",
    "rank_6_10",
    "rank_11_25",
    "missing",
)
SUSPICIOUS_RUNTIME_KEY_TERMS = (
    "gold",
    "label",
    "f1",
    "best_candidate",
    "oracle",
    "target",
    "ground_truth",
    "answer_doc",
    "answer_document",
)
RAW_TEXT_RUNTIME_KEYS = (
    "candidate_sentence",
    "sentence",
    "text",
    "raw_text",
    "document_text",
    "document_title",
    "document_id",
    "question_text",
    "question_title",
    "question_id",
)


def load_candidate_reranker_rows(path: Path) -> list[dict[str, Any]]:
    """Load candidate-reranker rows from a JSONL dataset file."""

    rows = []
    with path.open(encoding="utf-8") as row_file:
        for line_number, line in enumerate(row_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row at line {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at line {line_number} must be an object")
            rows.append(row)
    return rows


def load_candidate_reranker_summary(path: Path) -> dict[str, Any]:
    """Load the Stage 31 candidate-reranker summary JSON."""

    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("candidate reranker summary must be a JSON object")
    return summary


def audit_candidate_reranker_dataset(
    rows: Sequence[Mapping[str, Any]],
    summary_report: Mapping[str, Any],
) -> CandidateRerankerDatasetAudit:
    """Audit label distribution, route gains, split differences, and leakage risk."""

    if not rows:
        raise ValueError("rows must not be empty")

    question_audits = _load_question_audits(summary_report)
    rows_by_split = Counter(str(row["split"]) for row in rows)
    rows_by_route = Counter(_row_route(row) for row in rows)
    questions_by_split = Counter(audit.split for audit in question_audits)
    questions_by_route = Counter(audit.question_route for audit in question_audits)

    return CandidateRerankerDatasetAudit(
        total_rows=len(rows),
        total_questions=len(question_audits),
        rows_by_split=dict(sorted(rows_by_split.items())),
        rows_by_route=dict(sorted(rows_by_route.items())),
        questions_by_split=dict(sorted(questions_by_split.items())),
        questions_by_route=dict(sorted(questions_by_route.items())),
        candidate_token_f1_distribution=_build_f1_distribution(
            [_candidate_token_f1(row) for row in rows]
        ),
        best_candidate_token_f1_distribution=_build_f1_distribution(
            [audit.best_candidate_token_f1 for audit in question_audits]
        ),
        best_candidate_rank_distribution=_build_rank_distribution(question_audits),
        route_oracle_gain=_summarize_route_oracle_gain(question_audits),
        split_summaries=_summarize_splits(rows=rows, question_audits=question_audits),
        feature_leakage_audit=_audit_feature_leakage(rows),
        consistency_audit=_audit_consistency(
            rows=rows,
            summary_report=summary_report,
            question_audits=question_audits,
        ),
    )


def candidate_reranker_dataset_audit_to_dict(
    audit: CandidateRerankerDatasetAudit,
) -> dict[str, Any]:
    """Convert an audit result to a JSON-safe dictionary."""

    return asdict(audit)


def write_audit_visualizations(
    audit: CandidateRerankerDatasetAudit,
    output_dir: Path,
) -> list[VisualizationArtifact]:
    """Write SVG visualizations for the main audit distributions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "candidate_label_f1_distribution.svg": _render_bar_chart_svg(
            title="Candidate token F1 label distribution",
            bars=[
                _Bar(
                    label=bucket.label,
                    value=float(bucket.count),
                    value_label=f"{bucket.count} ({_format_rate(bucket.rate)})",
                )
                for bucket in audit.candidate_token_f1_distribution
            ],
            x_label="candidate rows",
        ),
        "best_candidate_rank_distribution.svg": _render_bar_chart_svg(
            title="Best candidate rank distribution",
            bars=[
                _Bar(
                    label=bucket.label.replace("_", " "),
                    value=float(bucket.count),
                    value_label=f"{bucket.count} ({_format_rate(bucket.rate)})",
                )
                for bucket in audit.best_candidate_rank_distribution
            ],
            x_label="questions",
        ),
        "route_oracle_gain.svg": _render_bar_chart_svg(
            title="Average oracle gain by question route",
            bars=[
                _Bar(
                    label=summary.question_route,
                    value=summary.average_oracle_gain_vs_top_candidate,
                    value_label=(
                        f"{summary.average_oracle_gain_vs_top_candidate:+.4f} "
                        f"(n={summary.question_count})"
                    ),
                )
                for summary in audit.route_oracle_gain
            ],
            x_label="average F1 gain",
        ),
        "split_oracle_gain.svg": _render_bar_chart_svg(
            title="Average oracle gain by split",
            bars=[
                _Bar(
                    label=summary.split,
                    value=summary.average_oracle_gain_vs_top_candidate,
                    value_label=(
                        f"{summary.average_oracle_gain_vs_top_candidate:+.4f} "
                        f"(n={summary.question_count})"
                    ),
                )
                for summary in audit.split_summaries
            ],
            x_label="average F1 gain",
        ),
    }

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(VisualizationArtifact(name=filename, path=str(path)))
    return artifacts


def _load_question_audits(
    summary_report: Mapping[str, Any],
) -> list[CandidateRerankerQuestionAudit]:
    raw_summaries = summary_report.get("question_summaries")
    if not isinstance(raw_summaries, list):
        raise ValueError("summary_report must contain a list field named question_summaries")

    audits = []
    for raw_summary in raw_summaries:
        if not isinstance(raw_summary, Mapping):
            raise ValueError("each question summary must be an object")
        audits.append(
            CandidateRerankerQuestionAudit(
                split=str(raw_summary["split"]),
                question_id=str(raw_summary["question_id"]),
                question_route=str(raw_summary["question_route"]),
                candidate_count=int(raw_summary["candidate_count"]),
                gold_document_candidate_count=int(
                    raw_summary["gold_document_candidate_count"]
                ),
                top_candidate_token_f1=float(raw_summary["top_candidate_token_f1"]),
                best_candidate_token_f1=float(raw_summary["best_candidate_token_f1"]),
                best_candidate_rank=_optional_int(raw_summary["best_candidate_rank"]),
                oracle_gain_vs_top_candidate=float(
                    raw_summary["oracle_gain_vs_top_candidate"]
                ),
            )
        )
    return audits


def _build_f1_distribution(values: Sequence[float]) -> list[F1DistributionBucket]:
    total = len(values)
    counts = Counter(_f1_bucket_label(value) for value in values)
    return [
        F1DistributionBucket(
            label=label,
            lower_bound=lower_bound,
            upper_bound=upper_bound if upper_bound <= 1 else 1.0,
            count=counts[label],
            rate=_ratio(counts[label], total),
        )
        for label, lower_bound, upper_bound in F1_BUCKETS
    ]


def _build_rank_distribution(
    question_audits: Sequence[CandidateRerankerQuestionAudit],
) -> list[RankDistributionBucket]:
    total = len(question_audits)
    counts = Counter(_rank_bucket_label(audit.best_candidate_rank) for audit in question_audits)
    return [
        RankDistributionBucket(
            label=label,
            count=counts[label],
            rate=_ratio(counts[label], total),
        )
        for label in RANK_BUCKET_LABELS
    ]


def _summarize_route_oracle_gain(
    question_audits: Sequence[CandidateRerankerQuestionAudit],
) -> list[RouteOracleGainSummary]:
    audits_by_route: dict[str, list[CandidateRerankerQuestionAudit]] = defaultdict(list)
    for audit in question_audits:
        audits_by_route[audit.question_route].append(audit)

    summaries = [
        RouteOracleGainSummary(
            question_route=route,
            question_count=len(route_audits),
            average_candidate_count=_rounded_mean(
                [audit.candidate_count for audit in route_audits]
            ),
            average_top_candidate_token_f1=_rounded_mean(
                [audit.top_candidate_token_f1 for audit in route_audits]
            ),
            average_best_candidate_token_f1=_rounded_mean(
                [audit.best_candidate_token_f1 for audit in route_audits]
            ),
            average_oracle_gain_vs_top_candidate=_rounded_mean(
                [audit.oracle_gain_vs_top_candidate for audit in route_audits]
            ),
            median_oracle_gain_vs_top_candidate=_rounded_median(
                [audit.oracle_gain_vs_top_candidate for audit in route_audits]
            ),
            positive_oracle_gain_question_count=sum(
                audit.oracle_gain_vs_top_candidate > 0 for audit in route_audits
            ),
            positive_oracle_gain_rate=_ratio(
                sum(audit.oracle_gain_vs_top_candidate > 0 for audit in route_audits),
                len(route_audits),
            ),
            gold_document_candidate_question_count=sum(
                audit.gold_document_candidate_count > 0 for audit in route_audits
            ),
            gold_document_candidate_rate=_ratio(
                sum(audit.gold_document_candidate_count > 0 for audit in route_audits),
                len(route_audits),
            ),
        )
        for route, route_audits in audits_by_route.items()
    ]
    return sorted(
        summaries,
        key=lambda summary: (
            summary.average_oracle_gain_vs_top_candidate,
            summary.question_count,
        ),
        reverse=True,
    )


def _summarize_splits(
    rows: Sequence[Mapping[str, Any]],
    question_audits: Sequence[CandidateRerankerQuestionAudit],
) -> list[SplitAuditSummary]:
    rows_by_split = Counter(str(row["split"]) for row in rows)
    audits_by_split: dict[str, list[CandidateRerankerQuestionAudit]] = defaultdict(list)
    for audit in question_audits:
        audits_by_split[audit.split].append(audit)

    summaries = []
    for split, split_audits in audits_by_split.items():
        positive_gain_count = sum(
            audit.oracle_gain_vs_top_candidate > 0 for audit in split_audits
        )
        best_rank_1_count = sum(audit.best_candidate_rank == 1 for audit in split_audits)
        gold_candidate_count = sum(
            audit.gold_document_candidate_count > 0 for audit in split_audits
        )
        summaries.append(
            SplitAuditSummary(
                split=split,
                question_count=len(split_audits),
                row_count=rows_by_split[split],
                average_candidate_count=_rounded_mean(
                    [audit.candidate_count for audit in split_audits]
                ),
                average_top_candidate_token_f1=_rounded_mean(
                    [audit.top_candidate_token_f1 for audit in split_audits]
                ),
                average_best_candidate_token_f1=_rounded_mean(
                    [audit.best_candidate_token_f1 for audit in split_audits]
                ),
                average_oracle_gain_vs_top_candidate=_rounded_mean(
                    [audit.oracle_gain_vs_top_candidate for audit in split_audits]
                ),
                positive_oracle_gain_question_count=positive_gain_count,
                positive_oracle_gain_rate=_ratio(positive_gain_count, len(split_audits)),
                best_rank_1_question_count=best_rank_1_count,
                best_rank_1_rate=_ratio(best_rank_1_count, len(split_audits)),
                gold_document_candidate_question_count=gold_candidate_count,
                gold_document_candidate_rate=_ratio(
                    gold_candidate_count,
                    len(split_audits),
                ),
            )
        )
    return sorted(summaries, key=lambda summary: summary.split)


def _audit_feature_leakage(rows: Sequence[Mapping[str, Any]]) -> FeatureLeakageAudit:
    runtime_keys = _collect_nested_keys(rows, section="runtime_features")
    gold_label_keys = _collect_nested_keys(rows, section="gold_labels")
    metadata_keys = _collect_nested_keys(rows, section="metadata")
    suspicious_keys = [
        key
        for key in runtime_keys
        if any(term in key.lower() for term in SUSPICIOUS_RUNTIME_KEY_TERMS)
    ]
    text_like_keys = [key for key in runtime_keys if _is_text_like_runtime_key(key)]
    non_scalar_keys = sorted(
        {
            key
            for row in rows
            for key, value in _runtime_features(row).items()
            if not isinstance(value, (str, int, float, bool, type(None)))
        }
    )

    return FeatureLeakageAudit(
        runtime_feature_keys=runtime_keys,
        gold_label_keys=gold_label_keys,
        metadata_keys=metadata_keys,
        suspicious_runtime_feature_keys=suspicious_keys,
        text_like_runtime_feature_keys=text_like_keys,
        non_scalar_runtime_feature_keys=non_scalar_keys,
        label_leakage_detected_from_keys=bool(
            suspicious_keys or text_like_keys or non_scalar_keys
        ),
        audit_scope=(
            "Static key/value-shape audit only. It detects obvious leakage from "
            "runtime feature names or non-scalar values; it is not a statistical "
            "proof that every feature is causally safe."
        ),
    )


def _audit_consistency(
    rows: Sequence[Mapping[str, Any]],
    summary_report: Mapping[str, Any],
    question_audits: Sequence[CandidateRerankerQuestionAudit],
) -> DatasetConsistencyAudit:
    summary = summary_report.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("summary_report must contain an object field named summary")

    row_question_ids = {_question_key(row["split"], row["question_id"]) for row in rows}
    summary_question_ids = {
        _question_key(audit.split, audit.question_id) for audit in question_audits
    }
    row_questions_without_summary = sorted(row_question_ids - summary_question_ids)
    summary_questions_without_rows = sorted(summary_question_ids - row_question_ids)
    actual_rows_by_split = dict(Counter(str(row["split"]) for row in rows))
    actual_rows_by_route = dict(Counter(_row_route(row) for row in rows))

    return DatasetConsistencyAudit(
        summary_total_rows=int(summary["total_rows"]),
        actual_total_rows=len(rows),
        total_rows_match=int(summary["total_rows"]) == len(rows),
        summary_total_questions=int(summary["total_questions"]),
        actual_question_summary_count=len(question_audits),
        row_question_count=len(row_question_ids),
        total_questions_match=int(summary["total_questions"]) == len(question_audits),
        row_questions_without_summary_count=len(row_questions_without_summary),
        summary_questions_without_rows_count=len(summary_questions_without_rows),
        row_questions_without_summary=row_questions_without_summary[:20],
        summary_questions_without_rows=summary_questions_without_rows[:20],
        rows_by_split_match=dict(summary["rows_by_split"]) == actual_rows_by_split,
        rows_by_route_match=dict(summary["rows_by_route"]) == actual_rows_by_route,
    )


def _collect_nested_keys(
    rows: Sequence[Mapping[str, Any]],
    section: str,
) -> list[str]:
    keys = set()
    for row in rows:
        raw_section = row.get(section)
        if not isinstance(raw_section, Mapping):
            raise ValueError(f"row section must be an object: {section}")
        keys.update(str(key) for key in raw_section)
    return sorted(keys)


def _is_text_like_runtime_key(key: str) -> bool:
    normalized = key.lower()
    if normalized in RAW_TEXT_RUNTIME_KEYS:
        return True
    return normalized.endswith(("_sentence", "_text", "_title", "_id"))


def _runtime_features(row: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime_features = row.get("runtime_features")
    if not isinstance(runtime_features, Mapping):
        raise ValueError("row runtime_features must be an object")
    return runtime_features


def _row_route(row: Mapping[str, Any]) -> str:
    return str(_runtime_features(row)["question_route"])


def _candidate_token_f1(row: Mapping[str, Any]) -> float:
    gold_labels = row.get("gold_labels")
    if not isinstance(gold_labels, Mapping):
        raise ValueError("row gold_labels must be an object")
    return float(gold_labels["candidate_token_f1"])


def _f1_bucket_label(value: float) -> str:
    for label, lower_bound, upper_bound in F1_BUCKETS:
        if lower_bound <= value < upper_bound:
            return label
    raise ValueError(f"F1 value must be between 0 and 1: {value}")


def _rank_bucket_label(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    if 4 <= rank <= 5:
        return "rank_4_5"
    if 6 <= rank <= 10:
        return "rank_6_10"
    if 11 <= rank <= 25:
        return "rank_11_25"
    return "missing"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _question_key(split: Any, question_id: Any) -> str:
    return f"{split}::{question_id}"


def _rounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _rounded_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(median(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


@dataclass(frozen=True)
class _Bar:
    label: str
    value: float
    value_label: str


def _render_bar_chart_svg(
    title: str,
    bars: Sequence[_Bar],
    x_label: str,
) -> str:
    width = 960
    margin_left = 270
    margin_right = 190
    margin_top = 56
    margin_bottom = 56
    row_height = 38
    height = margin_top + margin_bottom + max(1, len(bars)) * row_height
    plot_width = width - margin_left - margin_right
    max_value = max((bar.value for bar in bars), default=0.0)
    scale_denominator = max_value if max_value > 0 else 1.0
    chart_id = _svg_id(title)

    bar_lines = []
    for index, bar in enumerate(bars):
        y = margin_top + index * row_height
        bar_width = int(round(plot_width * bar.value / scale_denominator))
        label_y = y + 21
        bar_lines.append(
            "\n".join(
                [
                    f'<text x="{margin_left - 12}" y="{label_y}" text-anchor="end">'
                    f"{escape(bar.label)}</text>",
                    (
                        f'<rect x="{margin_left}" y="{y + 6}" width="{bar_width}" '
                        'height="22" rx="3" fill="#2f6fed" />'
                    ),
                    f'<text x="{margin_left + bar_width + 8}" y="{label_y}">'
                    f"{escape(bar.value_label)}</text>",
                ]
            )
        )

    axis_y = height - margin_bottom + 12
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                f'aria-labelledby="{chart_id}-title {chart_id}-desc">'
            ),
            f'<title id="{chart_id}-title">{escape(title)}</title>',
            (
                f'<desc id="{chart_id}-desc">Horizontal bar chart showing '
                f'{escape(x_label)} for {len(bars)} categories.</desc>'
            ),
            '<rect width="100%" height="100%" fill="#ffffff" />',
            (
                '<style>text{font-family:Arial, sans-serif;font-size:13px;'
                'fill:#1f2937}.title{font-size:18px;font-weight:700}'
                '.axis{fill:#4b5563;font-size:12px}.grid{stroke:#e5e7eb}'
                '</style>'
            ),
            f'<text x="24" y="32" class="title">{escape(title)}</text>',
            (
                f'<line x1="{margin_left}" x2="{margin_left + plot_width}" '
                f'y1="{height - margin_bottom}" y2="{height - margin_bottom}" '
                'class="grid" />'
            ),
            *bar_lines,
            f'<text x="{margin_left}" y="{axis_y}" class="axis">{escape(x_label)}</text>',
            "</svg>",
        ]
    )


def _svg_id(title: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-")


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"
