from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.evidence_selection import classify_question_route
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)

STAGE160_DEV_SPLIT_FILENAME = "primeqa_hybrid_split_stage68_dev.jsonl"
STAGE160_EXPECTED_DEV_SHA256 = "071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f"
STAGE160_EXPECTED_DEV_ROW_COUNT = 121
STAGE160_EXPECTED_ORDER_SHA256 = "3b8a39cae397db4402080a2780178ade0fd4fc3a9ba5facb25d041510e8b69b7"
STAGE160_EXPECTED_GROUPING_SHA256 = (
    "7aa271a775c2926b32226e0a4fccc96cff3a7bf98fc90246c8002d79561fd6d0"
)
STAGE160_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
STAGE160_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
STAGE160_MAX_TURNS_PER_THREAD = 4
STAGE160_DIAGNOSTIC_FOLD_COUNT = 5
STAGE159_END_TO_END_P95_MS = 11835.247

_ANSWERABLE_REFUSAL_BUCKETS = (
    "answerable_refusal_gold_absent_candidate_pool",
    "answerable_refusal_gold_absent_generation_top10",
    "answerable_refusal_gold_visible_model_refused",
    "answerable_refusal_after_compose",
)


@dataclass(frozen=True)
class Stage160DiagnosticSample:
    private_identity_sha256: str
    query_digest_sha256: str
    diagnostic_group_sha256: str
    answerable: bool
    gold_answer: str
    gold_document_id: str | None
    gold_document_sha256: str | None
    question_route: str
    split_subtype: str
    runtime_query: PrimeQARuntimeQuery


@dataclass(frozen=True)
class Stage160DevDiagnosticSet:
    source_size_bytes: int
    source_sha256: str
    order_sha256: str
    samples: tuple[Stage160DiagnosticSample, ...]

    def public_summary(self) -> dict[str, Any]:
        answerable_count = sum(sample.answerable for sample in self.samples)
        return {
            "source_size_bytes": self.source_size_bytes,
            "source_sha256": self.source_sha256,
            "dev_row_count": len(self.samples),
            "answerable_count": answerable_count,
            "unanswerable_count": len(self.samples) - answerable_count,
            "stable_order_sha256": self.order_sha256,
            "assigned_split": "dev",
            "split_name": STAGE160_SPLIT_NAME,
            "protocol_version": STAGE160_PROTOCOL_VERSION,
            "gold_fields_used_for_diagnosis": True,
            "gold_fields_projected_into_runtime": False,
            "gold_fields_used_for_selection_or_tuning": False,
        }


@dataclass(frozen=True)
class Stage160ThreadWorkload:
    ordinal: int
    samples: tuple[Stage160DiagnosticSample, ...]


@dataclass(frozen=True)
class Stage160WorkloadPlan:
    threads: tuple[Stage160ThreadWorkload, ...]
    grouping_sha256: str

    @property
    def ordered_samples(self) -> tuple[Stage160DiagnosticSample, ...]:
        return tuple(sample for thread in self.threads for sample in thread.samples)

    def public_summary(self) -> dict[str, Any]:
        turn_position_counts = Counter(
            position for thread in self.threads for position in range(1, len(thread.samples) + 1)
        )
        return {
            "grouping_policy": "stage159_stable_hash_order_consecutive_groups_of_four",
            "semantic_relationship_policy": "synthetic_grouping_not_natural_conversation",
            "thread_count": len(self.threads),
            "full_four_turn_thread_count": sum(
                len(thread.samples) == STAGE160_MAX_TURNS_PER_THREAD for thread in self.threads
            ),
            "trailing_thread_turn_count": len(self.threads[-1].samples),
            "turn_count": len(self.ordered_samples),
            "turn_position_counts": {
                str(position): turn_position_counts[position]
                for position in range(1, STAGE160_MAX_TURNS_PER_THREAD + 1)
            },
            "grouping_sha256": self.grouping_sha256,
        }


@dataclass(frozen=True)
class Stage160FoldAssignment:
    fold_by_private_identity: Mapping[str, int]
    group_count: int
    row_counts: tuple[int, ...]
    group_counts: tuple[int, ...]
    assignment_sha256: str

    def public_summary(self) -> dict[str, Any]:
        return {
            "analysis_mode": "five_fold_grouped_diagnostic_stability_no_model_selection",
            "fold_count": len(self.row_counts),
            "group_key": "normalized_question_plus_answer_document_or_unanswerable",
            "group_count": self.group_count,
            "row_counts": {str(index): count for index, count in enumerate(self.row_counts)},
            "group_counts": {str(index): count for index, count in enumerate(self.group_counts)},
            "assignment_sha256": self.assignment_sha256,
            "fit_models": False,
            "select_policy": False,
            "tune_thresholds": False,
        }


@dataclass(frozen=True)
class Stage160CaseObservation:
    private_identity_sha256: str
    query_digest_sha256: str
    diagnostic_group_sha256: str
    gold_document_sha256: str | None
    fold_id: int
    thread_ordinal: int
    turn_position: int
    question_route: str
    split_subtype: str
    answerable: bool
    selected_action: str
    terminal_state: str
    refused: bool
    candidate_pool_count: int
    generation_context_count: int
    verification_context_count: int
    gold_candidate_rank: int | None
    gold_generation_rank: int | None
    gold_verification_rank: int | None
    gold_cited: bool
    citation_count: int
    answer_token_f1: float | None
    top_candidate_score: float | None
    gold_candidate_score: float | None
    router_input_token_count: int
    router_output_token_count: int
    router_generation_latency_ms: float
    end_to_end_latency_ms: float
    retained_state_bytes: int
    completed_turn_count: int

    @property
    def failure_bucket(self) -> str:
        if not self.answerable:
            return "unanswerable_correct_refusal" if self.refused else "unanswerable_false_answer"
        if self.refused:
            if self.gold_candidate_rank is None:
                return "answerable_refusal_gold_absent_candidate_pool"
            if self.gold_generation_rank is None:
                return "answerable_refusal_gold_absent_generation_top10"
            if self.selected_action == "refuse_insufficient_evidence":
                return "answerable_refusal_gold_visible_model_refused"
            return "answerable_refusal_after_compose"
        if self.gold_candidate_rank is None:
            return "answerable_answer_gold_absent_candidate_pool"
        if self.gold_generation_rank is None:
            return "answerable_answer_gold_absent_generation_top10"
        if self.gold_cited:
            return "answerable_answer_gold_cited"
        return "answerable_answer_gold_not_cited"

    @property
    def stage159_p95_exceeded(self) -> bool:
        return self.end_to_end_latency_ms > STAGE159_END_TO_END_P95_MS

    def to_private_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "failure_bucket": self.failure_bucket,
            "stage159_p95_exceeded": self.stage159_p95_exceeded,
        }


def load_stage160_dev_diagnostic_samples(
    path: Path,
    *,
    expected_sha256: str = STAGE160_EXPECTED_DEV_SHA256,
    expected_row_count: int = STAGE160_EXPECTED_DEV_ROW_COUNT,
) -> Stage160DevDiagnosticSet:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Stage160 dev source must be a file")
    source_sha256 = _file_sha256(resolved)
    if source_sha256 != expected_sha256:
        raise ValueError("Stage160 dev source SHA-256 does not match the frozen split")
    raw_samples = load_primeqa_hybrid_split_samples(resolved)
    if len(raw_samples) != expected_row_count:
        raise ValueError("Stage160 dev source row count does not match the protocol")
    samples = tuple(_diagnostic_sample(sample) for sample in raw_samples)
    if len({sample.private_identity_sha256 for sample in samples}) != len(samples):
        raise ValueError("Stage160 dev source contains duplicate private identities")
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.private_identity_sha256,
                sample.runtime_query.id,
            ),
        )
    )
    order_sha256 = _digest_lines(sample.private_identity_sha256 for sample in ordered)
    return Stage160DevDiagnosticSet(
        source_size_bytes=resolved.stat().st_size,
        source_sha256=source_sha256,
        order_sha256=order_sha256,
        samples=ordered,
    )


def build_stage160_workload_plan(
    diagnostic_set: Stage160DevDiagnosticSet,
) -> Stage160WorkloadPlan:
    if not diagnostic_set.samples:
        raise ValueError("Stage160 workload requires at least one sample")
    threads = tuple(
        Stage160ThreadWorkload(
            ordinal=(offset // STAGE160_MAX_TURNS_PER_THREAD) + 1,
            samples=diagnostic_set.samples[offset : offset + STAGE160_MAX_TURNS_PER_THREAD],
        )
        for offset in range(0, len(diagnostic_set.samples), STAGE160_MAX_TURNS_PER_THREAD)
    )
    grouping_sha256 = _digest_lines(
        f"{thread.ordinal}:{','.join(sample.private_identity_sha256 for sample in thread.samples)}"
        for thread in threads
    )
    return Stage160WorkloadPlan(threads=threads, grouping_sha256=grouping_sha256)


def build_stage160_grouped_fold_assignment(
    samples: Sequence[Stage160DiagnosticSample],
    *,
    fold_count: int = STAGE160_DIAGNOSTIC_FOLD_COUNT,
) -> Stage160FoldAssignment:
    if fold_count < 2:
        raise ValueError("Stage160 diagnostic fold count must be at least two")
    groups: dict[str, list[Stage160DiagnosticSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.diagnostic_group_sha256].append(sample)
    if fold_count > len(groups):
        raise ValueError("Stage160 diagnostic fold count exceeds group count")
    fold_rows: list[int] = [0] * fold_count
    fold_groups: list[int] = [0] * fold_count
    fold_by_identity: dict[str, int] = {}
    for _group_sha, group_samples in sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        fold_id = min(
            range(fold_count),
            key=lambda candidate: (
                fold_rows[candidate],
                fold_groups[candidate],
                candidate,
            ),
        )
        for sample in group_samples:
            fold_by_identity[sample.private_identity_sha256] = fold_id
        fold_rows[fold_id] += len(group_samples)
        fold_groups[fold_id] += 1
    assignment_sha256 = _digest_lines(
        f"{identity}:{fold_by_identity[identity]}" for identity in sorted(fold_by_identity)
    )
    return Stage160FoldAssignment(
        fold_by_private_identity=fold_by_identity,
        group_count=len(groups),
        row_counts=tuple(fold_rows),
        group_counts=tuple(fold_groups),
        assignment_sha256=assignment_sha256,
    )


def summarize_stage160_observations(
    observations: Sequence[Stage160CaseObservation],
) -> dict[str, Any]:
    if not observations:
        raise ValueError("Stage160 summary requires observations")
    ordered = tuple(
        sorted(observations, key=lambda item: (item.thread_ordinal, item.turn_position))
    )
    answerable = tuple(item for item in ordered if item.answerable)
    unanswerable = tuple(item for item in ordered if not item.answerable)
    refusals = tuple(item for item in ordered if item.refused)
    answerable_refusals = tuple(item for item in answerable if item.refused)
    complete_answerable = tuple(item for item in answerable if not item.refused)
    false_answers = tuple(item for item in unanswerable if not item.refused)
    failure_counts = Counter(item.failure_bucket for item in ordered)
    return {
        "overview": {
            "case_count": len(ordered),
            "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "answer_count": len(ordered) - len(refusals),
            "refusal_count": len(refusals),
            "answerable_refusal_count": len(answerable_refusals),
            "unanswerable_false_answer_count": len(false_answers),
            "selected_action_counts": _counter_dict(item.selected_action for item in ordered),
            "terminal_state_counts": _counter_dict(item.terminal_state for item in ordered),
        },
        "quality_diagnostics": {
            "answerable_refusal_rate": _ratio(len(answerable_refusals), len(answerable)),
            "unanswerable_refusal_rate": _ratio(
                len(unanswerable) - len(false_answers), len(unanswerable)
            ),
            "unanswerable_false_answer_rate": _ratio(len(false_answers), len(unanswerable)),
            "answerable_gold_candidate_pool_hit_rate": _ratio(
                sum(item.gold_candidate_rank is not None for item in answerable),
                len(answerable),
            ),
            "answerable_gold_generation_top10_hit_rate": _ratio(
                sum(item.gold_generation_rank is not None for item in answerable),
                len(answerable),
            ),
            "answerable_gold_verification_context_hit_rate": _ratio(
                sum(item.gold_verification_rank is not None for item in answerable),
                len(answerable),
            ),
            "answerable_gold_citation_rate": _ratio(
                sum(item.gold_cited for item in answerable), len(answerable)
            ),
            "average_answerable_token_f1_all": _average(
                [float(item.answer_token_f1 or 0.0) for item in answerable]
            ),
            "average_answerable_token_f1_completed": _average(
                [float(item.answer_token_f1 or 0.0) for item in complete_answerable]
            ),
        },
        "failure_bucket_counts": dict(sorted(failure_counts.items())),
        "failure_bucket_rates": {
            bucket: _ratio(count, len(ordered)) for bucket, count in sorted(failure_counts.items())
        },
        "answerable_refusal_flow": _answerable_refusal_flow(
            answerable=answerable,
            answerable_refusals=answerable_refusals,
        ),
        "by_turn_position": _group_summaries(
            ordered,
            key=lambda item: str(item.turn_position),
        ),
        "by_selected_action": _group_summaries(
            ordered,
            key=lambda item: item.selected_action,
        ),
        "by_answerability": _group_summaries(
            ordered,
            key=lambda item: "answerable" if item.answerable else "unanswerable",
        ),
        "by_question_route": _group_summaries(
            ordered,
            key=lambda item: item.question_route,
        ),
        "latency_diagnostics": _latency_diagnostics(ordered),
        "fold_diagnostic_stability": _fold_diagnostic_stability(ordered),
    }


def stage160_private_report(
    observations: Sequence[Stage160CaseObservation],
) -> dict[str, Any]:
    rows = [
        item.to_private_dict()
        for item in sorted(
            observations,
            key=lambda item: (item.thread_ordinal, item.turn_position),
        )
    ]
    return {
        "stage": "Stage 160 private diagnostic",
        "privacy_class": "local_ignored_hashed_case_features",
        "contains_raw_question": False,
        "contains_raw_answer": False,
        "contains_raw_document_id": False,
        "contains_raw_document_text": False,
        "contains_hashed_sample_identity": True,
        "row_count": len(rows),
        "rows": rows,
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _diagnostic_sample(sample: PrimeQAHybridSplitSample) -> Stage160DiagnosticSample:
    if sample.assigned_split != "dev":
        raise ValueError("Stage160 rejects non-dev rows")
    if sample.split_name != STAGE160_SPLIT_NAME:
        raise ValueError("Stage160 rejects an unexpected split name")
    if sample.protocol_version != STAGE160_PROTOCOL_VERSION:
        raise ValueError("Stage160 rejects an unexpected protocol version")
    if sample.answerable and not sample.answer_doc_id:
        raise ValueError("Stage160 answerable row requires a gold document")
    private_identity = _sha256_text(sample.sample_id)
    query_digest = _query_digest(sample.question_title, sample.question_text)
    group_material = (
        f"{_normalize_question(sample.question_title, sample.question_text)}::"
        f"{sample.answer_doc_id or 'UNANSWERABLE'}"
    )
    runtime_query = PrimeQARuntimeQuery(
        id=f"stage160-dev-{private_identity[:16]}",
        title=sample.question_title,
        text=sample.question_text,
    )
    return Stage160DiagnosticSample(
        private_identity_sha256=private_identity,
        query_digest_sha256=query_digest,
        diagnostic_group_sha256=_sha256_text(group_material),
        answerable=sample.answerable,
        gold_answer=sample.answer,
        gold_document_id=sample.answer_doc_id,
        gold_document_sha256=(_sha256_text(sample.answer_doc_id) if sample.answer_doc_id else None),
        question_route=classify_question_route(runtime_query),
        split_subtype=sample.split_subtype,
        runtime_query=runtime_query,
    )


def _answerable_refusal_flow(
    *,
    answerable: Sequence[Stage160CaseObservation],
    answerable_refusals: Sequence[Stage160CaseObservation],
) -> dict[str, Any]:
    bucket_counts = Counter(item.failure_bucket for item in answerable_refusals)
    gold_pool = sum(item.gold_candidate_rank is not None for item in answerable)
    gold_generation = sum(item.gold_generation_rank is not None for item in answerable)
    visible_refused = bucket_counts.get("answerable_refusal_gold_visible_model_refused", 0)
    return {
        "answerable_count": len(answerable),
        "answerable_refusal_count": len(answerable_refusals),
        "gold_present_candidate_pool_count": gold_pool,
        "gold_present_generation_top10_count": gold_generation,
        "gold_absent_candidate_pool_refusal_count": bucket_counts.get(
            "answerable_refusal_gold_absent_candidate_pool", 0
        ),
        "gold_lost_before_generation_refusal_count": bucket_counts.get(
            "answerable_refusal_gold_absent_generation_top10", 0
        ),
        "gold_visible_model_refusal_count": visible_refused,
        "post_compose_refusal_count": bucket_counts.get("answerable_refusal_after_compose", 0),
        "gold_visible_model_refusal_rate_given_visible": _ratio(
            visible_refused,
            sum(item.gold_generation_rank is not None for item in answerable_refusals),
        ),
    }


def _group_summaries(
    observations: Sequence[Stage160CaseObservation],
    *,
    key,
) -> dict[str, Any]:
    grouped: dict[str, list[Stage160CaseObservation]] = defaultdict(list)
    for observation in observations:
        grouped[str(key(observation))].append(observation)
    return {group: _case_group_summary(items) for group, items in sorted(grouped.items())}


def _case_group_summary(items: Sequence[Stage160CaseObservation]) -> dict[str, Any]:
    answerable = [item for item in items if item.answerable]
    unanswerable = [item for item in items if not item.answerable]
    return {
        "case_count": len(items),
        "answerable_count": len(answerable),
        "refusal_count": sum(item.refused for item in items),
        "answerable_refusal_rate": _ratio(
            sum(item.refused for item in answerable), len(answerable)
        ),
        "unanswerable_false_answer_rate": _ratio(
            sum(not item.refused for item in unanswerable), len(unanswerable)
        ),
        "gold_generation_hit_rate_answerable": _ratio(
            sum(item.gold_generation_rank is not None for item in answerable),
            len(answerable),
        ),
        "average_answerable_token_f1": _average(
            [float(item.answer_token_f1 or 0.0) for item in answerable]
        ),
        "end_to_end_latency_ms": _distribution([item.end_to_end_latency_ms for item in items]),
        "router_generation_latency_ms": _distribution(
            [item.router_generation_latency_ms for item in items]
        ),
        "router_input_token_count": _distribution(
            [item.router_input_token_count for item in items]
        ),
        "retained_state_bytes": _distribution([item.retained_state_bytes for item in items]),
        "stage159_p95_exceedance_count": sum(item.stage159_p95_exceeded for item in items),
    }


def _latency_diagnostics(
    observations: Sequence[Stage160CaseObservation],
) -> dict[str, Any]:
    e2e = [item.end_to_end_latency_ms for item in observations]
    generation = [item.router_generation_latency_ms for item in observations]
    long_tail = [item for item in observations if item.stage159_p95_exceeded]
    overhead = [
        max(0.0, item.end_to_end_latency_ms - item.router_generation_latency_ms)
        for item in observations
    ]
    return {
        "stage159_reference_p95_ms": STAGE159_END_TO_END_P95_MS,
        "end_to_end_latency_ms": _distribution(e2e),
        "router_generation_latency_ms": _distribution(generation),
        "non_generation_overhead_ms": _distribution(overhead),
        "generation_share_of_total_average": round(
            sum(generation) / sum(e2e) if sum(e2e) else 0.0,
            6,
        ),
        "stage159_p95_exceedance_count": len(long_tail),
        "stage159_p95_exceedance_rate": _ratio(len(long_tail), len(observations)),
        "long_tail_action_counts": _counter_dict(item.selected_action for item in long_tail),
        "long_tail_turn_position_counts": _counter_dict(
            str(item.turn_position) for item in long_tail
        ),
        "long_tail_answerability_counts": _counter_dict(
            "answerable" if item.answerable else "unanswerable" for item in long_tail
        ),
        "spearman_correlations_with_generation_latency": {
            "turn_position": _spearman([item.turn_position for item in observations], generation),
            "router_input_token_count": _spearman(
                [item.router_input_token_count for item in observations], generation
            ),
            "router_output_token_count": _spearman(
                [item.router_output_token_count for item in observations], generation
            ),
            "retained_state_bytes": _spearman(
                [item.retained_state_bytes for item in observations], generation
            ),
            "candidate_pool_count": _spearman(
                [item.candidate_pool_count for item in observations], generation
            ),
        },
    }


def _fold_diagnostic_stability(
    observations: Sequence[Stage160CaseObservation],
) -> dict[str, Any]:
    fold_ids = sorted({item.fold_id for item in observations})
    fold_reports = {
        str(fold_id): _fold_summary([item for item in observations if item.fold_id == fold_id])
        for fold_id in fold_ids
    }
    metric_names = (
        "answerable_refusal_rate",
        "unanswerable_false_answer_rate",
        "gold_candidate_hit_rate_answerable",
        "gold_generation_hit_rate_answerable",
        "average_answerable_token_f1",
        "stage159_p95_exceedance_rate",
    )
    ranges = {}
    for metric in metric_names:
        values = [float(report[metric]) for report in fold_reports.values()]
        ranges[metric] = {
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "range": round(max(values) - min(values), 6),
            "mean": round(statistics.fmean(values), 6),
            "population_stddev": round(statistics.pstdev(values), 6),
        }
    return {
        "fold_count": len(fold_ids),
        "fit_models": False,
        "select_policy": False,
        "folds": fold_reports,
        "metric_stability": ranges,
    }


def _fold_summary(items: Sequence[Stage160CaseObservation]) -> dict[str, Any]:
    answerable = [item for item in items if item.answerable]
    unanswerable = [item for item in items if not item.answerable]
    return {
        "case_count": len(items),
        "diagnostic_group_count": len({item.diagnostic_group_sha256 for item in items}),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answerable_refusal_rate": _ratio(
            sum(item.refused for item in answerable), len(answerable)
        ),
        "unanswerable_false_answer_rate": _ratio(
            sum(not item.refused for item in unanswerable), len(unanswerable)
        ),
        "gold_candidate_hit_rate_answerable": _ratio(
            sum(item.gold_candidate_rank is not None for item in answerable),
            len(answerable),
        ),
        "gold_generation_hit_rate_answerable": _ratio(
            sum(item.gold_generation_rank is not None for item in answerable),
            len(answerable),
        ),
        "average_answerable_token_f1": _average(
            [float(item.answer_token_f1 or 0.0) for item in answerable]
        ),
        "stage159_p95_exceedance_rate": _ratio(
            sum(item.stage159_p95_exceeded for item in items), len(items)
        ),
        "failure_bucket_counts": _counter_dict(item.failure_bucket for item in items),
    }


def _distribution(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "average": 0.0,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "median": round(float(statistics.median(ordered)), 3),
        "p95": _percentile(ordered, 95),
        "max": round(ordered[-1], 3),
        "average": round(statistics.fmean(ordered), 3),
    }


def _percentile(ordered: Sequence[float], percentile: int) -> float:
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = round((percentile / 100) * (len(ordered) - 1))
    return round(ordered[index], 3)


def _spearman(left: Sequence[int | float], right: Sequence[int | float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _average_ranks([float(value) for value in left])
    right_ranks = _average_ranks([float(value) for value in right])
    return round(_pearson(left_ranks, right_ranks), 6)


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for index, _ in indexed[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _normalize_question(title: str, text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", f"{title} {text}".lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _query_digest(title: str, text: str) -> str:
    return _sha256_text(f"{title}\0{text}")


def query_digest(title: str, text: str) -> str:
    return _query_digest(title, text)


def score_answer(answer: str, gold_answer: str, *, refused: bool) -> float:
    return 0.0 if refused else round(token_f1(answer, gold_answer), 6)


def _average(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 6) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_lines(lines) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
