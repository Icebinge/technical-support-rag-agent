from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol as stage160,
)
from ts_rag_agent.application.evidence_selection import classify_question_route
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)

STAGE165_TRAIN_SPLIT_FILENAME = "primeqa_hybrid_split_stage68_train.jsonl"
STAGE165_EXPECTED_TRAIN_SHA256 = "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155"
STAGE165_EXPECTED_TRAIN_ROW_COUNT = 562
STAGE165_EXPECTED_ANSWERABLE_COUNT = 370
STAGE165_EXPECTED_UNANSWERABLE_COUNT = 192
STAGE165_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
STAGE165_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
STAGE165_MAX_TURNS_PER_SYNTHETIC_THREAD = 4
STAGE165_FOLD_COUNT = 5
STAGE165_EXPECTED_ORDER_SHA256 = "e1e5fb1902bfd32a2900a9a4f44b041fd469748a847904657e8e104007e48d25"
STAGE165_EXPECTED_GROUPING_SHA256 = (
    "a00ebf94a9e1e2125805c0120da4fd109df260483173d1adc56a943f4284621c"
)
STAGE165_EXPECTED_ARM_SCHEDULE_SHA256 = (
    "b675b662132b9255050dd9df92623f5d2e07ef71f37a3a85d188488ad2566493"
)
STAGE165_EXPECTED_FOLD_ASSIGNMENT_SHA256 = (
    "5f1536a65ce0afe03685babd045e7d212eedeffb2954e071f9f23a7867febb98"
)

Stage165Arm = Literal["isolated", "synthetic_history"]


@dataclass(frozen=True)
class Stage165TrainSample:
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
class Stage165TrainDiagnosticSet:
    source_size_bytes: int
    source_sha256: str
    order_sha256: str
    samples: tuple[Stage165TrainSample, ...]

    def public_summary(self) -> dict[str, object]:
        answerable_count = sum(sample.answerable for sample in self.samples)
        return {
            "source_size_bytes": self.source_size_bytes,
            "source_sha256": self.source_sha256,
            "train_row_count": len(self.samples),
            "answerable_count": answerable_count,
            "unanswerable_count": len(self.samples) - answerable_count,
            "stable_order_sha256": self.order_sha256,
            "assigned_split": "train",
            "split_name": STAGE165_SPLIT_NAME,
            "protocol_version": STAGE165_PROTOCOL_VERSION,
            "gold_fields_used_after_runtime_only": True,
            "gold_fields_projected_into_runtime": False,
            "dev_loaded": False,
            "test_loaded": False,
        }


@dataclass(frozen=True)
class Stage165SyntheticThread:
    ordinal: int
    samples: tuple[Stage165TrainSample, ...]


@dataclass(frozen=True)
class Stage165PairedWorkloadPlan:
    threads: tuple[Stage165SyntheticThread, ...]
    grouping_sha256: str
    arm_schedule_sha256: str

    @property
    def ordered_samples(self) -> tuple[Stage165TrainSample, ...]:
        return tuple(sample for thread in self.threads for sample in thread.samples)

    def arm_order(self, sample: Stage165TrainSample) -> tuple[Stage165Arm, Stage165Arm]:
        if int(sample.private_identity_sha256[:8], 16) % 2 == 0:
            return ("isolated", "synthetic_history")
        return ("synthetic_history", "isolated")

    def public_summary(self) -> dict[str, object]:
        positions = Counter(
            position for thread in self.threads for position in range(1, len(thread.samples) + 1)
        )
        isolated_first = sum(
            self.arm_order(sample)[0] == "isolated" for sample in self.ordered_samples
        )
        return {
            "design": "full_train_paired_isolated_vs_synthetic_history",
            "grouping_policy": "stable_hash_order_consecutive_groups_of_four",
            "semantic_relationship_policy": "synthetic_grouping_not_natural_conversation",
            "thread_count": len(self.threads),
            "full_four_turn_thread_count": sum(len(thread.samples) == 4 for thread in self.threads),
            "trailing_thread_turn_count": len(self.threads[-1].samples),
            "unique_sample_count": len(self.ordered_samples),
            "agent_turn_count": len(self.ordered_samples) * 2,
            "arm_turn_counts": {
                "isolated": len(self.ordered_samples),
                "synthetic_history": len(self.ordered_samples),
            },
            "arm_first_counts": {
                "isolated": isolated_first,
                "synthetic_history": len(self.ordered_samples) - isolated_first,
            },
            "turn_position_counts": {
                str(position): positions[position]
                for position in range(1, STAGE165_MAX_TURNS_PER_SYNTHETIC_THREAD + 1)
            },
            "first_turn_negative_control_count": positions[1],
            "post_first_turn_primary_count": len(self.ordered_samples) - positions[1],
            "grouping_sha256": self.grouping_sha256,
            "arm_schedule_sha256": self.arm_schedule_sha256,
            "runtime_instance_count": 1,
            "model_load_count": 1,
            "resource_build_count": 1,
            "queue_allowed": False,
            "retry_allowed": False,
            "fallback_allowed": False,
        }


@dataclass(frozen=True)
class Stage165FoldAssignment:
    fold_by_private_identity: Mapping[str, int]
    group_count: int
    row_counts: tuple[int, ...]
    group_counts: tuple[int, ...]
    assignment_sha256: str

    def public_summary(self) -> dict[str, object]:
        return {
            "analysis_mode": "five_fold_grouped_paired_diagnostic_stability",
            "fold_count": len(self.row_counts),
            "group_key": "normalized_question_plus_answer_document_or_unanswerable",
            "group_count": self.group_count,
            "row_counts": {str(index): count for index, count in enumerate(self.row_counts)},
            "group_counts": {str(index): count for index, count in enumerate(self.group_counts)},
            "assignment_sha256": self.assignment_sha256,
            "fit_models": False,
            "select_policy": False,
            "tune_thresholds": False,
            "cross_fold_outcome_use": "directional_stability_only",
        }


@dataclass(frozen=True)
class Stage165ArmObservation:
    private_identity_sha256: str
    query_digest_sha256: str
    diagnostic_group_sha256: str
    gold_document_sha256: str | None
    fold_id: int
    synthetic_thread_ordinal: int
    synthetic_turn_position: int
    arm: Stage165Arm
    arm_order_position: int
    answerable: bool
    question_route: str
    split_subtype: str
    selected_action: str
    terminal_state: str
    refused: bool
    history_turn_count_before: int
    completed_turn_count_after: int
    retained_state_bytes_after: int
    candidate_pool_count: int
    generation_context_count: int
    verification_context_count: int
    candidate_context_sha256: str
    generation_context_sha256: str
    verification_context_sha256: str
    output_sha256: str
    gold_candidate_rank: int | None
    gold_generation_rank: int | None
    gold_verification_rank: int | None
    gold_cited: bool
    citation_count: int
    answer_token_f1: float | None
    top_candidate_score: float | None
    gold_candidate_score: float | None
    question_token_recall_in_gold_prompt: float | None
    answer_token_recall_in_gold_prompt: float | None
    answer_exact_span_visible: bool | None
    router_input_token_count: int
    router_output_token_count: int
    router_generation_latency_ms: float
    end_to_end_latency_ms: float
    retrieval_call_count: int
    model_decision_count: int
    composition_call_count: int
    verification_call_count: int
    diagnostic_observation_count: int
    retry_action_count: int
    fallback_action_count: int

    def to_private_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Stage165PairObservation:
    private_identity_sha256: str
    diagnostic_group_sha256: str
    fold_id: int
    synthetic_thread_ordinal: int
    synthetic_turn_position: int
    answerable: bool
    question_route: str
    split_subtype: str
    isolated_refused: bool
    synthetic_history_refused: bool
    refusal_delta_synthetic_minus_isolated: int
    false_answer_delta_isolated_minus_synthetic: int | None
    answer_f1_delta_isolated_minus_synthetic: float | None
    citation_delta_isolated_minus_synthetic: int
    input_token_delta_synthetic_minus_isolated: int
    generation_latency_delta_synthetic_minus_isolated_ms: float
    context_signatures_exact: bool
    first_turn_output_exact: bool | None
    gold_visible_in_both: bool
    question_token_recall_in_gold_prompt: float | None
    answer_token_recall_in_gold_prompt: float | None
    answer_exact_span_visible: bool | None


def load_stage165_train_diagnostic_samples(path: Path) -> Stage165TrainDiagnosticSet:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Stage165 train source must be a file")
    source_sha256 = _file_sha256(resolved)
    if source_sha256 != STAGE165_EXPECTED_TRAIN_SHA256:
        raise ValueError("Stage165 train source SHA-256 does not match the frozen split")
    raw_samples = load_primeqa_hybrid_split_samples(resolved)
    if len(raw_samples) != STAGE165_EXPECTED_TRAIN_ROW_COUNT:
        raise ValueError("Stage165 train source row count does not match the protocol")
    samples = tuple(_diagnostic_sample(sample) for sample in raw_samples)
    if sum(sample.answerable for sample in samples) != STAGE165_EXPECTED_ANSWERABLE_COUNT:
        raise ValueError("Stage165 train answerable count does not match the protocol")
    if len({sample.private_identity_sha256 for sample in samples}) != len(samples):
        raise ValueError("Stage165 train source contains duplicate private identities")
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
    return Stage165TrainDiagnosticSet(
        source_size_bytes=resolved.stat().st_size,
        source_sha256=source_sha256,
        order_sha256=order_sha256,
        samples=ordered,
    )


def build_stage165_paired_workload_plan(
    diagnostic_set: Stage165TrainDiagnosticSet,
) -> Stage165PairedWorkloadPlan:
    if not diagnostic_set.samples:
        raise ValueError("Stage165 workload requires at least one sample")
    threads = tuple(
        Stage165SyntheticThread(
            ordinal=(offset // STAGE165_MAX_TURNS_PER_SYNTHETIC_THREAD) + 1,
            samples=diagnostic_set.samples[
                offset : offset + STAGE165_MAX_TURNS_PER_SYNTHETIC_THREAD
            ],
        )
        for offset in range(
            0,
            len(diagnostic_set.samples),
            STAGE165_MAX_TURNS_PER_SYNTHETIC_THREAD,
        )
    )
    grouping_sha256 = _digest_lines(
        f"{thread.ordinal}:{','.join(sample.private_identity_sha256 for sample in thread.samples)}"
        for thread in threads
    )
    provisional = Stage165PairedWorkloadPlan(
        threads=threads,
        grouping_sha256=grouping_sha256,
        arm_schedule_sha256="",
    )
    schedule_sha256 = _digest_lines(
        f"{sample.private_identity_sha256}:{','.join(provisional.arm_order(sample))}"
        for sample in provisional.ordered_samples
    )
    return Stage165PairedWorkloadPlan(
        threads=threads,
        grouping_sha256=grouping_sha256,
        arm_schedule_sha256=schedule_sha256,
    )


def build_stage165_grouped_fold_assignment(
    samples: Sequence[Stage165TrainSample],
    *,
    fold_count: int = STAGE165_FOLD_COUNT,
) -> Stage165FoldAssignment:
    if fold_count < 2:
        raise ValueError("Stage165 diagnostic fold count must be at least two")
    groups: dict[str, list[Stage165TrainSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.diagnostic_group_sha256].append(sample)
    if fold_count > len(groups):
        raise ValueError("Stage165 diagnostic fold count exceeds group count")
    fold_rows = [0] * fold_count
    fold_groups = [0] * fold_count
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
    return Stage165FoldAssignment(
        fold_by_private_identity=fold_by_identity,
        group_count=len(groups),
        row_counts=tuple(fold_rows),
        group_counts=tuple(fold_groups),
        assignment_sha256=assignment_sha256,
    )


def pair_stage165_observations(
    observations: Sequence[Stage165ArmObservation],
) -> tuple[Stage165PairObservation, ...]:
    by_identity: dict[str, dict[Stage165Arm, Stage165ArmObservation]] = defaultdict(dict)
    for observation in observations:
        arms = by_identity[observation.private_identity_sha256]
        if observation.arm in arms:
            raise ValueError("Stage165 observations contain a duplicate sample arm")
        arms[observation.arm] = observation
    pairs = []
    for identity, arms in sorted(by_identity.items()):
        if set(arms) != {"isolated", "synthetic_history"}:
            raise ValueError("Stage165 observations do not contain both paired arms")
        isolated = arms["isolated"]
        synthetic = arms["synthetic_history"]
        _validate_pair_identity(isolated, synthetic)
        context_exact = (
            isolated.candidate_context_sha256 == synthetic.candidate_context_sha256
            and isolated.generation_context_sha256 == synthetic.generation_context_sha256
            and isolated.verification_context_sha256 == synthetic.verification_context_sha256
        )
        pairs.append(
            Stage165PairObservation(
                private_identity_sha256=identity,
                diagnostic_group_sha256=isolated.diagnostic_group_sha256,
                fold_id=isolated.fold_id,
                synthetic_thread_ordinal=isolated.synthetic_thread_ordinal,
                synthetic_turn_position=isolated.synthetic_turn_position,
                answerable=isolated.answerable,
                question_route=isolated.question_route,
                split_subtype=isolated.split_subtype,
                isolated_refused=isolated.refused,
                synthetic_history_refused=synthetic.refused,
                refusal_delta_synthetic_minus_isolated=(
                    int(synthetic.refused) - int(isolated.refused)
                ),
                false_answer_delta_isolated_minus_synthetic=(
                    None
                    if isolated.answerable
                    else int(not isolated.refused) - int(not synthetic.refused)
                ),
                answer_f1_delta_isolated_minus_synthetic=(
                    None
                    if not isolated.answerable
                    else round(
                        float(isolated.answer_token_f1 or 0.0)
                        - float(synthetic.answer_token_f1 or 0.0),
                        6,
                    )
                ),
                citation_delta_isolated_minus_synthetic=(
                    isolated.citation_count - synthetic.citation_count
                ),
                input_token_delta_synthetic_minus_isolated=(
                    synthetic.router_input_token_count - isolated.router_input_token_count
                ),
                generation_latency_delta_synthetic_minus_isolated_ms=round(
                    synthetic.router_generation_latency_ms - isolated.router_generation_latency_ms,
                    3,
                ),
                context_signatures_exact=context_exact,
                first_turn_output_exact=(
                    isolated.output_sha256 == synthetic.output_sha256
                    if isolated.synthetic_turn_position == 1
                    else None
                ),
                gold_visible_in_both=(
                    isolated.gold_generation_rank is not None
                    and synthetic.gold_generation_rank is not None
                ),
                question_token_recall_in_gold_prompt=(
                    isolated.question_token_recall_in_gold_prompt if context_exact else None
                ),
                answer_token_recall_in_gold_prompt=(
                    isolated.answer_token_recall_in_gold_prompt if context_exact else None
                ),
                answer_exact_span_visible=(
                    isolated.answer_exact_span_visible if context_exact else None
                ),
            )
        )
    return tuple(pairs)


def summarize_stage165_pairs(
    observations: Sequence[Stage165ArmObservation],
) -> dict[str, object]:
    pairs = pair_stage165_observations(observations)
    answerable = tuple(pair for pair in pairs if pair.answerable)
    unanswerable = tuple(pair for pair in pairs if not pair.answerable)
    post_first = tuple(pair for pair in pairs if pair.synthetic_turn_position > 1)
    answerable_post_first = tuple(pair for pair in post_first if pair.answerable)
    unanswerable_post_first = tuple(pair for pair in post_first if not pair.answerable)
    visible_post_first = tuple(pair for pair in answerable_post_first if pair.gold_visible_in_both)
    return {
        "overview": {
            "pair_count": len(pairs),
            "arm_observation_count": len(observations),
            "answerable_pair_count": len(answerable),
            "unanswerable_pair_count": len(unanswerable),
            "first_turn_pair_count": len(pairs) - len(post_first),
            "post_first_turn_pair_count": len(post_first),
            "context_signature_exact_count": sum(pair.context_signatures_exact for pair in pairs),
            "first_turn_output_exact_count": sum(
                pair.first_turn_output_exact is True for pair in pairs
            ),
        },
        "arm_outcomes": {
            arm: _arm_summary(observations, arm=arm) for arm in ("isolated", "synthetic_history")
        },
        "primary_post_first_answerable_effect": _answerable_pair_effect(answerable_post_first),
        "gold_visible_post_first_answerable_effect": _answerable_pair_effect(visible_post_first),
        "unanswerable_post_first_safety_effect": _unanswerable_pair_effect(unanswerable_post_first),
        "first_turn_negative_control": _first_turn_control(pairs),
        "by_turn_position": {
            str(position): _mixed_pair_summary(
                [pair for pair in pairs if pair.synthetic_turn_position == position]
            )
            for position in range(1, STAGE165_MAX_TURNS_PER_SYNTHETIC_THREAD + 1)
        },
        "question_alignment": _question_alignment_summary(visible_post_first),
        "grouped_fold_stability": _fold_stability(pairs),
        "route_effects": {
            route: _answerable_pair_effect(
                [pair for pair in answerable_post_first if pair.question_route == route]
            )
            for route in sorted({pair.question_route for pair in answerable_post_first})
        },
    }


def stage165_private_report(
    observations: Sequence[Stage165ArmObservation],
) -> dict[str, object]:
    rows = [
        observation.to_private_dict()
        for observation in sorted(
            observations,
            key=lambda item: (
                item.private_identity_sha256,
                item.arm,
            ),
        )
    ]
    return {
        "artifact_id": "primeqa_hybrid_stage165_train_history_isolation_private_v1",
        "privacy": "hashed_content_free_train_diagnostics",
        "contains_raw_question": False,
        "contains_raw_answer": False,
        "contains_raw_document_id": False,
        "contains_raw_document_text": False,
        "contains_raw_model_output": False,
        "contains_hashed_sample_identity": True,
        "arm_row_count": len(rows),
        "pair_count": len(rows) // 2,
        "rows": rows,
    }


def _diagnostic_sample(sample: PrimeQAHybridSplitSample) -> Stage165TrainSample:
    if sample.assigned_split != "train":
        raise ValueError("Stage165 rejects non-train rows")
    if sample.split_name != STAGE165_SPLIT_NAME:
        raise ValueError("Stage165 rejects an unexpected split name")
    if sample.protocol_version != STAGE165_PROTOCOL_VERSION:
        raise ValueError("Stage165 rejects an unexpected protocol version")
    if sample.answerable and not sample.answer_doc_id:
        raise ValueError("Stage165 answerable row requires a gold document")
    private_identity = _sha256_text(sample.sample_id)
    query_digest_sha256 = _sha256_text(f"{sample.question_title}\0{sample.question_text}")
    group_material = (
        f"{_normalize_question(sample.question_title, sample.question_text)}::"
        f"{sample.answer_doc_id or 'UNANSWERABLE'}"
    )
    runtime_query = PrimeQARuntimeQuery(
        id=f"stage165-train-{private_identity[:16]}",
        title=sample.question_title,
        text=sample.question_text,
    )
    return Stage165TrainSample(
        private_identity_sha256=private_identity,
        query_digest_sha256=query_digest_sha256,
        diagnostic_group_sha256=_sha256_text(group_material),
        answerable=sample.answerable,
        gold_answer=sample.answer,
        gold_document_id=sample.answer_doc_id,
        gold_document_sha256=(_sha256_text(sample.answer_doc_id) if sample.answer_doc_id else None),
        question_route=classify_question_route(runtime_query),
        split_subtype=sample.split_subtype,
        runtime_query=runtime_query,
    )


def _validate_pair_identity(
    isolated: Stage165ArmObservation,
    synthetic: Stage165ArmObservation,
) -> None:
    fields = (
        "query_digest_sha256",
        "diagnostic_group_sha256",
        "gold_document_sha256",
        "fold_id",
        "synthetic_thread_ordinal",
        "synthetic_turn_position",
        "answerable",
        "question_route",
        "split_subtype",
    )
    if any(getattr(isolated, field) != getattr(synthetic, field) for field in fields):
        raise ValueError("Stage165 paired arm identities differ")


def _arm_summary(
    observations: Sequence[Stage165ArmObservation],
    *,
    arm: Stage165Arm,
) -> dict[str, object]:
    rows = [row for row in observations if row.arm == arm]
    answerable = [row for row in rows if row.answerable]
    unanswerable = [row for row in rows if not row.answerable]
    post_first = [row for row in rows if row.synthetic_turn_position > 1]
    post_first_answerable = [row for row in post_first if row.answerable]
    post_first_unanswerable = [row for row in post_first if not row.answerable]
    return {
        "row_count": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answerable_refusal_count": sum(row.refused for row in answerable),
        "answerable_refusal_rate": _ratio(sum(row.refused for row in answerable), len(answerable)),
        "unanswerable_false_answer_count": sum(not row.refused for row in unanswerable),
        "unanswerable_false_answer_rate": _ratio(
            sum(not row.refused for row in unanswerable), len(unanswerable)
        ),
        "post_first_answerable_refusal_rate": _ratio(
            sum(row.refused for row in post_first_answerable),
            len(post_first_answerable),
        ),
        "post_first_unanswerable_false_answer_rate": _ratio(
            sum(not row.refused for row in post_first_unanswerable),
            len(post_first_unanswerable),
        ),
        "answerable_average_token_f1": _average(
            [float(row.answer_token_f1 or 0.0) for row in answerable]
        ),
        "answerable_gold_citation_count": sum(row.gold_cited for row in answerable),
        "selected_action_counts": _counter_dict(row.selected_action for row in rows),
        "router_input_token_count": _distribution([row.router_input_token_count for row in rows]),
        "router_generation_latency_ms": _distribution(
            [row.router_generation_latency_ms for row in rows]
        ),
        "end_to_end_latency_ms": _distribution([row.end_to_end_latency_ms for row in rows]),
    }


def _answerable_pair_effect(
    pairs: Sequence[Stage165PairObservation],
) -> dict[str, object]:
    count = len(pairs)
    isolated_refusals = sum(pair.isolated_refused for pair in pairs)
    synthetic_refusals = sum(pair.synthetic_history_refused for pair in pairs)
    harmed = sum(not pair.isolated_refused and pair.synthetic_history_refused for pair in pairs)
    helped = sum(pair.isolated_refused and not pair.synthetic_history_refused for pair in pairs)
    f1_deltas = [float(pair.answer_f1_delta_isolated_minus_synthetic or 0.0) for pair in pairs]
    citation_delta = sum(pair.citation_delta_isolated_minus_synthetic for pair in pairs)
    return {
        "pair_count": count,
        "isolated_refusal_count": isolated_refusals,
        "synthetic_history_refusal_count": synthetic_refusals,
        "isolated_refusal_rate": _ratio(isolated_refusals, count),
        "synthetic_history_refusal_rate": _ratio(synthetic_refusals, count),
        "refusal_rate_difference_synthetic_minus_isolated": round(
            _ratio(synthetic_refusals, count) - _ratio(isolated_refusals, count),
            6,
        ),
        "isolated_answer_to_synthetic_refusal_count": harmed,
        "isolated_refusal_to_synthetic_answer_count": helped,
        "discordant_pair_count": harmed + helped,
        "mcnemar_exact_two_sided_p": _exact_two_sided_binomial(harmed, helped),
        "average_answer_f1_difference_isolated_minus_synthetic": _average(f1_deltas),
        "positive_f1_pair_count": sum(delta > 0 for delta in f1_deltas),
        "negative_f1_pair_count": sum(delta < 0 for delta in f1_deltas),
        "tie_f1_pair_count": sum(delta == 0 for delta in f1_deltas),
        "gold_citation_difference_isolated_minus_synthetic": citation_delta,
        "average_input_token_reduction_isolated_vs_synthetic": _average(
            [float(pair.input_token_delta_synthetic_minus_isolated) for pair in pairs]
        ),
        "average_generation_latency_reduction_isolated_vs_synthetic_ms": _average(
            [pair.generation_latency_delta_synthetic_minus_isolated_ms for pair in pairs]
        ),
    }


def _unanswerable_pair_effect(
    pairs: Sequence[Stage165PairObservation],
) -> dict[str, object]:
    count = len(pairs)
    isolated_false = sum(not pair.isolated_refused for pair in pairs)
    synthetic_false = sum(not pair.synthetic_history_refused for pair in pairs)
    worsened = sum(not pair.isolated_refused and pair.synthetic_history_refused for pair in pairs)
    improved = sum(pair.isolated_refused and not pair.synthetic_history_refused for pair in pairs)
    return {
        "pair_count": count,
        "isolated_false_answer_count": isolated_false,
        "synthetic_history_false_answer_count": synthetic_false,
        "isolated_false_answer_rate": _ratio(isolated_false, count),
        "synthetic_history_false_answer_rate": _ratio(synthetic_false, count),
        "false_answer_rate_difference_isolated_minus_synthetic": round(
            _ratio(isolated_false, count) - _ratio(synthetic_false, count),
            6,
        ),
        "synthetic_refusal_to_isolated_false_answer_count": worsened,
        "synthetic_false_answer_to_isolated_refusal_count": improved,
        "discordant_pair_count": worsened + improved,
        "mcnemar_exact_two_sided_p": _exact_two_sided_binomial(worsened, improved),
    }


def _first_turn_control(
    pairs: Sequence[Stage165PairObservation],
) -> dict[str, object]:
    rows = [pair for pair in pairs if pair.synthetic_turn_position == 1]
    return {
        "pair_count": len(rows),
        "context_signature_exact_count": sum(pair.context_signatures_exact for pair in rows),
        "output_exact_count": sum(pair.first_turn_output_exact is True for pair in rows),
        "refusal_disagreement_count": sum(
            pair.isolated_refused != pair.synthetic_history_refused for pair in rows
        ),
        "average_input_token_difference_synthetic_minus_isolated": _average(
            [float(pair.input_token_delta_synthetic_minus_isolated) for pair in rows]
        ),
    }


def _mixed_pair_summary(
    pairs: Sequence[Stage165PairObservation],
) -> dict[str, object]:
    answerable = [pair for pair in pairs if pair.answerable]
    unanswerable = [pair for pair in pairs if not pair.answerable]
    return {
        "pair_count": len(pairs),
        "answerable": _answerable_pair_effect(answerable),
        "unanswerable": _unanswerable_pair_effect(unanswerable),
    }


def _question_alignment_summary(
    pairs: Sequence[Stage165PairObservation],
) -> dict[str, object]:
    bins = (
        ("0.00_to_0.25", 0.0, 0.25),
        ("0.25_to_0.50", 0.25, 0.50),
        ("0.50_to_0.75", 0.50, 0.75),
        ("0.75_to_1.00", 0.75, 1.0),
    )
    by_bin: dict[str, object] = {}
    for index, (label, lower, upper) in enumerate(bins):
        rows = [
            pair
            for pair in pairs
            if pair.question_token_recall_in_gold_prompt is not None
            and (
                float(pair.question_token_recall_in_gold_prompt) >= lower
                if index == 0
                else float(pair.question_token_recall_in_gold_prompt) > lower
            )
            and float(pair.question_token_recall_in_gold_prompt) <= upper
        ]
        by_bin[label] = _answerable_pair_effect(rows)
    low = [
        pair
        for pair in pairs
        if pair.question_token_recall_in_gold_prompt is not None
        and float(pair.question_token_recall_in_gold_prompt) <= 0.5
    ]
    high = [
        pair
        for pair in pairs
        if pair.question_token_recall_in_gold_prompt is not None
        and float(pair.question_token_recall_in_gold_prompt) > 0.5
    ]
    return {
        "cohort": "post_first_answerable_gold_visible_in_both_arms",
        "fixed_absolute_bins_selected_before_runtime": True,
        "fixed_low_alignment_boundary": 0.5,
        "by_fixed_bin": by_bin,
        "low_alignment": _answerable_pair_effect(low),
        "high_alignment": _answerable_pair_effect(high),
        "no_model_fit": True,
        "no_threshold_tuning": True,
        "causal_claim": False,
    }


def _fold_stability(
    pairs: Sequence[Stage165PairObservation],
) -> dict[str, object]:
    fold_ids = sorted({pair.fold_id for pair in pairs})
    folds = {}
    for fold_id in fold_ids:
        fold_pairs = [pair for pair in pairs if pair.fold_id == fold_id]
        post_first_answerable = [
            pair for pair in fold_pairs if pair.synthetic_turn_position > 1 and pair.answerable
        ]
        post_first_unanswerable = [
            pair for pair in fold_pairs if pair.synthetic_turn_position > 1 and not pair.answerable
        ]
        visible = [pair for pair in post_first_answerable if pair.gold_visible_in_both]
        folds[str(fold_id)] = {
            "pair_count": len(fold_pairs),
            "diagnostic_group_count": len({pair.diagnostic_group_sha256 for pair in fold_pairs}),
            "post_first_answerable": _answerable_pair_effect(post_first_answerable),
            "post_first_gold_visible_answerable": _answerable_pair_effect(visible),
            "post_first_unanswerable": _unanswerable_pair_effect(post_first_unanswerable),
        }
    primary_deltas = [
        float(fold["post_first_answerable"]["refusal_rate_difference_synthetic_minus_isolated"])
        for fold in folds.values()
    ]
    visible_deltas = [
        float(
            fold["post_first_gold_visible_answerable"][
                "refusal_rate_difference_synthetic_minus_isolated"
            ]
        )
        for fold in folds.values()
    ]
    safety_deltas = [
        float(
            fold["post_first_unanswerable"]["false_answer_rate_difference_isolated_minus_synthetic"]
        )
        for fold in folds.values()
    ]
    return {
        "fold_count": len(fold_ids),
        "folds": folds,
        "primary_answerable_refusal_delta_direction": _direction_summary(primary_deltas),
        "gold_visible_refusal_delta_direction": _direction_summary(visible_deltas),
        "unanswerable_false_answer_delta_direction": _direction_summary(safety_deltas),
        "fit_models": False,
        "select_policy": False,
        "tune_thresholds": False,
    }


def _direction_summary(values: Sequence[float]) -> dict[str, object]:
    return {
        "values": [round(value, 6) for value in values],
        "positive_count": sum(value > 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "minimum": round(min(values), 6) if values else 0.0,
        "maximum": round(max(values), 6) if values else 0.0,
    }


def _exact_two_sided_binomial(left: int, right: int) -> float:
    total = left + right
    if total == 0:
        return 1.0
    tail = sum(_binomial_probability(total, count) for count in range(min(left, right) + 1))
    return round(min(1.0, 2.0 * tail), 6)


def _binomial_probability(total: int, successes: int) -> float:
    import math

    return math.comb(total, successes) * (0.5**total)


def _distribution(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        return {
            "count": 0,
            "minimum": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
            "average": 0.0,
        }
    ordered = sorted(float(value) for value in values)
    p95_index = round(0.95 * (len(ordered) - 1))
    return {
        "count": len(ordered),
        "minimum": round(ordered[0], 3),
        "median": round(float(statistics.median(ordered)), 3),
        "p95": round(ordered[p95_index], 3),
        "maximum": round(ordered[-1], 3),
        "average": round(statistics.fmean(ordered), 3),
    }


def _average(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 6) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _normalize_question(title: str, text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", f"{title} {text}".lower())
    return re.sub(r"\s+", " ", normalized).strip()


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


def protocol_sha256(value: object) -> str:
    return stage160.canonical_json_sha256(json.loads(json.dumps(value, ensure_ascii=True)))
