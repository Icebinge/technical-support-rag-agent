from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    ThreadStateSummary,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    BoundedDynamicAgentRuntimeRun,
    PrimeQAHybridBoundedDynamicAgentRuntime,
)
from ts_rag_agent.domain.dataset import PrimeQAQuery, PrimeQARuntimeQuery

STAGE159_DEV_SPLIT_FILENAME = "primeqa_hybrid_split_stage68_dev.jsonl"
STAGE159_EXPECTED_DEV_SHA256 = "071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f"
STAGE159_EXPECTED_DEV_ROW_COUNT = 121
STAGE159_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
STAGE159_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
STAGE159_MAX_TURNS_PER_THREAD = 4


@dataclass(frozen=True)
class Stage159DevRuntimeQuery:
    """One dev query with a private stable identity used only for ordering."""

    private_identity_sha256: str
    runtime_query: PrimeQARuntimeQuery


@dataclass(frozen=True)
class Stage159DevQuerySet:
    """Exact query-only projection of the frozen Stage68 dev split."""

    source_size_bytes: int
    source_sha256: str
    order_sha256: str
    queries: tuple[Stage159DevRuntimeQuery, ...]

    def public_summary(self) -> dict[str, Any]:
        return {
            "source_size_bytes": self.source_size_bytes,
            "source_sha256": self.source_sha256,
            "dev_query_count": len(self.queries),
            "stable_order_sha256": self.order_sha256,
            "assigned_split": "dev",
            "split_name": STAGE159_SPLIT_NAME,
            "protocol_version": STAGE159_PROTOCOL_VERSION,
            "label_fields_used_for_selection": False,
            "label_fields_projected_into_runtime": False,
            "label_fields_used_for_metrics": False,
        }


@dataclass(frozen=True)
class Stage159ThreadWorkload:
    ordinal: int
    queries: tuple[Stage159DevRuntimeQuery, ...]


@dataclass(frozen=True)
class Stage159WorkloadPlan:
    threads: tuple[Stage159ThreadWorkload, ...]
    grouping_sha256: str

    def public_summary(self) -> dict[str, Any]:
        turn_position_counts = Counter(
            position for thread in self.threads for position in range(1, len(thread.queries) + 1)
        )
        return {
            "grouping_policy": "stable_hash_order_consecutive_groups_of_four",
            "semantic_relationship_policy": "synthetic_grouping_not_natural_conversation",
            "thread_count": len(self.threads),
            "full_four_turn_thread_count": sum(
                len(thread.queries) == STAGE159_MAX_TURNS_PER_THREAD for thread in self.threads
            ),
            "trailing_thread_turn_count": len(self.threads[-1].queries),
            "turn_count": sum(len(thread.queries) for thread in self.threads),
            "turn_position_counts": {
                str(position): turn_position_counts[position]
                for position in range(1, STAGE159_MAX_TURNS_PER_THREAD + 1)
            },
            "grouping_sha256": self.grouping_sha256,
        }


@dataclass(frozen=True)
class Stage159TurnObservation:
    thread_ordinal: int
    turn_position: int
    http_status: int
    refused: bool
    citation_count: int
    terminal_state: str
    selected_action: str
    completed_turn_count: int
    retained_state_bytes: int
    router_input_token_count: int
    router_output_token_count: int
    router_generation_latency_ms: float
    end_to_end_latency_ms: float
    retrieval_call_count: int
    model_decision_count: int
    composition_call_count: int
    verification_call_count: int
    diagnostic_observation_count: int

    @property
    def branch_protocol_valid(self) -> bool:
        if self.selected_action == "refuse_insufficient_evidence":
            return (
                self.terminal_state == "refuse"
                and self.composition_call_count == 0
                and self.verification_call_count == 0
                and self.diagnostic_observation_count == 0
            )
        if self.selected_action == "compose_grounded_answer":
            return (
                self.terminal_state in {"complete", "refuse"}
                and self.composition_call_count == 1
                and self.verification_call_count == 1
                and self.diagnostic_observation_count == 1
            )
        return False


class Stage159RuntimeObservationGate:
    """Validation-only event gate around one admitted real runtime turn."""

    def __init__(self, runtime: PrimeQAHybridBoundedDynamicAgentRuntime) -> None:
        self._runtime = runtime
        self._lock = Lock()
        self._entered = Event()
        self._release = Event()
        self._armed = False

    @property
    def last_public_trace(self):
        return self._runtime.last_public_trace

    def topology(self) -> dict[str, Any]:
        return self._runtime.topology()

    def open_thread(self, opaque_thread_handle: str) -> ThreadStateSummary:
        return self._runtime.open_thread(opaque_thread_handle)

    def close_thread(self, opaque_thread_handle: str) -> ThreadStateSummary:
        return self._runtime.close_thread(opaque_thread_handle)

    def thread_summary(self, opaque_thread_handle: str) -> ThreadStateSummary:
        return self._runtime.thread_summary(opaque_thread_handle)

    def arm(self) -> None:
        with self._lock:
            if self._armed:
                raise RuntimeError("Stage159 runtime observation gate is already armed")
            self._entered.clear()
            self._release.clear()
            self._armed = True

    def wait_until_entered(self) -> None:
        self._entered.wait()

    def release(self) -> None:
        with self._lock:
            if not self._armed:
                raise RuntimeError("Stage159 runtime observation gate is not armed")
            self._armed = False
            self._release.set()

    def run_turn(
        self,
        *,
        opaque_thread_handle: str,
        question: PrimeQAQuery,
    ) -> BoundedDynamicAgentRuntimeRun:
        with self._lock:
            gated = self._armed
        if gated:
            self._entered.set()
            self._release.wait()
        return self._runtime.run_turn(
            opaque_thread_handle=opaque_thread_handle,
            question=question,
        )


def load_stage159_dev_runtime_queries(
    path: Path,
    *,
    expected_sha256: str = STAGE159_EXPECTED_DEV_SHA256,
    expected_row_count: int = STAGE159_EXPECTED_DEV_ROW_COUNT,
) -> Stage159DevQuerySet:
    """Load only serving fields from the exact frozen dev JSONL."""

    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Stage159 dev source must be a file")
    source_sha256 = _file_sha256(resolved)
    if source_sha256 != expected_sha256:
        raise ValueError("Stage159 dev source SHA-256 does not match the frozen dev split")
    queries: list[Stage159DevRuntimeQuery] = []
    private_identities: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _load_exact_json_object(line=line, line_number=line_number)
            query = _project_dev_runtime_query(row=row, line_number=line_number)
            if query.private_identity_sha256 in private_identities:
                raise ValueError("Stage159 dev source contains duplicate sample identities")
            private_identities.add(query.private_identity_sha256)
            queries.append(query)
    if len(queries) != expected_row_count:
        raise ValueError("Stage159 dev source row count does not match the frozen protocol")
    ordered = tuple(
        sorted(
            queries,
            key=lambda query: (
                query.private_identity_sha256,
                query.runtime_query.id,
            ),
        )
    )
    order_sha256 = _digest_lines(query.private_identity_sha256 for query in ordered)
    return Stage159DevQuerySet(
        source_size_bytes=resolved.stat().st_size,
        source_sha256=source_sha256,
        order_sha256=order_sha256,
        queries=ordered,
    )


def build_stage159_workload_plan(
    query_set: Stage159DevQuerySet,
) -> Stage159WorkloadPlan:
    if not query_set.queries:
        raise ValueError("Stage159 workload requires at least one dev query")
    threads = tuple(
        Stage159ThreadWorkload(
            ordinal=(offset // STAGE159_MAX_TURNS_PER_THREAD) + 1,
            queries=query_set.queries[offset : offset + STAGE159_MAX_TURNS_PER_THREAD],
        )
        for offset in range(0, len(query_set.queries), STAGE159_MAX_TURNS_PER_THREAD)
    )
    grouping_sha256 = _digest_lines(
        f"{thread.ordinal}:{','.join(query.private_identity_sha256 for query in thread.queries)}"
        for thread in threads
    )
    return Stage159WorkloadPlan(threads=threads, grouping_sha256=grouping_sha256)


def summarize_stage159_turn_observations(
    observations: Sequence[Stage159TurnObservation],
) -> dict[str, Any]:
    if not observations:
        raise ValueError("Stage159 turn summary requires observations")
    ordered = tuple(
        sorted(observations, key=lambda item: (item.thread_ordinal, item.turn_position))
    )
    by_position = {
        position: [item for item in ordered if item.turn_position == position]
        for position in range(1, STAGE159_MAX_TURNS_PER_THREAD + 1)
    }
    per_position = {
        str(position): {
            "turn_count": len(items),
            "end_to_end_latency_ms": _distribution([item.end_to_end_latency_ms for item in items]),
            "router_generation_latency_ms": _distribution(
                [item.router_generation_latency_ms for item in items]
            ),
            "router_input_token_count": _distribution(
                [item.router_input_token_count for item in items]
            ),
            "router_output_token_count": _distribution(
                [item.router_output_token_count for item in items]
            ),
            "retained_state_bytes": _distribution([item.retained_state_bytes for item in items]),
        }
        for position, items in by_position.items()
    }
    thread_sequences: dict[int, list[Stage159TurnObservation]] = {}
    for observation in ordered:
        thread_sequences.setdefault(observation.thread_ordinal, []).append(observation)
    monotonic_threads = sum(
        all(
            current.completed_turn_count == current.turn_position
            and current.retained_state_bytes > previous.retained_state_bytes
            for previous, current in zip(sequence, sequence[1:], strict=False)
        )
        and sequence[0].completed_turn_count == 1
        and sequence[0].retained_state_bytes > 0
        for sequence in thread_sequences.values()
    )
    action_counts = Counter(item.selected_action for item in ordered)
    terminal_counts = Counter(item.terminal_state for item in ordered)
    http_status_counts = Counter(str(item.http_status) for item in ordered)
    refused_count = sum(item.refused for item in ordered)
    return {
        "turn_count": len(ordered),
        "thread_count": len(thread_sequences),
        "http_status_counts": dict(sorted(http_status_counts.items())),
        "selected_action_counts": dict(sorted(action_counts.items())),
        "terminal_state_counts": dict(sorted(terminal_counts.items())),
        "answer_count": len(ordered) - refused_count,
        "refusal_count": refused_count,
        "answer_rate": round((len(ordered) - refused_count) / len(ordered), 6),
        "refusal_rate": round(refused_count / len(ordered), 6),
        "citation_count": sum(item.citation_count for item in ordered),
        "retrieval_call_count": sum(item.retrieval_call_count for item in ordered),
        "model_decision_count": sum(item.model_decision_count for item in ordered),
        "composition_call_count": sum(item.composition_call_count for item in ordered),
        "verification_call_count": sum(item.verification_call_count for item in ordered),
        "diagnostic_observation_count": sum(item.diagnostic_observation_count for item in ordered),
        "branch_protocol_valid_count": sum(item.branch_protocol_valid for item in ordered),
        "state_growth_monotonic_thread_count": monotonic_threads,
        "all_turns": {
            "end_to_end_latency_ms": _distribution(
                [item.end_to_end_latency_ms for item in ordered]
            ),
            "router_generation_latency_ms": _distribution(
                [item.router_generation_latency_ms for item in ordered]
            ),
            "router_input_token_count": _distribution(
                [item.router_input_token_count for item in ordered]
            ),
            "router_output_token_count": _distribution(
                [item.router_output_token_count for item in ordered]
            ),
            "retained_state_bytes": _distribution([item.retained_state_bytes for item in ordered]),
        },
        "by_turn_position": per_position,
    }


def _load_exact_json_object(*, line: str, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid Stage159 dev JSON object on line {line_number}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected Stage159 dev object on line {line_number}")
    return value


def _project_dev_runtime_query(
    *,
    row: Mapping[str, Any],
    line_number: int,
) -> Stage159DevRuntimeQuery:
    required = (
        "sample_id",
        "question_title",
        "question_text",
        "assigned_split",
        "split_name",
        "protocol_version",
    )
    if any(field not in row for field in required):
        raise ValueError(f"Missing Stage159 dev serving field on line {line_number}")
    if row["assigned_split"] != "dev":
        raise ValueError(f"Non-dev row rejected on line {line_number}")
    if row["split_name"] != STAGE159_SPLIT_NAME:
        raise ValueError(f"Unexpected frozen split name on line {line_number}")
    if row["protocol_version"] != STAGE159_PROTOCOL_VERSION:
        raise ValueError(f"Unexpected frozen protocol version on line {line_number}")
    sample_identity = row["sample_id"]
    title = row["question_title"]
    text = row["question_text"]
    if not isinstance(sample_identity, str) or not sample_identity:
        raise ValueError(f"Invalid private sample identity on line {line_number}")
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError(f"Invalid Stage159 dev query text type on line {line_number}")
    if not title.strip() and not text.strip():
        raise ValueError(f"Empty Stage159 dev query on line {line_number}")
    identity_sha256 = hashlib.sha256(sample_identity.encode("utf-8")).hexdigest()
    return Stage159DevRuntimeQuery(
        private_identity_sha256=identity_sha256,
        runtime_query=PrimeQARuntimeQuery(
            id=f"stage159-dev-{identity_sha256[:16]}",
            title=title,
            text=text,
        ),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _distribution(values: Sequence[float | int]) -> dict[str, float | int]:
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
        "average": round(sum(ordered) / len(ordered), 3),
    }


def _percentile(ordered: Sequence[float], percentile: int) -> float:
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = round((percentile / 100) * (len(ordered) - 1))
    return round(ordered[index], 3)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_lines(lines: Sequence[str] | Any) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
