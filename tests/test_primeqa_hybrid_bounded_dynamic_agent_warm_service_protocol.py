from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Thread

import pytest

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_warm_service_protocol import (
    STAGE159_EXPECTED_DEV_ROW_COUNT,
    STAGE159_MAX_TURNS_PER_THREAD,
    Stage159DevQuerySet,
    Stage159DevRuntimeQuery,
    Stage159RuntimeObservationGate,
    Stage159TurnObservation,
    build_stage159_workload_plan,
    load_stage159_dev_runtime_queries,
    summarize_stage159_turn_observations,
)
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery


def test_query_only_loader_uses_exact_dev_source_and_stable_hash_order(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    _write_rows(path, [_row(index) for index in range(5)])
    source_hash = _sha256(path)

    query_set = load_stage159_dev_runtime_queries(
        path,
        expected_sha256=source_hash,
        expected_row_count=5,
    )

    assert len(query_set.queries) == 5
    assert [item.private_identity_sha256 for item in query_set.queries] == sorted(
        item.private_identity_sha256 for item in query_set.queries
    )
    assert all(item.runtime_query.id.startswith("stage159-dev-") for item in query_set.queries)
    assert query_set.public_summary() == {
        "source_size_bytes": path.stat().st_size,
        "source_sha256": source_hash,
        "dev_query_count": 5,
        "stable_order_sha256": query_set.order_sha256,
        "assigned_split": "dev",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "label_fields_used_for_selection": False,
        "label_fields_projected_into_runtime": False,
        "label_fields_used_for_metrics": False,
    }
    serialized = json.dumps(query_set.public_summary())
    assert "Question " not in serialized
    assert "Answer " not in serialized


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "assigned_split": "test"}), "Non-dev"),
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "sample_id": rows[0]["sample_id"]}),
            "duplicate",
        ),
        (lambda rows: rows.__setitem__(1, {**rows[1], "question_text": 3}), "text type"),
    ],
)
def test_query_only_loader_rejects_nonexact_rows(tmp_path: Path, mutation, message: str) -> None:
    rows = [_row(index) for index in range(3)]
    mutation(rows)
    path = tmp_path / "dev.jsonl"
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=message):
        load_stage159_dev_runtime_queries(
            path,
            expected_sha256=_sha256(path),
            expected_row_count=3,
        )


def test_full_dev_plan_is_thirty_four_turn_threads_plus_one_trailing_turn() -> None:
    query_set = _query_set(STAGE159_EXPECTED_DEV_ROW_COUNT)

    plan = build_stage159_workload_plan(query_set)

    assert len(plan.threads) == 31
    assert [len(thread.queries) for thread in plan.threads[:30]] == [4] * 30
    assert len(plan.threads[-1].queries) == 1
    assert plan.public_summary() == {
        "grouping_policy": "stable_hash_order_consecutive_groups_of_four",
        "semantic_relationship_policy": "synthetic_grouping_not_natural_conversation",
        "thread_count": 31,
        "full_four_turn_thread_count": 30,
        "trailing_thread_turn_count": 1,
        "turn_count": 121,
        "turn_position_counts": {"1": 31, "2": 30, "3": 30, "4": 30},
        "grouping_sha256": plan.grouping_sha256,
    }
    assert STAGE159_MAX_TURNS_PER_THREAD == 4


def test_turn_summary_aggregates_positions_branches_and_monotonic_state() -> None:
    observations = [
        _observation(thread=1, position=1, refused=True, retained=100),
        _observation(thread=1, position=2, refused=False, retained=220),
        _observation(thread=1, position=3, refused=True, retained=330),
        _observation(thread=1, position=4, refused=False, retained=460),
        _observation(thread=2, position=1, refused=True, retained=90),
    ]

    summary = summarize_stage159_turn_observations(observations)

    assert summary["turn_count"] == 5
    assert summary["thread_count"] == 2
    assert summary["answer_count"] == 2
    assert summary["refusal_count"] == 3
    assert summary["branch_protocol_valid_count"] == 5
    assert summary["state_growth_monotonic_thread_count"] == 2
    assert summary["by_turn_position"]["1"]["turn_count"] == 2
    assert summary["by_turn_position"]["4"]["retained_state_bytes"]["average"] == 460.0
    assert summary["retrieval_call_count"] == summary["model_decision_count"] == 5
    assert summary["composition_call_count"] == 2


def test_runtime_observation_gate_pauses_one_turn_without_timeout() -> None:
    runtime = _FakeRuntime()
    gate = Stage159RuntimeObservationGate(runtime)  # type: ignore[arg-type]
    result = {}
    gate.arm()

    def run() -> None:
        result["value"] = gate.run_turn(
            opaque_thread_handle="thread",
            question=PrimeQARuntimeQuery(id="query", text="Question"),
        )

    worker = Thread(target=run)
    worker.start()
    gate.wait_until_entered()
    assert runtime.run_count == 0
    gate.release()
    worker.join()

    assert result["value"] == "completed"
    assert runtime.run_count == 1
    with pytest.raises(RuntimeError, match="not armed"):
        gate.release()


def _row(index: int) -> dict:
    return {
        "sample_id": f"dev:{index}",
        "question_title": f"Title {index}",
        "question_text": f"Question {index}",
        "assigned_split": "dev",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "answer": f"Answer {index}",
        "answerable": index % 2 == 0,
        "answer_doc_id": f"doc-{index}",
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _query_set(count: int) -> Stage159DevQuerySet:
    queries = tuple(
        Stage159DevRuntimeQuery(
            private_identity_sha256=f"{index:064x}",
            runtime_query=PrimeQARuntimeQuery(
                id=f"query-{index}",
                title=f"Title {index}",
                text=f"Question {index}",
            ),
        )
        for index in range(count)
    )
    return Stage159DevQuerySet(
        source_size_bytes=1,
        source_sha256="a" * 64,
        order_sha256="b" * 64,
        queries=queries,
    )


def _observation(
    *,
    thread: int,
    position: int,
    refused: bool,
    retained: int,
) -> Stage159TurnObservation:
    compose = not refused
    return Stage159TurnObservation(
        thread_ordinal=thread,
        turn_position=position,
        http_status=200,
        refused=refused,
        citation_count=0 if refused else 1,
        terminal_state="refuse" if refused else "complete",
        selected_action=("refuse_insufficient_evidence" if refused else "compose_grounded_answer"),
        completed_turn_count=position,
        retained_state_bytes=retained,
        router_input_token_count=2000 + position,
        router_output_token_count=10,
        router_generation_latency_ms=1000.0 + position,
        end_to_end_latency_ms=1200.0 + position,
        retrieval_call_count=1,
        model_decision_count=1,
        composition_call_count=int(compose),
        verification_call_count=int(compose),
        diagnostic_observation_count=int(compose),
    )


class _FakeRuntime:
    last_public_trace = None

    def __init__(self) -> None:
        self.run_count = 0

    def topology(self):
        return {}

    def open_thread(self, handle):
        return handle

    def close_thread(self, handle):
        return handle

    def thread_summary(self, handle):
        return handle

    def run_turn(self, *, opaque_thread_handle, question):
        _ = opaque_thread_handle, question
        self.run_count += 1
        return "completed"
