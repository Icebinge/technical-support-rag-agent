from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol as stage160,
)


def test_stage160_loader_projects_runtime_query_without_gold(tmp_path) -> None:
    path = tmp_path / "dev.jsonl"
    rows = [
        _row(
            sample_id="sample-b",
            title="Install service",
            text="How do I install it?",
            answerable=True,
            answer="Run the installer.",
            answer_doc_id="doc-b",
        ),
        _row(
            sample_id="sample-a",
            title="Unknown state",
            text="Is this supported?",
            answerable=False,
            answer="",
            answer_doc_id=None,
        ),
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    result = stage160.load_stage160_dev_diagnostic_samples(
        path,
        expected_sha256=expected_sha,
        expected_row_count=2,
    )

    assert len(result.samples) == 2
    assert result.samples[0].private_identity_sha256 < result.samples[1].private_identity_sha256
    assert all(sample.runtime_query.id.startswith("stage160-dev-") for sample in result.samples)
    runtime_payload = result.samples[0].runtime_query.model_dump()
    assert set(runtime_payload) == {"id", "title", "text"}
    assert "answer" not in runtime_payload
    assert "document" not in runtime_payload
    assert result.public_summary()["gold_fields_projected_into_runtime"] is False


def test_stage160_loader_rejects_wrong_hash(tmp_path) -> None:
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        stage160.load_stage160_dev_diagnostic_samples(
            path,
            expected_sha256="0" * 64,
            expected_row_count=1,
        )


def test_stage160_workload_uses_consecutive_groups_of_four(tmp_path) -> None:
    diagnostic_set = _diagnostic_set(tmp_path, count=9)

    plan = stage160.build_stage160_workload_plan(diagnostic_set)

    assert [len(thread.samples) for thread in plan.threads] == [4, 4, 1]
    assert [thread.ordinal for thread in plan.threads] == [1, 2, 3]
    assert plan.ordered_samples == diagnostic_set.samples


def test_stage160_grouped_folds_keep_duplicate_group_together(tmp_path) -> None:
    diagnostic_set = _diagnostic_set(tmp_path, count=12)
    samples = list(diagnostic_set.samples)
    samples[1] = replace(
        samples[1],
        diagnostic_group_sha256=samples[0].diagnostic_group_sha256,
    )

    assignment = stage160.build_stage160_grouped_fold_assignment(samples, fold_count=3)

    assert (
        assignment.fold_by_private_identity[samples[0].private_identity_sha256]
        == (assignment.fold_by_private_identity[samples[1].private_identity_sha256])
    )
    assert sum(assignment.row_counts) == 12
    assert max(assignment.row_counts) - min(assignment.row_counts) <= 1
    assert len(assignment.assignment_sha256) == 64


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        (
            {
                "answerable": True,
                "refused": True,
                "gold_candidate_rank": None,
                "gold_generation_rank": None,
            },
            "answerable_refusal_gold_absent_candidate_pool",
        ),
        (
            {
                "answerable": True,
                "refused": True,
                "gold_candidate_rank": 3,
                "gold_generation_rank": None,
            },
            "answerable_refusal_gold_absent_generation_top10",
        ),
        (
            {
                "answerable": True,
                "refused": True,
                "gold_candidate_rank": 3,
                "gold_generation_rank": 3,
                "selected_action": "refuse_insufficient_evidence",
            },
            "answerable_refusal_gold_visible_model_refused",
        ),
        (
            {
                "answerable": True,
                "refused": True,
                "gold_candidate_rank": 3,
                "gold_generation_rank": 3,
                "selected_action": "compose_grounded_answer",
            },
            "answerable_refusal_after_compose",
        ),
        (
            {"answerable": False, "refused": False},
            "unanswerable_false_answer",
        ),
        (
            {"answerable": False, "refused": True},
            "unanswerable_correct_refusal",
        ),
    ],
)
def test_stage160_failure_buckets(updates, expected) -> None:
    observation = replace(_observation(), **updates)

    assert observation.failure_bucket == expected


def test_stage160_summary_reports_quality_latency_and_fold_stability() -> None:
    observations = [
        replace(
            _observation(index=index),
            fold_id=index % 5,
            answerable=index < 6,
            refused=index in {0, 1, 2, 6, 7},
            selected_action=(
                "refuse_insufficient_evidence"
                if index in {0, 1, 2, 6, 7}
                else "compose_grounded_answer"
            ),
            gold_candidate_rank=(1 if index < 5 else None),
            gold_generation_rank=(1 if index < 4 else None),
            gold_verification_rank=(1 if index < 4 else None),
            gold_cited=index in {3},
            answer_token_f1=(0.8 if index == 3 else 0.0) if index < 6 else None,
            router_input_token_count=2000 + index * 100,
            router_generation_latency_ms=1000.0 + index * 200,
            end_to_end_latency_ms=(13000.0 if index == 9 else 1200.0 + index * 200),
            retained_state_bytes=500 + index * 100,
        )
        for index in range(10)
    ]

    summary = stage160.summarize_stage160_observations(observations)

    assert summary["overview"]["case_count"] == 10
    assert summary["overview"]["answerable_count"] == 6
    assert summary["quality_diagnostics"]["answerable_refusal_rate"] == 0.5
    assert summary["quality_diagnostics"]["unanswerable_false_answer_rate"] == 0.5
    assert summary["latency_diagnostics"]["stage159_p95_exceedance_count"] == 1
    assert summary["fold_diagnostic_stability"]["fold_count"] == 5
    assert summary["fold_diagnostic_stability"]["fit_models"] is False
    assert set(summary["by_turn_position"]) == {"1", "2", "3", "4"}


def test_stage160_private_report_contains_hashes_but_no_raw_content() -> None:
    report = stage160.stage160_private_report([_observation()])
    serialized = json.dumps(report, sort_keys=True)

    assert report["row_count"] == 1
    assert report["contains_hashed_sample_identity"] is True
    assert report["contains_raw_question"] is False
    assert "private_identity_sha256" in report["rows"][0]
    assert "question_text" not in serialized
    assert "answer_doc_id" not in serialized
    assert len(stage160.canonical_json_sha256(report)) == 64


def _diagnostic_set(tmp_path, *, count: int):
    path = tmp_path / "dev.jsonl"
    rows = [
        _row(
            sample_id=f"sample-{index}",
            title=f"Question {index}",
            text=f"How to perform operation {index}?",
            answerable=index % 2 == 0,
            answer=f"Answer {index}" if index % 2 == 0 else "",
            answer_doc_id=f"doc-{index}" if index % 2 == 0 else None,
        )
        for index in range(count)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return stage160.load_stage160_dev_diagnostic_samples(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_row_count=count,
    )


def _row(
    *,
    sample_id: str = "sample-1",
    title: str = "Title",
    text: str = "Question text",
    answerable: bool = True,
    answer: str = "Answer text",
    answer_doc_id: str | None = "doc-1",
) -> dict:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": "dev",
        "split_subtype": "random_grouped",
        "source_split": "DEV",
        "sample_id": sample_id,
        "question_id": sample_id,
        "question_title": title,
        "question_text": text,
        "answerable": answerable,
        "answer": answer,
        "answer_doc_id": answer_doc_id,
        "candidate_doc_ids": [answer_doc_id] if answer_doc_id else [],
        "answer_span": {"start_offset": 0, "end_offset": len(answer)},
    }


def _observation(index: int = 0) -> stage160.Stage160CaseObservation:
    return stage160.Stage160CaseObservation(
        private_identity_sha256=f"{index + 1:064x}",
        query_digest_sha256=f"{index + 2:064x}",
        diagnostic_group_sha256=f"{index + 3:064x}",
        gold_document_sha256=f"{index + 4:064x}",
        fold_id=0,
        thread_ordinal=(index // 4) + 1,
        turn_position=(index % 4) + 1,
        question_route="how_to_or_lookup",
        split_subtype="random_grouped",
        answerable=True,
        selected_action="refuse_insufficient_evidence",
        terminal_state="refuse",
        refused=True,
        candidate_pool_count=400,
        generation_context_count=10,
        verification_context_count=200,
        gold_candidate_rank=2,
        gold_generation_rank=2,
        gold_verification_rank=2,
        gold_cited=False,
        citation_count=0,
        answer_token_f1=0.0,
        top_candidate_score=1.0,
        gold_candidate_score=0.9,
        router_input_token_count=2400,
        router_output_token_count=11,
        router_generation_latency_ms=1500.0,
        end_to_end_latency_ms=1700.0,
        retained_state_bytes=600,
        completed_turn_count=(index % 4) + 1,
    )
