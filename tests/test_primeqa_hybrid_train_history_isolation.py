from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from scripts.analyze_primeqa_hybrid_train_history_isolation import app
from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_protocol as protocol,
)
from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_validation as validation,
)
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery


def test_workload_builds_paired_stable_hash_threads_and_balanced_orders() -> None:
    samples = tuple(_sample(index) for index in range(5))
    diagnostic_set = protocol.Stage165TrainDiagnosticSet(
        source_size_bytes=1,
        source_sha256="a" * 64,
        order_sha256="b" * 64,
        samples=samples,
    )

    workload = protocol.build_stage165_paired_workload_plan(diagnostic_set)

    assert [len(thread.samples) for thread in workload.threads] == [4, 1]
    assert workload.public_summary()["agent_turn_count"] == 10
    assert workload.public_summary()["first_turn_negative_control_count"] == 2
    assert all(
        set(workload.arm_order(sample)) == {"isolated", "synthetic_history"} for sample in samples
    )
    assert len(workload.grouping_sha256) == 64
    assert len(workload.arm_schedule_sha256) == 64


def test_grouped_folds_keep_duplicate_diagnostic_groups_together() -> None:
    samples = [
        _sample(0, group="shared"),
        _sample(1, group="shared"),
        _sample(2, group="other-a"),
        _sample(3, group="other-b"),
    ]

    folds = protocol.build_stage165_grouped_fold_assignment(samples, fold_count=2)

    assert (
        folds.fold_by_private_identity[samples[0].private_identity_sha256]
        == folds.fold_by_private_identity[samples[1].private_identity_sha256]
    )
    assert sum(folds.row_counts) == 4
    assert sum(folds.group_counts) == 3


def test_pairing_rejects_a_missing_arm() -> None:
    with pytest.raises(ValueError, match="both paired arms"):
        protocol.pair_stage165_observations([_observation(0, arm="isolated")])


def test_pairing_rejects_duplicate_arm_rows() -> None:
    row = _observation(0, arm="isolated")
    with pytest.raises(ValueError, match="duplicate sample arm"):
        protocol.pair_stage165_observations([row, row])


def test_pair_summary_localizes_history_harm_and_safety() -> None:
    rows = [
        _observation(0, arm="isolated", position=2, answerable=True, refused=False),
        _observation(0, arm="synthetic_history", position=2, answerable=True, refused=True),
        _observation(1, arm="isolated", position=3, answerable=True, refused=True),
        _observation(1, arm="synthetic_history", position=3, answerable=True, refused=False),
        _observation(2, arm="isolated", position=2, answerable=False, refused=False),
        _observation(2, arm="synthetic_history", position=2, answerable=False, refused=True),
    ]

    summary = protocol.summarize_stage165_pairs(rows)
    primary = summary["primary_post_first_answerable_effect"]
    safety = summary["unanswerable_post_first_safety_effect"]

    assert primary["isolated_answer_to_synthetic_refusal_count"] == 1
    assert primary["isolated_refusal_to_synthetic_answer_count"] == 1
    assert primary["mcnemar_exact_two_sided_p"] == 1.0
    assert safety["false_answer_rate_difference_isolated_minus_synthetic"] == 1.0


def test_question_alignment_zero_is_in_first_fixed_bin() -> None:
    rows = [
        _observation(
            0,
            arm="isolated",
            position=2,
            answerable=True,
            refused=False,
            question_recall=0.0,
        ),
        _observation(
            0,
            arm="synthetic_history",
            position=2,
            answerable=True,
            refused=True,
            question_recall=0.0,
        ),
    ]

    alignment = protocol.summarize_stage165_pairs(rows)["question_alignment"]

    assert alignment["by_fixed_bin"]["0.00_to_0.25"]["pair_count"] == 1


def test_first_turn_control_requires_context_and_output_equality() -> None:
    exact = [
        _observation(0, arm="isolated", position=1),
        _observation(0, arm="synthetic_history", position=1),
    ]
    changed = [exact[0], replace(exact[1], output_sha256="f" * 64)]

    exact_control = protocol.summarize_stage165_pairs(exact)["first_turn_negative_control"]
    changed_control = protocol.summarize_stage165_pairs(changed)["first_turn_negative_control"]

    assert exact_control["output_exact_count"] == 1
    assert changed_control["output_exact_count"] == 0


def test_private_report_contains_only_hashed_content_free_rows() -> None:
    report = protocol.stage165_private_report(
        [
            _observation(0, arm="isolated"),
            _observation(0, arm="synthetic_history"),
        ]
    )

    assert report["arm_row_count"] == 2
    assert report["pair_count"] == 1
    assert report["contains_raw_question"] is False
    assert "question_text" not in report["rows"][0]
    assert "document_id" not in report["rows"][0]


def test_executor_interleaves_arms_and_closes_every_thread() -> None:
    samples = tuple(_sample(index) for index in range(5))
    diagnostic_set = protocol.Stage165TrainDiagnosticSet(1, "a" * 64, "b" * 64, samples)
    workload = protocol.build_stage165_paired_workload_plan(diagnostic_set)
    folds = protocol.build_stage165_grouped_fold_assignment(samples, fold_count=2)
    session = _FakeSession()
    incrementally_persisted = []

    rows = validation.Stage165PairedWorkloadExecutor(session=session).execute(
        workload=workload,
        folds=folds,
        observation_sink=incrementally_persisted.append,
    )

    assert len(rows) == 10
    assert tuple(incrementally_persisted) == rows
    assert session.opened == {}
    assert session.open_count == session.close_count == 7
    paired = protocol.pair_stage165_observations(rows)
    assert all(pair.context_signatures_exact for pair in paired)
    assert all(
        pair.first_turn_output_exact is True for pair in paired if pair.synthetic_turn_position == 1
    )


def test_public_safe_scanner_detects_forbidden_raw_key() -> None:
    safe = validation._public_safe_contract({"aggregate": {"pair_count": 2}})
    unsafe = validation._public_safe_contract({"question_text": "private"})

    assert safe["public_safe"] is True
    assert unsafe["public_safe"] is False
    assert unsafe["forbidden_keys_found"] == ["question_text"]


def test_all_process_guards_accept_exact_full_train_shape() -> None:
    observations = _full_shape_observations()
    report = _full_shape_report(observations)

    checks = validation._guard_checks(report, observations=observations)

    assert len(checks) == 21
    assert all(check["passed"] for check in checks)


def test_process_guards_reject_test_access_and_context_mismatch() -> None:
    observations = list(_full_shape_observations())
    observations[0] = replace(observations[0], candidate_context_sha256="f" * 64)
    report = _full_shape_report(observations)
    report["execution_boundaries"]["test_loaded"] = True

    checks = {
        check["name"]: check["passed"]
        for check in validation._guard_checks(report, observations=observations)
    }

    assert checks["paired_retrieval_contexts_exact"] is False
    assert checks["development_test_and_runtime_default_closed"] is False


def test_decision_opens_only_dev_candidate_gate_for_strict_train_safe_result() -> None:
    report = _decision_report(
        primary_delta=0.1,
        primary_fold_values=[0.1] * 5,
        f1_delta=0.01,
        citation_delta=2,
        safety_delta=-0.02,
        safety_fold_values=[-0.01, 0.0, -0.02, 0.0, -0.01],
    )

    decision = validation._decision(report)

    assert decision["candidate_eligible_for_frozen_dev_validation"] is True
    assert decision["development_gate_opened"] is True
    assert decision["test_gate_opened"] is False
    assert decision["policy_selected"] is False


def test_decision_rejects_isolation_when_unanswerable_safety_regresses() -> None:
    report = _decision_report(
        primary_delta=0.1,
        primary_fold_values=[0.1] * 5,
        f1_delta=0.01,
        citation_delta=2,
        safety_delta=0.02,
        safety_fold_values=[0.01, 0.0, 0.02, 0.0, 0.01],
    )

    decision = validation._decision(report)

    assert decision["isolated_unanswerable_safety_nonregression"] is False
    assert decision["candidate_eligible_for_frozen_dev_validation"] is False
    assert decision["status"] == "primeqa_hybrid_train_history_isolation_not_train_safe"


def test_unanswerable_transition_labels_follow_synthetic_to_isolated_direction() -> None:
    observations = [
        _observation(1, arm="isolated", position=2, answerable=False, refused=False),
        _observation(1, arm="synthetic_history", position=2, answerable=False, refused=True),
        _observation(2, arm="isolated", position=2, answerable=False, refused=True),
        _observation(2, arm="synthetic_history", position=2, answerable=False, refused=False),
        _observation(3, arm="isolated", position=2, answerable=False, refused=False),
        _observation(3, arm="synthetic_history", position=2, answerable=False, refused=True),
    ]

    safety = protocol.summarize_stage165_pairs(observations)[
        "unanswerable_post_first_safety_effect"
    ]

    assert safety["synthetic_refusal_to_isolated_false_answer_count"] == 2
    assert safety["synthetic_false_answer_to_isolated_refusal_count"] == 1
    assert safety["false_answer_rate_difference_isolated_minus_synthetic"] == 0.333334


def test_visualizations_write_ten_parseable_svgs(tmp_path) -> None:
    observations = [
        _observation(0, arm="isolated", position=1),
        _observation(0, arm="synthetic_history", position=1),
        _observation(1, arm="isolated", position=2, answerable=True, refused=False),
        _observation(1, arm="synthetic_history", position=2, answerable=True, refused=True),
        _observation(2, arm="isolated", position=2, answerable=False, refused=True),
        _observation(2, arm="synthetic_history", position=2, answerable=False, refused=True),
    ]
    report = {
        "paired_diagnostics": protocol.summarize_stage165_pairs(observations),
        "guard_checks": [{"name": "synthetic", "passed": True}],
    }

    visuals = validation.write_stage165_visualizations(report=report, output_dir=tmp_path)

    assert len(visuals) == 10
    for visual in visuals:
        ET.parse(visual.path)


def test_cli_requires_explicit_full_train_confirmation() -> None:
    result = CliRunner().invoke(app, ["--model-snapshot", "."])

    assert result.exit_code != 0
    assert "user-confirmed-full-train-pairing" in result.output


def _sample(index: int, *, group: str | None = None) -> protocol.Stage165TrainSample:
    identity = hashlib.sha256(f"sample-{index}".encode()).hexdigest()
    return protocol.Stage165TrainSample(
        private_identity_sha256=identity,
        query_digest_sha256=hashlib.sha256(f"query-{index}".encode()).hexdigest(),
        diagnostic_group_sha256=hashlib.sha256((group or f"group-{index}").encode()).hexdigest(),
        answerable=index % 2 == 0,
        gold_answer="answer" if index % 2 == 0 else "",
        gold_document_id=f"doc-{index}" if index % 2 == 0 else None,
        gold_document_sha256=(
            hashlib.sha256(f"doc-{index}".encode()).hexdigest() if index % 2 == 0 else None
        ),
        question_route="other",
        split_subtype="random_group",
        runtime_query=PrimeQARuntimeQuery(
            id=f"stage165-{index}",
            title=f"Title {index}",
            text=f"Question {index}",
        ),
    )


def _observation(
    index: int,
    *,
    arm: protocol.Stage165Arm,
    position: int = 1,
    answerable: bool = True,
    refused: bool = False,
    question_recall: float | None = 0.6,
    fold_id: int | None = None,
) -> protocol.Stage165ArmObservation:
    identity = hashlib.sha256(f"observation-{index}".encode()).hexdigest()
    gold_visible = answerable
    history = 0 if arm == "isolated" else position - 1
    return protocol.Stage165ArmObservation(
        private_identity_sha256=identity,
        query_digest_sha256=hashlib.sha256(f"query-{index}".encode()).hexdigest(),
        diagnostic_group_sha256=hashlib.sha256(f"group-{index}".encode()).hexdigest(),
        gold_document_sha256=(
            hashlib.sha256(f"doc-{index}".encode()).hexdigest() if answerable else None
        ),
        fold_id=index % 5 if fold_id is None else fold_id,
        synthetic_thread_ordinal=(index // 4) + 1,
        synthetic_turn_position=position,
        arm=arm,
        arm_order_position=1 if arm == "isolated" else 2,
        answerable=answerable,
        question_route="other",
        split_subtype="random_group",
        selected_action="refuse_insufficient_evidence" if refused else "compose_grounded_answer",
        terminal_state="refuse" if refused else "complete",
        refused=refused,
        history_turn_count_before=history,
        completed_turn_count_after=history + 1,
        retained_state_bytes_after=(history + 1) * 100,
        candidate_pool_count=400,
        generation_context_count=10,
        verification_context_count=200,
        candidate_context_sha256="a" * 64,
        generation_context_sha256="b" * 64,
        verification_context_sha256="c" * 64,
        output_sha256="d" * 64,
        gold_candidate_rank=1 if gold_visible else None,
        gold_generation_rank=1 if gold_visible else None,
        gold_verification_rank=1 if gold_visible else None,
        gold_cited=answerable and not refused,
        citation_count=1 if answerable and not refused else 0,
        answer_token_f1=1.0 if answerable and not refused else (0.0 if answerable else None),
        top_candidate_score=1.0,
        gold_candidate_score=1.0 if gold_visible else None,
        question_token_recall_in_gold_prompt=question_recall if gold_visible else None,
        answer_token_recall_in_gold_prompt=1.0 if gold_visible else None,
        answer_exact_span_visible=True if gold_visible else None,
        router_input_token_count=2000 + (history * 100),
        router_output_token_count=5,
        router_generation_latency_ms=100.0 + (history * 10),
        end_to_end_latency_ms=120.0 + (history * 10),
        retrieval_call_count=1,
        model_decision_count=1,
        composition_call_count=0 if refused else 1,
        verification_call_count=0 if refused else 1,
        diagnostic_observation_count=0 if refused else 1,
        retry_action_count=0,
        fallback_action_count=0,
    )


class _FakeSession:
    def __init__(self) -> None:
        self.opened: dict[str, int] = {}
        self.open_count = 0
        self.close_count = 0

    def open_thread(self, handle: str) -> None:
        assert handle not in self.opened
        self.opened[handle] = 0
        self.open_count += 1

    def close_thread(self, handle: str) -> None:
        del self.opened[handle]
        self.close_count += 1

    def measure_turn(
        self,
        *,
        handle: str,
        sample: protocol.Stage165TrainSample,
        fold_id: int,
        synthetic_thread_ordinal: int,
        synthetic_turn_position: int,
        arm: protocol.Stage165Arm,
        arm_order_position: int,
    ) -> protocol.Stage165ArmObservation:
        history = self.opened[handle]
        self.opened[handle] += 1
        index = int(sample.runtime_query.id.rsplit("-", 1)[1])
        return replace(
            _observation(
                index,
                arm=arm,
                position=synthetic_turn_position,
                answerable=sample.answerable,
                fold_id=fold_id,
            ),
            private_identity_sha256=sample.private_identity_sha256,
            query_digest_sha256=sample.query_digest_sha256,
            diagnostic_group_sha256=sample.diagnostic_group_sha256,
            gold_document_sha256=sample.gold_document_sha256,
            synthetic_thread_ordinal=synthetic_thread_ordinal,
            arm_order_position=arm_order_position,
            history_turn_count_before=history,
            completed_turn_count_after=history + 1,
        )


def _full_shape_observations() -> tuple[protocol.Stage165ArmObservation, ...]:
    rows = []
    for index in range(562):
        position = (index % 4) + 1
        answerable = index < 370
        for arm in ("isolated", "synthetic_history"):
            rows.append(
                _observation(
                    index,
                    arm=arm,
                    position=position,
                    answerable=answerable,
                    refused=False,
                    fold_id=index % 5,
                )
            )
    return tuple(rows)


def _full_shape_report(observations) -> dict:
    frozen = {"protocol_id": "test"}
    diagnostics = protocol.summarize_stage165_pairs(observations)
    return {
        "user_confirmation": {
            "full_train_pairing_confirmed": True,
            "selected_option": "A",
        },
        "frozen_protocol": frozen,
        "frozen_protocol_sha256": validation._canonical_json_sha256(frozen),
        "source_authorization": {
            "authorized": True,
            "stage164_status": validation._EXPECTED_STAGE164_STATUS,
            "stage164_next_direction": validation._EXPECTED_STAGE164_NEXT,
            "fingerprints": {
                name: {"sha256": sha} for name, sha in validation._EXPECTED_SOURCE_HASHES.items()
            },
        },
        "train_diagnostic_protocol": {
            "source_sha256": protocol.STAGE165_EXPECTED_TRAIN_SHA256,
            "train_row_count": 562,
            "answerable_count": 370,
            "unanswerable_count": 192,
            "stable_order_sha256": protocol.STAGE165_EXPECTED_ORDER_SHA256,
            "assigned_split": "train",
            "dev_loaded": False,
            "test_loaded": False,
        },
        "workload_plan": {
            "unique_sample_count": 562,
            "agent_turn_count": 1124,
            "thread_count": 141,
            "full_four_turn_thread_count": 140,
            "trailing_thread_turn_count": 2,
            "first_turn_negative_control_count": 141,
            "post_first_turn_primary_count": 421,
            "grouping_sha256": protocol.STAGE165_EXPECTED_GROUPING_SHA256,
            "arm_schedule_sha256": protocol.STAGE165_EXPECTED_ARM_SCHEDULE_SHA256,
            "arm_first_counts": {"isolated": 284, "synthetic_history": 278},
        },
        "grouped_fold_protocol": {
            "fold_count": 5,
            "group_count": 534,
            "group_counts": {"0": 107, "1": 107, "2": 106, "3": 107, "4": 107},
            "row_counts": {"0": 113, "1": 113, "2": 112, "3": 112, "4": 112},
            "fit_models": False,
            "select_policy": False,
            "tune_thresholds": False,
            "assignment_sha256": protocol.STAGE165_EXPECTED_FOLD_ASSIGNMENT_SHA256,
        },
        "runtime": {
            "single_runtime_instance": True,
            "resource_factory_build_count": 1,
            "model_generation_call_count": 1125,
            "session": {
                "open_count": 703,
                "close_count": 703,
                "opened_thread_count_after_run": 0,
            },
        },
        "paired_diagnostics": diagnostics,
        "private_diagnostic_artifact_contract": {
            "arm_row_count": 1124,
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "contains_raw_model_output": False,
            "public_report_contains_case_rows": False,
        },
        "execution_boundaries": {
            "model_fit": False,
            "threshold_tuned": False,
            "policy_selected": False,
            "development_loaded": False,
            "test_loaded": False,
            "runtime_registered_as_default": False,
            "remote_exposure": False,
            "http_server_started": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
        },
        "current_source_fingerprints_before": {"same": True},
        "current_source_fingerprints_after": {"same": True},
    }


def _decision_report(
    *,
    primary_delta: float,
    primary_fold_values: list[float],
    f1_delta: float,
    citation_delta: int,
    safety_delta: float,
    safety_fold_values: list[float],
) -> dict:
    return {
        "guard_checks": [{"name": "all", "passed": True}],
        "paired_diagnostics": {
            "primary_post_first_answerable_effect": {
                "refusal_rate_difference_synthetic_minus_isolated": primary_delta,
                "average_answer_f1_difference_isolated_minus_synthetic": f1_delta,
                "gold_citation_difference_isolated_minus_synthetic": citation_delta,
            },
            "unanswerable_post_first_safety_effect": {
                "false_answer_rate_difference_isolated_minus_synthetic": safety_delta,
            },
            "grouped_fold_stability": {
                "primary_answerable_refusal_delta_direction": {
                    "positive_count": sum(value > 0 for value in primary_fold_values),
                    "zero_count": sum(value == 0 for value in primary_fold_values),
                    "negative_count": sum(value < 0 for value in primary_fold_values),
                },
                "unanswerable_false_answer_delta_direction": {
                    "positive_count": sum(value > 0 for value in safety_fold_values),
                    "zero_count": sum(value == 0 for value in safety_fold_values),
                    "negative_count": sum(value < 0 for value in safety_fold_values),
                },
            },
        },
    }
