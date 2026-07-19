from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from scripts.analyze_primeqa_hybrid_train_history_isolation_sharded import (
    app as orchestrator_app,
)
from scripts.run_primeqa_hybrid_train_history_isolation_shard import app as shard_app
from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_protocol as protocol,
)
from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_sharded_validation as sharded,
)
from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_sharding_protocol as sharding,
)
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery


def test_sharding_plan_uses_contiguous_twelve_thread_boundaries() -> None:
    workload = _workload(100)

    plan = sharding.build_stage165_sharding_plan(workload)

    assert [len(shard.threads) for shard in plan.shards] == [12, 12, 1]
    assert [(shard.start_thread_ordinal, shard.end_thread_ordinal) for shard in plan.shards] == [
        (1, 12),
        (13, 24),
        (25, 25),
    ]
    assert plan.pair_count == 100
    assert plan.arm_row_count == 200
    assert plan.public_summary()["timeout"] is False
    assert plan.public_summary()["retry"] is False


def test_shard_observation_sequence_round_trips_exact_jsonl(tmp_path: Path) -> None:
    workload = _workload(8)
    shard = sharding.build_stage165_sharding_plan(workload).shards[0]
    observations = _observations_for_shard(shard=shard, workload=workload)
    path = tmp_path / "observations.jsonl"
    path.write_text("", encoding="utf-8")

    for observation in observations:
        sharding.write_stage165_observation_jsonl_row(
            path=path,
            observation=observation,
        )
    persisted = sharding.load_stage165_observation_jsonl(path)
    sharding.validate_stage165_shard_observations(
        shard=shard,
        workload=workload,
        observations=persisted,
    )

    assert persisted == observations
    assert sharding.stage165_observation_sequence_sha256(persisted) == (
        sharding.stage165_observation_sequence_sha256(observations)
    )


def test_shard_observation_sequence_rejects_reordered_arms() -> None:
    workload = _workload(4)
    shard = sharding.build_stage165_sharding_plan(workload).shards[0]
    observations = list(_observations_for_shard(shard=shard, workload=workload))
    observations[0], observations[1] = observations[1], observations[0]

    with pytest.raises(ValueError, match="sequence is not exact"):
        sharding.validate_stage165_shard_observations(
            shard=shard,
            workload=workload,
            observations=observations,
        )


def test_process_orchestrator_stops_at_first_nonzero_shard(tmp_path: Path) -> None:
    plan = sharding.build_stage165_sharding_plan(_workload(100))
    runner = _FakeRunner(exit_codes={1: 0, 2: 7, 3: 0})
    artifact_paths = sharded.Stage165ShardedArtifactPaths(tmp_path)

    outcome = sharded.Stage165ShardProcessOrchestrator(runner=runner).execute(
        shards=plan.shards,
        artifact_paths=artifact_paths,
        cwd=tmp_path,
        command_factory=lambda shard: ("python", str(shard.ordinal)),
        result_loader=lambda public, observations, shard: (
            {"shard": shard.public_summary()},
            (),
        ),
        total_agent_turn_count=plan.arm_row_count,
    )

    assert runner.attempted_ordinals == [1, 2]
    assert [result.shard_ordinal for result in outcome.process_results] == [1, 2]
    assert len(outcome.shard_reports) == 1
    assert outcome.failure["failed_shard_ordinal"] == 2
    assert outcome.failure["automatic_retry"] is False
    assert outcome.failure["continued_to_later_shard"] is False


def test_process_orchestrator_stops_at_invalid_success_artifact(
    tmp_path: Path,
) -> None:
    plan = sharding.build_stage165_sharding_plan(_workload(100))
    runner = _FakeRunner(exit_codes={1: 0, 2: 0, 3: 0})

    def loader(public: Path, observations: Path, shard):
        if shard.ordinal == 2:
            raise ValueError("invalid artifact")
        return {"shard": shard.public_summary()}, ()

    outcome = sharded.Stage165ShardProcessOrchestrator(runner=runner).execute(
        shards=plan.shards,
        artifact_paths=sharded.Stage165ShardedArtifactPaths(tmp_path),
        cwd=tmp_path,
        command_factory=lambda shard: ("python", str(shard.ordinal)),
        result_loader=loader,
        total_agent_turn_count=plan.arm_row_count,
    )

    assert runner.attempted_ordinals == [1, 2]
    assert outcome.failure["reason"] == "artifact_validation:ValueError"


def test_real_process_runner_passes_no_timeout(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sharded.subprocess, "run", fake_run)

    result = sharded.SequentialSubprocessStage165ShardRunner().run(
        command=("python", "child.py"),
        cwd=tmp_path,
        shard_ordinal=1,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.exit_code == 0
    assert "timeout" not in captured["kwargs"]
    assert captured["kwargs"]["check"] is False


def test_new_artifact_directory_rejects_existing_content(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new or empty"):
        sharded._prepare_new_artifact_directory(tmp_path)


def test_shard_guards_accept_exact_persisted_process_shape(monkeypatch) -> None:
    workload = _workload(8)
    plan = sharding.build_stage165_sharding_plan(workload)
    shard = plan.shards[0]
    observations = _observations_for_shard(shard=shard, workload=workload)
    monkeypatch.setattr(
        sharded,
        "STAGE165_EXPECTED_SHARD_ASSIGNMENT_SHA256",
        plan.assignment_sha256,
    )
    report = {
        "user_confirmation": {
            "selected_option": "A_12_contiguous_process_shards",
            "shard_execution_confirmed": True,
        },
        "source_authorization": {"authorized": True},
        "sharding_plan": plan.public_summary(),
        "shard": shard.public_summary(),
        "execution": {
            "observation_sequence_sha256": (
                sharding.stage165_observation_sequence_sha256(observations)
            ),
            "persisted_sequence_sha256": (
                sharding.stage165_observation_sequence_sha256(observations)
            ),
            "resource_factory_build_count": 1,
            "warmup_generation_count": 1,
            "model_generation_call_count": shard.arm_row_count + 1,
            "session_open_count": shard.pair_count + len(shard.threads),
            "session_close_count": shard.pair_count + len(shard.threads),
            "session_opened_thread_count_after_run": 0,
        },
        "private_artifact_contract": {
            "forbidden_keys_found": [],
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "contains_raw_model_output": False,
        },
        "execution_boundaries": sharded._closed_boundaries(),
        "current_source_fingerprints_before": {"same": True},
        "current_source_fingerprints_after": {"same": True},
    }

    checks = sharded._shard_guard_checks(
        report,
        shard=shard,
        observations=observations,
        persisted=observations,
    )

    assert len(checks) == 12
    assert all(check["passed"] for check in checks)


def test_merged_guards_accept_exact_sharded_full_shape() -> None:
    observations = _full_shape_observations()
    report, shard_reports = _merged_guard_report(observations)

    checks = sharded._merged_guard_checks(
        report,
        observations=observations,
        shard_reports=shard_reports,
    )

    assert len(checks) == 23
    assert all(check["passed"] for check in checks)


def test_merged_guards_reject_missing_process_and_test_access() -> None:
    observations = _full_shape_observations()
    report, shard_reports = _merged_guard_report(observations)
    report["process_results"].pop()
    report["execution_boundaries"]["test_loaded"] = True

    checks = {
        check["name"]: check["passed"]
        for check in sharded._merged_guard_checks(
            report,
            observations=observations,
            shard_reports=shard_reports,
        )
    }

    assert checks["all_shard_processes_completed_once"] is False
    assert checks["development_test_and_runtime_default_closed"] is False


def test_shard_cli_requires_parent_confirmation() -> None:
    result = CliRunner().invoke(
        shard_app,
        [
            "--model-snapshot",
            ".",
            "--shard-ordinal",
            "1",
            "--output",
            "public.json",
            "--observation-jsonl",
            "observations.jsonl",
        ],
    )

    assert result.exit_code != 0
    assert "user-confirmed-stage165-shard" in result.output


def test_orchestrator_cli_requires_exact_option_a_confirmation() -> None:
    result = CliRunner().invoke(orchestrator_app, ["--model-snapshot", "."])

    assert result.exit_code != 0
    assert "user-confirmed-12-process-sharding" in result.output


class _FakeRunner:
    def __init__(self, *, exit_codes: dict[int, int]) -> None:
        self.exit_codes = exit_codes
        self.attempted_ordinals: list[int] = []

    def run(
        self,
        *,
        command,
        cwd,
        shard_ordinal,
        stdout_path,
        stderr_path,
    ) -> sharded.Stage165ShardProcessResult:
        self.attempted_ordinals.append(shard_ordinal)
        return sharded.Stage165ShardProcessResult(
            shard_ordinal=shard_ordinal,
            exit_code=self.exit_codes[shard_ordinal],
            duration_seconds=1.0,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


def _workload(count: int) -> protocol.Stage165PairedWorkloadPlan:
    samples = tuple(_sample(index) for index in range(count))
    diagnostic_set = protocol.Stage165TrainDiagnosticSet(
        source_size_bytes=1,
        source_sha256="a" * 64,
        order_sha256="b" * 64,
        samples=samples,
    )
    return protocol.build_stage165_paired_workload_plan(diagnostic_set)


def _sample(index: int) -> protocol.Stage165TrainSample:
    identity = hashlib.sha256(f"sample-{index}".encode()).hexdigest()
    return protocol.Stage165TrainSample(
        private_identity_sha256=identity,
        query_digest_sha256=hashlib.sha256(f"query-{index}".encode()).hexdigest(),
        diagnostic_group_sha256=hashlib.sha256(f"group-{index}".encode()).hexdigest(),
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


def _observations_for_shard(
    *,
    shard: sharding.Stage165ShardSpec,
    workload: protocol.Stage165PairedWorkloadPlan,
) -> tuple[protocol.Stage165ArmObservation, ...]:
    observations = []
    for thread in shard.threads:
        for position, sample in enumerate(thread.samples, start=1):
            index = int(sample.runtime_query.id.rsplit("-", 1)[1])
            for arm_order_position, arm in enumerate(workload.arm_order(sample), start=1):
                observations.append(
                    replace(
                        _observation(index, arm=arm, position=position),
                        private_identity_sha256=sample.private_identity_sha256,
                        query_digest_sha256=sample.query_digest_sha256,
                        diagnostic_group_sha256=sample.diagnostic_group_sha256,
                        synthetic_thread_ordinal=thread.ordinal,
                        arm_order_position=arm_order_position,
                    )
                )
    return tuple(observations)


def _observation(
    index: int,
    *,
    arm: protocol.Stage165Arm,
    position: int,
    answerable: bool = True,
    fold_id: int | None = None,
) -> protocol.Stage165ArmObservation:
    identity = hashlib.sha256(f"observation-{index}".encode()).hexdigest()
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
        selected_action="compose_grounded_answer",
        terminal_state="complete",
        refused=False,
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
        gold_candidate_rank=1 if answerable else None,
        gold_generation_rank=1 if answerable else None,
        gold_verification_rank=1 if answerable else None,
        gold_cited=answerable,
        citation_count=1 if answerable else 0,
        answer_token_f1=1.0 if answerable else None,
        top_candidate_score=1.0,
        gold_candidate_score=1.0 if answerable else None,
        question_token_recall_in_gold_prompt=0.6 if answerable else None,
        answer_token_recall_in_gold_prompt=1.0 if answerable else None,
        answer_exact_span_visible=True if answerable else None,
        router_input_token_count=2000 + (history * 100),
        router_output_token_count=5,
        router_generation_latency_ms=100.0,
        end_to_end_latency_ms=120.0,
        retrieval_call_count=1,
        model_decision_count=1,
        composition_call_count=1,
        verification_call_count=1,
        diagnostic_observation_count=1,
        retry_action_count=0,
        fallback_action_count=0,
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
                    fold_id=index % 5,
                )
            )
    return tuple(rows)


def _merged_guard_report(observations):
    frozen = {"protocol_id": "test"}
    shard_reports = [
        {
            "decision": {"all_process_guards_passed": True},
            "current_source_fingerprints_before": {"same": True},
            "current_source_fingerprints_after": {"same": True},
        }
        for _ in range(12)
    ]
    report = {
        "user_confirmation": {
            "selected_option": "A_12_contiguous_process_shards",
            "confirmed": True,
        },
        "frozen_protocol": frozen,
        "frozen_protocol_sha256": sharding.canonical_json_sha256(frozen),
        "source_authorization": {"authorized": True},
        "train_diagnostic_protocol": {
            "source_sha256": protocol.STAGE165_EXPECTED_TRAIN_SHA256,
            "train_row_count": 562,
            "answerable_count": 370,
            "unanswerable_count": 192,
            "stable_order_sha256": protocol.STAGE165_EXPECTED_ORDER_SHA256,
            "dev_loaded": False,
            "test_loaded": False,
        },
        "workload_plan": {
            "unique_sample_count": 562,
            "agent_turn_count": 1124,
            "thread_count": 141,
            "grouping_sha256": protocol.STAGE165_EXPECTED_GROUPING_SHA256,
            "arm_schedule_sha256": protocol.STAGE165_EXPECTED_ARM_SCHEDULE_SHA256,
        },
        "grouped_fold_protocol": {
            "row_counts": {"0": 113, "1": 113, "2": 112, "3": 112, "4": 112},
            "fit_models": False,
            "select_policy": False,
            "tune_thresholds": False,
            "assignment_sha256": protocol.STAGE165_EXPECTED_FOLD_ASSIGNMENT_SHA256,
        },
        "sharding_plan": {
            "assignment_sha256": sharding.STAGE165_EXPECTED_SHARD_ASSIGNMENT_SHA256,
            "shard_count": 12,
            "full_shard_count": 11,
            "final_shard_thread_count": 9,
            "pair_count": 562,
            "agent_turn_count": 1124,
        },
        "process_results": [{"shard_ordinal": ordinal, "exit_code": 0} for ordinal in range(1, 13)],
        "runtime": {
            "single_runtime_instance": False,
            "process_count": 12,
            "resource_factory_build_count_total": 12,
            "warmup_generation_count_total": 12,
            "model_generation_call_count_total": 1136,
            "session": {
                "open_count": 703,
                "close_count": 703,
                "opened_thread_count_after_run": 0,
            },
        },
        "paired_diagnostics": protocol.summarize_stage165_pairs(observations),
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
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "cuda_empty_cache_called": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
        },
        "current_source_fingerprints_before": {"same": True},
        "current_source_fingerprints_after": {"same": True},
    }
    return report, shard_reports
