from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    CanonicalBoundedDynamicAgentServicePaths,
    PrimeQAHybridBoundedDynamicAgentServiceEntrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    StructuredRouterPromptPolicy,
)
from ts_rag_agent.config import ProjectSettings

from . import primeqa_hybrid_train_history_isolation_validation as stage165
from .primeqa_hybrid_train_history_isolation_protocol import (
    STAGE165_EXPECTED_ANSWERABLE_COUNT,
    STAGE165_EXPECTED_ARM_SCHEDULE_SHA256,
    STAGE165_EXPECTED_FOLD_ASSIGNMENT_SHA256,
    STAGE165_EXPECTED_GROUPING_SHA256,
    STAGE165_EXPECTED_ORDER_SHA256,
    STAGE165_EXPECTED_TRAIN_ROW_COUNT,
    STAGE165_EXPECTED_TRAIN_SHA256,
    STAGE165_EXPECTED_UNANSWERABLE_COUNT,
    Stage165ArmObservation,
    build_stage165_grouped_fold_assignment,
    build_stage165_paired_workload_plan,
    load_stage165_train_diagnostic_samples,
    stage165_private_report,
    summarize_stage165_pairs,
)
from .primeqa_hybrid_train_history_isolation_sharding_protocol import (
    STAGE165_EXPECTED_FINAL_SHARD_PAIR_COUNT,
    STAGE165_EXPECTED_FINAL_SHARD_THREAD_COUNT,
    STAGE165_EXPECTED_FULL_SHARD_COUNT,
    STAGE165_EXPECTED_SHARD_ASSIGNMENT_SHA256,
    STAGE165_EXPECTED_SHARD_COUNT,
    STAGE165_THREADS_PER_SHARD,
    Stage165ShardingPlan,
    Stage165ShardSpec,
    build_stage165_sharding_plan,
    canonical_json_sha256,
    file_sha256,
    load_stage165_observation_jsonl,
    stage165_observation_sequence_sha256,
    validate_stage165_shard_observations,
    write_stage165_observation_jsonl_row,
)

_STAGE = "Stage 165"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_train_history_isolation_sharded_diagnostics_v2"
_SHARD_ANALYSIS_ID = "primeqa_hybrid_train_history_isolation_single_shard_v1"
_PROTOCOL_ID = "primeqa_hybrid_stage165_12_process_sharded_history_isolation_v2"
_EXPECTED_PAIR_COUNT = 562
_EXPECTED_ARM_ROW_COUNT = 1124
_EXPECTED_SYNTHETIC_THREAD_COUNT = 141
_EXPECTED_FIRST_TURN_COUNT = 141
_EXPECTED_TOTAL_GENERATION_COUNT = _EXPECTED_ARM_ROW_COUNT + STAGE165_EXPECTED_SHARD_COUNT
_EXPECTED_SESSION_OPEN_COUNT = _EXPECTED_PAIR_COUNT + _EXPECTED_SYNTHETIC_THREAD_COUNT


@dataclass(frozen=True)
class Stage165ShardExecutionRun:
    public_report: dict[str, Any]
    observations: tuple[Stage165ArmObservation, ...]


@dataclass(frozen=True)
class Stage165ShardProcessResult:
    shard_ordinal: int
    exit_code: int
    duration_seconds: float
    stdout_path: str
    stderr_path: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "shard_ordinal": self.shard_ordinal,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 6),
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }


@dataclass(frozen=True)
class Stage165ShardedValidationRun:
    public_report: dict[str, Any]
    private_report: dict[str, Any] | None


@dataclass(frozen=True)
class Stage165ShardedArtifactPaths:
    root: Path

    def shard_public(self, ordinal: int) -> Path:
        return self.root / f"shard_{ordinal:02d}.public.json"

    def shard_observations(self, ordinal: int) -> Path:
        return self.root / f"shard_{ordinal:02d}.observations.jsonl"

    def shard_stdout(self, ordinal: int) -> Path:
        return self.root / f"shard_{ordinal:02d}.stdout.log"

    def shard_stderr(self, ordinal: int) -> Path:
        return self.root / f"shard_{ordinal:02d}.stderr.log"


@dataclass(frozen=True)
class Stage165ShardOrchestrationOutcome:
    process_results: tuple[Stage165ShardProcessResult, ...]
    shard_reports: tuple[dict[str, Any], ...]
    observations: tuple[Stage165ArmObservation, ...]
    failure: dict[str, Any] | None


class Stage165ShardProcessRunner(Protocol):
    def run(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        shard_ordinal: int,
        stdout_path: Path,
        stderr_path: Path,
    ) -> Stage165ShardProcessResult: ...


class SequentialSubprocessStage165ShardRunner:
    """Run exactly one shard process at a time without timeout or recovery."""

    def run(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        shard_ordinal: int,
        stdout_path: Path,
        stderr_path: Path,
    ) -> Stage165ShardProcessResult:
        started_at = time.perf_counter()
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
            )
        return Stage165ShardProcessResult(
            shard_ordinal=shard_ordinal,
            exit_code=completed.returncode,
            duration_seconds=time.perf_counter() - started_at,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


class Stage165ShardProcessOrchestrator:
    """Stop at the first failed or invalid shard and never retry it."""

    def __init__(self, *, runner: Stage165ShardProcessRunner) -> None:
        self._runner = runner

    def execute(
        self,
        *,
        shards: Sequence[Stage165ShardSpec],
        artifact_paths: Stage165ShardedArtifactPaths,
        cwd: Path,
        command_factory: Callable[[Stage165ShardSpec], Sequence[str]],
        result_loader: Callable[
            [Path, Path, Stage165ShardSpec],
            tuple[dict[str, Any], tuple[Stage165ArmObservation, ...]],
        ],
        total_agent_turn_count: int,
        progress_sink: stage165.ProgressSink | None = None,
    ) -> Stage165ShardOrchestrationOutcome:
        process_results: list[Stage165ShardProcessResult] = []
        shard_reports: list[dict[str, Any]] = []
        observations: list[Stage165ArmObservation] = []
        failure: dict[str, Any] | None = None
        for shard in shards:
            public_path = artifact_paths.shard_public(shard.ordinal)
            observation_path = artifact_paths.shard_observations(shard.ordinal)
            result = self._runner.run(
                command=command_factory(shard),
                cwd=cwd,
                shard_ordinal=shard.ordinal,
                stdout_path=artifact_paths.shard_stdout(shard.ordinal),
                stderr_path=artifact_paths.shard_stderr(shard.ordinal),
            )
            process_results.append(result)
            if result.exit_code != 0:
                failure = _failed_shard_summary(
                    shard=shard,
                    result=result,
                    observation_path=observation_path,
                    reason="nonzero_exit",
                )
                break
            try:
                shard_report, shard_observations = result_loader(
                    public_path,
                    observation_path,
                    shard,
                )
            except Exception as error:
                failure = _failed_shard_summary(
                    shard=shard,
                    result=result,
                    observation_path=observation_path,
                    reason=f"artifact_validation:{type(error).__name__}",
                )
                break
            shard_reports.append(shard_report)
            observations.extend(shard_observations)
            _emit(
                progress_sink,
                phase="shard_process_completed",
                shard_ordinal=shard.ordinal,
                completed_shard_count=len(shard_reports),
                total_shard_count=len(shards),
                completed_agent_turn_count=len(observations),
                total_agent_turn_count=total_agent_turn_count,
            )
        return Stage165ShardOrchestrationOutcome(
            process_results=tuple(process_results),
            shard_reports=tuple(shard_reports),
            observations=tuple(observations),
            failure=failure,
        )


def execute_stage165_shard(
    *,
    settings: ProjectSettings,
    stage164_correction_path: Path,
    train_split_path: Path,
    shard_ordinal: int,
    observation_jsonl_path: Path,
    user_confirmed_shard_execution: bool,
    progress_sink: stage165.ProgressSink | None = None,
) -> Stage165ShardExecutionRun:
    """Execute one exact process-isolated shard and persist every completed turn."""

    import torch

    if not user_confirmed_shard_execution:
        raise ValueError("Stage165 shard execution requires explicit confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage165 shard execution requires CUDA")
    started_at = time.perf_counter()
    project_root = Path(__file__).resolve().parents[3]
    paths = CanonicalBoundedDynamicAgentServicePaths.from_settings(settings)
    source_authorization = stage165._authorize_sources(
        project_root=project_root,
        stage164_correction_path=stage164_correction_path,
        train_split_path=train_split_path,
        paths=paths,
    )
    diagnostic_set = load_stage165_train_diagnostic_samples(train_split_path)
    workload = build_stage165_paired_workload_plan(diagnostic_set)
    folds = build_stage165_grouped_fold_assignment(diagnostic_set.samples)
    sharding = build_stage165_sharding_plan(workload)
    _validate_sharding_plan(sharding)
    shard = sharding.shard(shard_ordinal)
    observation_path = observation_jsonl_path.expanduser().resolve()
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    if observation_path.exists():
        raise FileExistsError("Stage165 shard observation artifact already exists")
    observation_path.write_text("", encoding="utf-8")
    current_source_before = _current_source_fingerprints(project_root)
    authorized_at = time.perf_counter()
    _emit(
        progress_sink,
        phase="shard_sources_and_protocol_authorized",
        shard_ordinal=shard.ordinal,
    )

    torch.cuda.reset_peak_memory_stats()
    prepared = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=settings,
        paths=paths,
    ).prepare()
    prepared_at = time.perf_counter()
    _emit(
        progress_sink,
        phase="shard_runtime_prepared",
        shard_ordinal=shard.ordinal,
    )
    session = stage165.Stage165BoundedAgentSession(
        runtime=prepared.runtime,
        prompt_policy=StructuredRouterPromptPolicy(),
    )
    observations = stage165.Stage165PairedWorkloadExecutor(session=session).execute(
        workload=workload,
        folds=folds,
        threads=shard.threads,
        observation_sink=lambda observation: write_stage165_observation_jsonl_row(
            path=observation_path,
            observation=observation,
        ),
        progress_sink=progress_sink,
    )
    executed_at = time.perf_counter()
    validate_stage165_shard_observations(
        shard=shard,
        workload=workload,
        observations=observations,
    )
    persisted = load_stage165_observation_jsonl(observation_path)
    current_source_after = _current_source_fingerprints(project_root)
    private_scan = stage165._public_safe_contract(
        {"rows": [observation.to_private_dict() for observation in persisted]}
    )
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _SHARD_ANALYSIS_ID,
        "analysis_scope": (
            "One train-only process-isolated shard of the user-confirmed Stage165 "
            "12-shard protocol. Every completed Agent turn is flushed once. No retry, "
            "fallback, cache clear, development access, or test access is allowed."
        ),
        "user_confirmation": {
            "selected_option": "A_12_contiguous_process_shards",
            "shard_execution_confirmed": True,
        },
        "source_authorization": source_authorization,
        "train_diagnostic_protocol": diagnostic_set.public_summary(),
        "workload_identity": {
            "grouping_sha256": workload.grouping_sha256,
            "arm_schedule_sha256": workload.arm_schedule_sha256,
            "fold_assignment_sha256": folds.assignment_sha256,
        },
        "sharding_plan": sharding.public_summary(),
        "shard": shard.public_summary(),
        "execution": {
            "observation_count": len(observations),
            "persisted_observation_count": len(persisted),
            "observation_sequence_sha256": stage165_observation_sequence_sha256(observations),
            "persisted_sequence_sha256": stage165_observation_sequence_sha256(persisted),
            "observation_jsonl_path": str(observation_path),
            "observation_jsonl_sha256": file_sha256(observation_path),
            "resource_factory_build_count": prepared.resource_factory_build_count,
            "model_generation_call_count": prepared.backend.generation_call_count,
            "warmup_generation_count": 1,
            "session_open_count": session.open_count,
            "session_close_count": session.close_count,
            "session_opened_thread_count_after_run": session.opened_thread_count,
            "gpu": {
                "device_name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
        },
        "private_artifact_contract": {
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "contains_raw_model_output": False,
            "forbidden_keys_found": private_scan["forbidden_keys_found"],
            "git_policy": "ignored_local_artifact",
        },
        "execution_boundaries": _closed_boundaries(),
        "timing_seconds": {
            "authorization_and_plan": round(authorized_at - started_at, 6),
            "runtime_prepare": round(prepared_at - authorized_at, 6),
            "agent_execution": round(executed_at - prepared_at, 6),
            "total": round(executed_at - started_at, 6),
        },
        "current_source_fingerprints_before": current_source_before,
        "current_source_fingerprints_after": current_source_after,
    }
    report["guard_checks"] = _shard_guard_checks(
        report,
        shard=shard,
        observations=observations,
        persisted=persisted,
    )
    report["public_safe_contract"] = stage165._public_safe_contract(report)
    report["decision"] = {
        "status": (
            "primeqa_hybrid_stage165_shard_completed"
            if all(check["passed"] for check in report["guard_checks"])
            else "primeqa_hybrid_stage165_shard_invalid"
        ),
        "all_process_guards_passed": all(check["passed"] for check in report["guard_checks"]),
        "failed_process_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "retry_authorized": False,
        "development_gate_opened": False,
        "test_gate_opened": False,
        "policy_selected": False,
    }
    return Stage165ShardExecutionRun(
        public_report=report,
        observations=observations,
    )


def validate_primeqa_hybrid_train_history_isolation_sharded(
    *,
    settings: ProjectSettings,
    stage164_correction_path: Path,
    train_split_path: Path,
    shard_artifact_dir: Path,
    user_confirmed_12_process_sharding: bool,
    process_runner: Stage165ShardProcessRunner | None = None,
    python_executable: Path | None = None,
    progress_sink: stage165.ProgressSink | None = None,
) -> Stage165ShardedValidationRun:
    """Run all 12 shards sequentially and merge only an exact complete execution."""

    if not user_confirmed_12_process_sharding:
        raise ValueError("Stage165 requires explicit 12-process sharding confirmation")
    started_at = time.perf_counter()
    project_root = Path(__file__).resolve().parents[3]
    paths = CanonicalBoundedDynamicAgentServicePaths.from_settings(settings)
    source_authorization = stage165._authorize_sources(
        project_root=project_root,
        stage164_correction_path=stage164_correction_path,
        train_split_path=train_split_path,
        paths=paths,
    )
    diagnostic_set = load_stage165_train_diagnostic_samples(train_split_path)
    workload = build_stage165_paired_workload_plan(diagnostic_set)
    folds = build_stage165_grouped_fold_assignment(diagnostic_set.samples)
    sharding = build_stage165_sharding_plan(workload)
    _validate_sharding_plan(sharding)
    protocol = _frozen_sharded_protocol()
    protocol_sha256 = canonical_json_sha256(protocol)
    current_source_before = _current_source_fingerprints(project_root)
    artifact_paths = Stage165ShardedArtifactPaths(root=shard_artifact_dir.expanduser().resolve())
    _prepare_new_artifact_directory(artifact_paths.root)
    runner = process_runner or SequentialSubprocessStage165ShardRunner()
    executable = (python_executable or Path(sys.executable)).expanduser().resolve(strict=True)
    child_cli = (
        project_root / "scripts" / "run_primeqa_hybrid_train_history_isolation_shard.py"
    ).resolve(strict=True)

    def command_factory(shard: Stage165ShardSpec) -> tuple[str, ...]:
        return _shard_command(
            executable=executable,
            child_cli=child_cli,
            model_snapshot=paths.model_snapshot,
            stage164_correction_path=stage164_correction_path,
            train_split_path=train_split_path,
            shard=shard,
            public_path=artifact_paths.shard_public(shard.ordinal),
            observation_path=artifact_paths.shard_observations(shard.ordinal),
        )

    def result_loader(
        public_path: Path,
        observation_path: Path,
        shard: Stage165ShardSpec,
    ) -> tuple[dict[str, Any], tuple[Stage165ArmObservation, ...]]:
        return _load_completed_shard(
            public_path=public_path,
            observation_path=observation_path,
            shard=shard,
            workload=workload,
        )

    outcome = Stage165ShardProcessOrchestrator(runner=runner).execute(
        shards=sharding.shards,
        artifact_paths=artifact_paths,
        cwd=project_root,
        command_factory=command_factory,
        result_loader=result_loader,
        total_agent_turn_count=sharding.arm_row_count,
        progress_sink=progress_sink,
    )

    current_source_after = _current_source_fingerprints(project_root)
    if outcome.failure is not None:
        return Stage165ShardedValidationRun(
            public_report=_failed_orchestration_report(
                source_authorization=source_authorization,
                diagnostic_set_summary=diagnostic_set.public_summary(),
                workload_summary=_sharded_workload_summary(workload.public_summary()),
                fold_summary=folds.public_summary(),
                sharding=sharding,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                process_results=outcome.process_results,
                completed_shard_reports=outcome.shard_reports,
                completed_observation_count=len(outcome.observations),
                failure=outcome.failure,
                current_source_before=current_source_before,
                current_source_after=current_source_after,
                elapsed_seconds=time.perf_counter() - started_at,
            ),
            private_report=None,
        )

    observations = outcome.observations
    _validate_merged_observations(
        sharding=sharding,
        workload=workload,
        observations=observations,
    )
    private_report = stage165_private_report(observations)
    private_sha256 = canonical_json_sha256(private_report)
    diagnostics = summarize_stage165_pairs(observations)
    report = _completed_orchestration_report(
        source_authorization=source_authorization,
        diagnostic_set_summary=diagnostic_set.public_summary(),
        workload_summary=_sharded_workload_summary(workload.public_summary()),
        fold_summary=folds.public_summary(),
        sharding=sharding,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        process_results=outcome.process_results,
        shard_reports=outcome.shard_reports,
        observations=observations,
        diagnostics=diagnostics,
        private_report=private_report,
        private_sha256=private_sha256,
        current_source_before=current_source_before,
        current_source_after=current_source_after,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    return Stage165ShardedValidationRun(
        public_report=report,
        private_report=private_report,
    )


def _completed_orchestration_report(
    *,
    source_authorization: Mapping[str, Any],
    diagnostic_set_summary: Mapping[str, Any],
    workload_summary: Mapping[str, Any],
    fold_summary: Mapping[str, Any],
    sharding: Stage165ShardingPlan,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    process_results: Sequence[Stage165ShardProcessResult],
    shard_reports: Sequence[Mapping[str, Any]],
    observations: Sequence[Stage165ArmObservation],
    diagnostics: Mapping[str, Any],
    private_report: Mapping[str, Any],
    private_sha256: str,
    current_source_before: Mapping[str, Any],
    current_source_after: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    session_open_count = sum(
        int(report["execution"]["session_open_count"]) for report in shard_reports
    )
    session_close_count = sum(
        int(report["execution"]["session_close_count"]) for report in shard_reports
    )
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Complete train-only paired history-isolation diagnostics executed as 12 "
            "strictly sequential fresh GPU processes on contiguous synthetic-thread "
            "boundaries. Shard outputs are merged only after exact coverage checks."
        ),
        "user_confirmation": {
            "selected_option": "A_12_contiguous_process_shards",
            "confirmed": True,
            "threads_per_full_shard": STAGE165_THREADS_PER_SHARD,
            "shard_count": STAGE165_EXPECTED_SHARD_COUNT,
        },
        "source_authorization": dict(source_authorization),
        "frozen_protocol": dict(protocol),
        "frozen_protocol_sha256": protocol_sha256,
        "train_diagnostic_protocol": dict(diagnostic_set_summary),
        "workload_plan": dict(workload_summary),
        "grouped_fold_protocol": dict(fold_summary),
        "sharding_plan": sharding.public_summary(),
        "runtime": {
            "execution_mode": "strictly_sequential_fresh_process_per_shard",
            "single_runtime_instance": False,
            "process_count": len(process_results),
            "resource_factory_build_count_total": sum(
                int(report["execution"]["resource_factory_build_count"]) for report in shard_reports
            ),
            "model_generation_call_count_total": sum(
                int(report["execution"]["model_generation_call_count"]) for report in shard_reports
            ),
            "warmup_generation_count_total": sum(
                int(report["execution"]["warmup_generation_count"]) for report in shard_reports
            ),
            "session": {
                "open_count": session_open_count,
                "close_count": session_close_count,
                "opened_thread_count_after_run": sum(
                    int(report["execution"]["session_opened_thread_count_after_run"])
                    for report in shard_reports
                ),
            },
            "shards": [
                {
                    "shard_ordinal": report["shard"]["shard_ordinal"],
                    "agent_turn_count": report["execution"]["observation_count"],
                    "model_generation_call_count": report["execution"][
                        "model_generation_call_count"
                    ],
                    "peak_allocated_bytes": report["execution"]["gpu"]["peak_allocated_bytes"],
                    "peak_reserved_bytes": report["execution"]["gpu"]["peak_reserved_bytes"],
                    "timing_seconds": report["timing_seconds"],
                }
                for report in shard_reports
            ],
        },
        "process_results": [result.to_public_dict() for result in process_results],
        "paired_diagnostics": dict(diagnostics),
        "private_diagnostic_artifact_contract": {
            "canonical_content_sha256": private_sha256,
            "arm_row_count": len(observations),
            "pair_count": len(observations) // 2,
            "contains_hashed_sample_identity": True,
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "contains_raw_model_output": False,
            "public_report_contains_case_rows": False,
            "git_policy": "ignored_local_artifact",
        },
        "execution_boundaries": _closed_boundaries(),
        "timing_seconds": {
            "total": round(elapsed_seconds, 6),
            "sum_shard_process_duration": round(
                sum(result.duration_seconds for result in process_results),
                6,
            ),
        },
        "current_source_fingerprints_before": dict(current_source_before),
        "current_source_fingerprints_after": dict(current_source_after),
    }
    report["guard_checks"] = _merged_guard_checks(
        report,
        observations=observations,
        shard_reports=shard_reports,
    )
    report["public_safe_contract"] = stage165._public_safe_contract(report)
    report["decision"] = stage165._decision(report)
    return report


def _failed_orchestration_report(
    *,
    source_authorization: Mapping[str, Any],
    diagnostic_set_summary: Mapping[str, Any],
    workload_summary: Mapping[str, Any],
    fold_summary: Mapping[str, Any],
    sharding: Stage165ShardingPlan,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    process_results: Sequence[Stage165ShardProcessResult],
    completed_shard_reports: Sequence[Mapping[str, Any]],
    completed_observation_count: int,
    failure: Mapping[str, Any],
    current_source_before: Mapping[str, Any],
    current_source_after: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": "Incomplete train-only Stage165 sharded execution.",
        "user_confirmation": {
            "selected_option": "A_12_contiguous_process_shards",
            "confirmed": True,
        },
        "source_authorization": dict(source_authorization),
        "frozen_protocol": dict(protocol),
        "frozen_protocol_sha256": protocol_sha256,
        "train_diagnostic_protocol": dict(diagnostic_set_summary),
        "workload_plan": dict(workload_summary),
        "grouped_fold_protocol": dict(fold_summary),
        "sharding_plan": sharding.public_summary(),
        "execution": {
            "completed_shard_count": len(completed_shard_reports),
            "attempted_shard_count": len(process_results),
            "completed_observation_count": completed_observation_count,
            "failure": dict(failure),
            "continued_after_failure": False,
            "retry_count": 0,
        },
        "process_results": [result.to_public_dict() for result in process_results],
        "execution_boundaries": _closed_boundaries(),
        "timing_seconds": {"total": round(elapsed_seconds, 6)},
        "current_source_fingerprints_before": dict(current_source_before),
        "current_source_fingerprints_after": dict(current_source_after),
    }
    report["public_safe_contract"] = stage165._public_safe_contract(report)
    report["decision"] = {
        "status": "primeqa_hybrid_stage165_sharded_execution_failed",
        "complete_quality_metrics_available": False,
        "all_process_guards_passed": False,
        "retry_authorized": False,
        "development_gate_opened": False,
        "test_gate_opened": False,
        "policy_selected": False,
        "runtime_registered_as_default": False,
        "next_direction": "inspect_failed_shard_without_automatic_retry",
    }
    return report


def _shard_guard_checks(
    report: Mapping[str, Any],
    *,
    shard: Stage165ShardSpec,
    observations: Sequence[Stage165ArmObservation],
    persisted: Sequence[Stage165ArmObservation],
) -> list[dict[str, Any]]:
    execution = report["execution"]
    boundaries = report["execution_boundaries"]
    expected_open_count = shard.pair_count + len(shard.threads)
    return [
        _check(
            "user_confirmed_exact_12_shard_option_a",
            report["user_confirmation"].get("selected_option") == "A_12_contiguous_process_shards"
            and report["user_confirmation"].get("shard_execution_confirmed") is True,
        ),
        _check(
            "upstream_sources_exact",
            report["source_authorization"].get("authorized") is True,
        ),
        _check(
            "sharding_assignment_exact",
            report["sharding_plan"].get("assignment_sha256")
            == STAGE165_EXPECTED_SHARD_ASSIGNMENT_SHA256,
        ),
        _check(
            "shard_shape_exact",
            report["shard"] == shard.public_summary(),
        ),
        _check(
            "observation_sequence_and_persistence_exact",
            len(observations) == shard.arm_row_count
            and tuple(observations) == tuple(persisted)
            and execution.get("observation_sequence_sha256")
            == execution.get("persisted_sequence_sha256"),
        ),
        _check(
            "one_resource_model_and_warmup_per_shard",
            execution.get("resource_factory_build_count") == 1
            and execution.get("warmup_generation_count") == 1
            and execution.get("model_generation_call_count") == shard.arm_row_count + 1,
        ),
        _check(
            "thread_lifecycle_closed_exact",
            execution.get("session_open_count") == expected_open_count
            and execution.get("session_close_count") == expected_open_count
            and execution.get("session_opened_thread_count_after_run") == 0,
        ),
        _check(
            "current_sources_unchanged_during_shard",
            report.get("current_source_fingerprints_before")
            == report.get("current_source_fingerprints_after"),
        ),
        _check(
            "private_incremental_artifact_content_free",
            report["private_artifact_contract"].get("forbidden_keys_found") == []
            and all(
                report["private_artifact_contract"].get(key) is False
                for key in (
                    "contains_raw_question",
                    "contains_raw_answer",
                    "contains_raw_document_id",
                    "contains_raw_document_text",
                    "contains_raw_model_output",
                )
            ),
        ),
        _check(
            "shard_public_report_content_free",
            stage165._public_safe_contract(report).get("public_safe") is True,
        ),
        _check(
            "train_only_dev_test_closed",
            boundaries.get("development_loaded") is False
            and boundaries.get("test_loaded") is False,
        ),
        _check(
            "no_retry_fallback_cache_clear_or_policy_change",
            boundaries.get("retry_actions_enabled") is False
            and boundaries.get("fallback_strategies_enabled") is False
            and boundaries.get("cuda_empty_cache_called") is False
            and boundaries.get("policy_selected") is False
            and boundaries.get("runtime_registered_as_default") is False,
        ),
    ]


def _merged_guard_checks(
    report: Mapping[str, Any],
    *,
    observations: Sequence[Stage165ArmObservation],
    shard_reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source = report["source_authorization"]
    train = report["train_diagnostic_protocol"]
    workload = report["workload_plan"]
    folds = report["grouped_fold_protocol"]
    sharding = report["sharding_plan"]
    runtime = report["runtime"]
    diagnostics = report["paired_diagnostics"]
    overview = diagnostics["overview"]
    first_turn = diagnostics["first_turn_negative_control"]
    boundaries = report["execution_boundaries"]
    arm_counts = Counter(observation.arm for observation in observations)
    histories_exact = all(
        observation.history_turn_count_before
        == (0 if observation.arm == "isolated" else observation.synthetic_turn_position - 1)
        for observation in observations
    )
    trace_counts_exact = all(
        observation.retrieval_call_count == 1
        and observation.model_decision_count == 1
        and observation.retry_action_count == 0
        and observation.fallback_action_count == 0
        for observation in observations
    )
    return [
        _check(
            "user_confirmed_option_a_12_process_shards",
            report["user_confirmation"].get("confirmed") is True
            and report["user_confirmation"].get("selected_option")
            == "A_12_contiguous_process_shards",
        ),
        _check(
            "frozen_protocol_identity_exact",
            report.get("frozen_protocol_sha256")
            == canonical_json_sha256(report.get("frozen_protocol")),
        ),
        _check("upstream_sources_exact", source.get("authorized") is True),
        _check(
            "train_source_and_counts_exact",
            train.get("source_sha256") == STAGE165_EXPECTED_TRAIN_SHA256
            and train.get("train_row_count") == STAGE165_EXPECTED_TRAIN_ROW_COUNT
            and train.get("answerable_count") == STAGE165_EXPECTED_ANSWERABLE_COUNT
            and train.get("unanswerable_count") == STAGE165_EXPECTED_UNANSWERABLE_COUNT
            and train.get("stable_order_sha256") == STAGE165_EXPECTED_ORDER_SHA256,
        ),
        _check(
            "train_only_split_boundary_exact",
            train.get("dev_loaded") is False and train.get("test_loaded") is False,
        ),
        _check(
            "paired_workload_shape_exact",
            workload.get("unique_sample_count") == _EXPECTED_PAIR_COUNT
            and workload.get("agent_turn_count") == _EXPECTED_ARM_ROW_COUNT
            and workload.get("thread_count") == _EXPECTED_SYNTHETIC_THREAD_COUNT
            and workload.get("grouping_sha256") == STAGE165_EXPECTED_GROUPING_SHA256
            and workload.get("arm_schedule_sha256") == STAGE165_EXPECTED_ARM_SCHEDULE_SHA256,
        ),
        _check(
            "grouped_five_fold_isolation",
            folds.get("assignment_sha256") == STAGE165_EXPECTED_FOLD_ASSIGNMENT_SHA256
            and sum(folds.get("row_counts", {}).values()) == _EXPECTED_PAIR_COUNT
            and folds.get("fit_models") is False
            and folds.get("select_policy") is False
            and folds.get("tune_thresholds") is False,
        ),
        _check(
            "sharding_plan_exact",
            sharding.get("assignment_sha256") == STAGE165_EXPECTED_SHARD_ASSIGNMENT_SHA256
            and sharding.get("shard_count") == STAGE165_EXPECTED_SHARD_COUNT
            and sharding.get("full_shard_count") == STAGE165_EXPECTED_FULL_SHARD_COUNT
            and sharding.get("final_shard_thread_count")
            == STAGE165_EXPECTED_FINAL_SHARD_THREAD_COUNT
            and sharding.get("pair_count") == _EXPECTED_PAIR_COUNT
            and sharding.get("agent_turn_count") == _EXPECTED_ARM_ROW_COUNT,
        ),
        _check(
            "all_shard_processes_completed_once",
            len(report["process_results"]) == STAGE165_EXPECTED_SHARD_COUNT
            and [item["shard_ordinal"] for item in report["process_results"]]
            == list(range(1, STAGE165_EXPECTED_SHARD_COUNT + 1))
            and all(item["exit_code"] == 0 for item in report["process_results"]),
        ),
        _check(
            "all_shard_guards_passed",
            len(shard_reports) == STAGE165_EXPECTED_SHARD_COUNT
            and all(
                shard_report["decision"].get("all_process_guards_passed") is True
                for shard_report in shard_reports
            ),
        ),
        _check(
            "arm_observations_and_pairs_exact",
            len(observations) == _EXPECTED_ARM_ROW_COUNT
            and arm_counts
            == {
                "isolated": _EXPECTED_PAIR_COUNT,
                "synthetic_history": _EXPECTED_PAIR_COUNT,
            }
            and overview.get("pair_count") == _EXPECTED_PAIR_COUNT,
        ),
        _check("history_state_assignment_exact", histories_exact),
        _check(
            "paired_retrieval_contexts_exact",
            overview.get("context_signature_exact_count") == _EXPECTED_PAIR_COUNT,
        ),
        _check(
            "first_turn_negative_control_exact",
            first_turn.get("pair_count") == _EXPECTED_FIRST_TURN_COUNT
            and first_turn.get("context_signature_exact_count") == _EXPECTED_FIRST_TURN_COUNT
            and first_turn.get("output_exact_count") == _EXPECTED_FIRST_TURN_COUNT
            and first_turn.get("refusal_disagreement_count") == 0
            and first_turn.get("average_input_token_difference_synthetic_minus_isolated") == 0.0,
        ),
        _check("one_retrieval_and_model_decision_per_turn", trace_counts_exact),
        _check(
            "fresh_resource_model_and_warmup_per_shard",
            runtime.get("single_runtime_instance") is False
            and runtime.get("process_count") == STAGE165_EXPECTED_SHARD_COUNT
            and runtime.get("resource_factory_build_count_total") == STAGE165_EXPECTED_SHARD_COUNT
            and runtime.get("warmup_generation_count_total") == STAGE165_EXPECTED_SHARD_COUNT
            and runtime.get("model_generation_call_count_total")
            == _EXPECTED_TOTAL_GENERATION_COUNT,
        ),
        _check(
            "thread_lifecycle_closed_exact",
            runtime["session"].get("open_count") == _EXPECTED_SESSION_OPEN_COUNT
            and runtime["session"].get("close_count") == _EXPECTED_SESSION_OPEN_COUNT
            and runtime["session"].get("opened_thread_count_after_run") == 0,
        ),
        _check(
            "current_sources_unchanged_during_run",
            report.get("current_source_fingerprints_before")
            == report.get("current_source_fingerprints_after")
            and all(
                shard_report.get("current_source_fingerprints_before")
                == shard_report.get("current_source_fingerprints_after")
                for shard_report in shard_reports
            ),
        ),
        _check(
            "private_artifact_content_free",
            report["private_diagnostic_artifact_contract"].get("arm_row_count")
            == _EXPECTED_ARM_ROW_COUNT
            and all(
                report["private_diagnostic_artifact_contract"].get(key) is False
                for key in (
                    "contains_raw_question",
                    "contains_raw_answer",
                    "contains_raw_document_id",
                    "contains_raw_document_text",
                    "contains_raw_model_output",
                    "public_report_contains_case_rows",
                )
            ),
        ),
        _check(
            "merged_public_report_content_free",
            stage165._public_safe_contract(report).get("public_safe") is True,
        ),
        _check(
            "no_fit_tuning_or_policy_selection",
            boundaries.get("model_fit") is False
            and boundaries.get("threshold_tuned") is False
            and boundaries.get("policy_selected") is False,
        ),
        _check(
            "development_test_and_runtime_default_closed",
            boundaries.get("development_loaded") is False
            and boundaries.get("test_loaded") is False
            and boundaries.get("runtime_registered_as_default") is False,
        ),
        _check(
            "queue_retry_fallback_cache_clear_rewrite_second_retrieval_closed",
            boundaries.get("queue_actions_enabled") is False
            and boundaries.get("retry_actions_enabled") is False
            and boundaries.get("fallback_strategies_enabled") is False
            and boundaries.get("cuda_empty_cache_called") is False
            and boundaries.get("query_rewrite_enabled") is False
            and boundaries.get("second_retrieval_enabled") is False,
        ),
    ]


def _load_completed_shard(
    *,
    public_path: Path,
    observation_path: Path,
    shard: Stage165ShardSpec,
    workload,
) -> tuple[dict[str, Any], tuple[Stage165ArmObservation, ...]]:
    report = json.loads(public_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Stage165 shard public artifact must be an object")
    observations = load_stage165_observation_jsonl(observation_path)
    validate_stage165_shard_observations(
        shard=shard,
        workload=workload,
        observations=observations,
    )
    execution = report.get("execution") or {}
    if report.get("shard") != shard.public_summary():
        raise ValueError("Stage165 shard public identity is not exact")
    if report.get("decision", {}).get("all_process_guards_passed") is not True:
        raise ValueError("Stage165 shard process guards did not pass")
    if execution.get("observation_jsonl_sha256") != file_sha256(observation_path):
        raise ValueError("Stage165 shard JSONL byte hash is not exact")
    if execution.get("persisted_sequence_sha256") != stage165_observation_sequence_sha256(
        observations
    ):
        raise ValueError("Stage165 shard JSONL sequence hash is not exact")
    return report, observations


def _validate_merged_observations(
    *,
    sharding: Stage165ShardingPlan,
    workload,
    observations: Sequence[Stage165ArmObservation],
) -> None:
    offset = 0
    for shard in sharding.shards:
        shard_rows = observations[offset : offset + shard.arm_row_count]
        validate_stage165_shard_observations(
            shard=shard,
            workload=workload,
            observations=shard_rows,
        )
        offset += shard.arm_row_count
    if offset != len(observations) or len(observations) != _EXPECTED_ARM_ROW_COUNT:
        raise ValueError("Stage165 merged observation coverage is not exact")


def _validate_sharding_plan(sharding: Stage165ShardingPlan) -> None:
    summary = sharding.public_summary()
    if not (
        summary["assignment_sha256"] == STAGE165_EXPECTED_SHARD_ASSIGNMENT_SHA256
        and summary["shard_count"] == STAGE165_EXPECTED_SHARD_COUNT
        and summary["full_shard_count"] == STAGE165_EXPECTED_FULL_SHARD_COUNT
        and summary["final_shard_thread_count"] == STAGE165_EXPECTED_FINAL_SHARD_THREAD_COUNT
        and sharding.shards[-1].pair_count == STAGE165_EXPECTED_FINAL_SHARD_PAIR_COUNT
        and summary["synthetic_thread_count"] == _EXPECTED_SYNTHETIC_THREAD_COUNT
        and summary["pair_count"] == _EXPECTED_PAIR_COUNT
        and summary["agent_turn_count"] == _EXPECTED_ARM_ROW_COUNT
    ):
        raise ValueError("Stage165 sharding plan is not the frozen option A protocol")


def _frozen_sharded_protocol() -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "user_selected_option": "A_12_contiguous_process_shards",
        "threads_per_full_shard": STAGE165_THREADS_PER_SHARD,
        "shard_count": STAGE165_EXPECTED_SHARD_COUNT,
        "assignment_sha256": STAGE165_EXPECTED_SHARD_ASSIGNMENT_SHA256,
        "execution_order": "strictly_sequential",
        "fresh_process_per_shard": True,
        "merge_requires_all_shards_exit_zero": True,
        "merge_requires_exact_artifact_hashes": True,
        "failure_stops_later_shards": True,
        "incremental_observation_flush": True,
        "timeout": False,
        "retry": False,
        "fallback": False,
        "cuda_empty_cache": False,
        "fit_models": False,
        "tune_thresholds": False,
        "select_runtime_policy": False,
        "development_loaded": False,
        "test_loaded": False,
        "runtime_registered_as_default": False,
    }


def _sharded_workload_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(summary),
        "runtime_instance_count": STAGE165_EXPECTED_SHARD_COUNT,
        "model_load_count": STAGE165_EXPECTED_SHARD_COUNT,
        "resource_build_count": STAGE165_EXPECTED_SHARD_COUNT,
    }


def _closed_boundaries() -> dict[str, Any]:
    return {
        "train_loaded": True,
        "development_loaded": False,
        "test_loaded": False,
        "gold_projected_into_runtime": False,
        "model_fit": False,
        "threshold_tuned": False,
        "policy_selected": False,
        "runtime_registered_as_default": False,
        "remote_exposure": False,
        "http_server_started": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "cuda_empty_cache_called": False,
        "query_rewrite_enabled": False,
        "second_retrieval_enabled": False,
    }


def _prepare_new_artifact_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Stage165 sharded artifact directory must be new or empty")
    path.mkdir(parents=True, exist_ok=True)


def _shard_command(
    *,
    executable: Path,
    child_cli: Path,
    model_snapshot: Path,
    stage164_correction_path: Path,
    train_split_path: Path,
    shard: Stage165ShardSpec,
    public_path: Path,
    observation_path: Path,
) -> tuple[str, ...]:
    return (
        str(executable),
        str(child_cli),
        "--model-snapshot",
        str(model_snapshot),
        "--stage164-correction",
        str(stage164_correction_path),
        "--train-split",
        str(train_split_path),
        "--shard-ordinal",
        str(shard.ordinal),
        "--output",
        str(public_path),
        "--observation-jsonl",
        str(observation_path),
        "--user-confirmed-stage165-shard",
    )


def _failed_shard_summary(
    *,
    shard: Stage165ShardSpec,
    result: Stage165ShardProcessResult,
    observation_path: Path,
    reason: str,
) -> dict[str, Any]:
    observation_count = 0
    observation_sha256 = None
    if observation_path.is_file():
        with observation_path.open("r", encoding="utf-8") as source:
            observation_count = sum(1 for line in source if line.strip())
        observation_sha256 = file_sha256(observation_path)
    return {
        "failed_shard_ordinal": shard.ordinal,
        "reason": reason,
        "exit_code": result.exit_code,
        "completed_incremental_observation_count": observation_count,
        "incremental_observation_jsonl_sha256": observation_sha256,
        "automatic_retry": False,
        "continued_to_later_shard": False,
    }


def _current_source_fingerprints(project_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "base_protocol": (
            project_root
            / "src"
            / "ts_rag_agent"
            / "application"
            / "primeqa_hybrid_train_history_isolation_protocol.py"
        ),
        "base_validation": (
            project_root
            / "src"
            / "ts_rag_agent"
            / "application"
            / "primeqa_hybrid_train_history_isolation_validation.py"
        ),
        "sharding_protocol": (
            project_root
            / "src"
            / "ts_rag_agent"
            / "application"
            / "primeqa_hybrid_train_history_isolation_sharding_protocol.py"
        ),
        "sharded_validation": Path(__file__).resolve(),
        "shard_cli": (
            project_root / "scripts" / "run_primeqa_hybrid_train_history_isolation_shard.py"
        ),
        "orchestrator_cli": (
            project_root / "scripts" / "analyze_primeqa_hybrid_train_history_isolation_sharded.py"
        ),
    }
    return {name: stage165._fingerprint(path) for name, path in paths.items()}


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: stage165.ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
