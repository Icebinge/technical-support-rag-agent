from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.probe_primeqa_hybrid_train_history_isolation_memory import app
from ts_rag_agent.application import (
    primeqa_hybrid_train_history_isolation_memory_probe as probe,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    GeneratedRouterText,
)


def test_observed_backend_flushes_success_event_incrementally(tmp_path: Path) -> None:
    delegate = _FakeDelegate()
    event_path = tmp_path / "events.jsonl"
    backend = probe.Stage165ObservedGenerationBackend(
        delegate=delegate,
        event_jsonl_path=event_path,
    )
    backend.set_probe_context(_context())

    generated = backend.generate(prompt="prompt", max_input_tokens=10, max_new_tokens=5)

    assert generated.text == "generated"
    assert backend.generation_call_count == 1
    event = backend.events[0]
    assert event.generation_completed is True
    assert event.input_token_count_preflight == 4
    assert event.cuda_peak_allocated == 300
    assert event.cuda_telemetry_failure_types == ()
    persisted = [json.loads(line) for line in event_path.read_text().splitlines()]
    serialized_event = json.loads(json.dumps(event.to_private_dict()))
    assert persisted == [serialized_event]


def test_observed_backend_records_cuda_oom_and_does_not_retry(tmp_path: Path) -> None:
    delegate = _FakeDelegate(failure=RuntimeError("CUDA out of memory"))
    backend = probe.Stage165ObservedGenerationBackend(
        delegate=delegate,
        event_jsonl_path=tmp_path / "events.jsonl",
    )
    backend.set_probe_context(_context())

    with pytest.raises(RuntimeError, match="out of memory"):
        backend.generate(prompt="prompt", max_input_tokens=10, max_new_tokens=5)

    assert delegate.attempt_count == 1
    event = backend.events[0]
    assert event.generation_completed is False
    assert event.failure_kind == "cuda_out_of_memory"
    assert event.failure_type == "RuntimeError"


def test_post_failure_cuda_telemetry_never_masks_original_failure(
    tmp_path: Path,
) -> None:
    delegate = _FakeDelegate(failure=RuntimeError("CUDA out of memory"))
    delegate._torch.cuda.peak_allocated_failure = RuntimeError("telemetry failed")
    backend = probe.Stage165ObservedGenerationBackend(
        delegate=delegate,
        event_jsonl_path=tmp_path / "events.jsonl",
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        backend.generate(prompt="prompt", max_input_tokens=10, max_new_tokens=5)

    event = backend.events[0]
    assert event.failure_kind == "cuda_out_of_memory"
    assert event.cuda_peak_allocated is None
    assert event.cuda_telemetry_failure_types == ("peak_allocated:RuntimeError",)


def test_memory_summary_preserves_unavailable_telemetry() -> None:
    event = _event(
        generation_completed=False,
        failure_kind="cuda_out_of_memory",
        cuda_peak_allocated=None,
        cuda_telemetry_failure_types=("peak_allocated:RuntimeError",),
    )

    summary = probe._memory_summary([event])

    assert summary["failed_count"] == 1
    assert summary["cuda_peak_allocated_bytes"]["count"] == 0
    assert summary["cuda_telemetry_failure_count"] == 1
    assert summary["failed_attempt"]["cuda_peak_allocated"] is None


def test_probe_guards_accept_exact_content_free_eight_turn_contract() -> None:
    report = _guard_report()

    checks = probe._guard_checks(report)

    assert len(checks) == 11
    assert all(check["passed"] for check in checks)


def test_probe_guards_reject_test_access_retry_and_raw_keys() -> None:
    report = _guard_report()
    report["probe_contract"]["test_loaded"] = True
    report["closed_boundaries"]["retry_actions"] = 1
    report["private_event_artifact_contract"]["forbidden_keys_found"] = ["question_text"]

    checks = {check["name"]: check["passed"] for check in probe._guard_checks(report)}

    assert checks["private_events_content_free"] is False
    assert checks["train_only_dev_test_closed"] is False
    assert checks["no_retry_fallback_cache_clear_rewrite_or_second_retrieval"] is False


def test_memory_probe_visualizations_write_three_parseable_svgs(
    tmp_path: Path,
) -> None:
    private_report = {
        "events": [
            _event().to_private_dict(),
            _event(
                arm="synthetic_history",
                synthetic_turn_position=2,
                cuda_peak_allocated=None,
                cuda_peak_reserved=None,
            ).to_private_dict(),
        ]
    }

    visuals = probe.write_stage165_memory_probe_visualizations(
        private_report=private_report,
        output_dir=tmp_path,
    )

    assert len(visuals) == 3
    for visual in visuals:
        ET.parse(visual.path)


def test_memory_probe_cli_requires_explicit_confirmation() -> None:
    result = CliRunner().invoke(app, ["--model-snapshot", "."])

    assert result.exit_code != 0
    assert "user-confirmed-thread37-probe" in result.output


class _Shape:
    def __getitem__(self, index: int) -> int:
        return (1, 4)[index]


class _InputIds:
    shape = _Shape()


class _FakeProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": _InputIds()}


class _FakeCuda:
    def __init__(self) -> None:
        self.peak_allocated_failure: Exception | None = None

    def synchronize(self) -> None:
        return None

    def reset_peak_memory_stats(self) -> None:
        return None

    def memory_allocated(self) -> int:
        return 100

    def memory_reserved(self) -> int:
        return 200

    def max_memory_allocated(self) -> int:
        if self.peak_allocated_failure is not None:
            raise self.peak_allocated_failure
        return 300

    def max_memory_reserved(self) -> int:
        return 400


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeDelegate:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self._processor = _FakeProcessor()
        self._torch = _FakeTorch()
        self.failure = failure
        self.attempt_count = 0
        self.generation_call_count = 0

    def generate(self, *, prompt: str, max_input_tokens: int, max_new_tokens: int):
        self.attempt_count += 1
        if self.failure is not None:
            raise self.failure
        self.generation_call_count += 1
        return GeneratedRouterText(
            text="generated",
            input_token_count=4,
            output_token_count=2,
            generation_latency_ms=3.0,
        )


def _context() -> probe.Stage165MemoryProbeContext:
    return probe.Stage165MemoryProbeContext(
        phase="thread37_probe",
        private_identity_sha256="a" * 64,
        synthetic_turn_position=1,
        arm="isolated",
        arm_order_position=1,
    )


def _event(**overrides) -> probe.Stage165MemoryProbeEvent:
    values = {
        "generation_attempt": 1,
        "phase": "thread37_probe",
        "private_identity_sha256": "a" * 64,
        "synthetic_turn_position": 1,
        "arm": "isolated",
        "arm_order_position": 1,
        "prompt_sha256": "b" * 64,
        "prompt_character_count": 100,
        "input_token_count_preflight": 25,
        "cuda_allocated_before": 100,
        "cuda_reserved_before": 200,
        "cuda_peak_allocated": 300,
        "cuda_peak_reserved": 400,
        "cuda_allocated_after": 100,
        "cuda_reserved_after": 200,
        "cuda_telemetry_failure_types": (),
        "generation_completed": True,
        "output_token_count": 2,
        "generation_latency_ms": 3.0,
        "failure_kind": None,
        "failure_type": None,
    }
    values.update(overrides)
    return probe.Stage165MemoryProbeEvent(**values)


def _guard_report() -> dict:
    return {
        "user_confirmation": {
            "selected_option": "A",
            "thread37_probe_confirmed": True,
        },
        "source_authorization": {"authorized": True},
        "probe_contract": {
            "train_only": True,
            "synthetic_thread_ordinal": 37,
            "sample_count": 4,
            "maximum_agent_turns": 8,
            "failure_stops_probe": True,
            "retry": False,
            "fallback": False,
            "cuda_empty_cache_called": False,
            "development_loaded": False,
            "test_loaded": False,
        },
        "execution": {
            "attempted_train_turn_count": 8,
            "completed_train_turn_count": 8,
            "failure": None,
            "session_open_count": 5,
            "session_close_count": 5,
            "session_opened_after_probe": 0,
        },
        "private_event_artifact_contract": {
            "event_count": 9,
            "train_probe_event_count": 8,
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "contains_raw_model_output": False,
            "forbidden_keys_found": [],
        },
        "closed_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
            "model_fit": False,
            "threshold_tuned": False,
            "policy_selected": False,
            "runtime_registered_as_default": False,
            "retry_actions": 0,
            "fallback_actions": 0,
            "query_rewrite": False,
            "second_retrieval": False,
        },
        "current_source_fingerprints_before": {"same": True},
        "current_source_fingerprints_after": {"same": True},
    }
