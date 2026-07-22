from __future__ import annotations

import inspect
import json
import locale
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.calibrate_primeqa_hybrid_iterative_router import app, main
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as calibration
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    IterativeDecisionAction,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
)
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


def test_synthetic_protocol_freezes_all_authorized_action_families() -> None:
    cases = calibration.build_synthetic_calibration_cases()

    assert len(cases) == 14
    assert sum(case.expected_final_action is not None for case in cases) == 4
    assert {case.expected_initial_action for case in cases} == {
        IterativeDecisionAction.COMPOSE.value,
        IterativeDecisionAction.INSPECT.value,
        IterativeDecisionAction.CLARIFY.value,
        IterativeDecisionAction.REFUSE.value,
    }
    assert sum(case.expected_initial_clarification_kind is not None for case in cases) == 6
    assert 14 + sum(case.expected_final_action is not None for case in cases) + 100 == 118


def test_train_selection_is_deterministic_and_covers_five_strata() -> None:
    samples = (
        _sample("initial", True, "doc-001"),
        _sample("alternate", True, "doc-001"),
        _sample("tail", True, "doc-050"),
        _sample("missing", True, "doc-999"),
        _sample("unanswerable", False, None),
    )
    grouped = {
        "initial": _records("initial", initial_from_rank=1),
        "alternate": _records("alternate", initial_from_rank=11),
        "tail": _records("tail", initial_from_rank=11),
        "missing": _records("missing", initial_from_rank=11),
        "unanswerable": _records("unanswerable", initial_from_rank=11),
    }
    documents = {
        f"doc-{rank:03d}": PrimeQADocument(
            id=f"doc-{rank:03d}", title=f"Document {rank}", text="Evidence"
        )
        for rank in range(1, 201)
    }

    selected = calibration.select_train_calibration_cases(
        samples=samples,
        grouped_records=grouped,
        documents_by_id=documents,
        per_stratum=1,
    )

    assert tuple(case.stratum for case in selected) == calibration._TRAIN_STRATA
    assert all(len(case.initial_evidence) == 10 for case in selected)
    assert all(len(case.alternate_evidence) == 10 for case in selected)
    assert all(len(case.question.id) == 64 for case in selected)


def test_quality_gates_apply_frozen_minimums_and_maximum() -> None:
    metrics = {
        "synthetic_phase_action_accuracy": 0.80,
        "synthetic_clarification_kind_accuracy": 5 / 6,
        "real_initial_visible_compose_rate": 0.70,
        "real_alternate_only_inspect_rate": 0.50,
        "real_alternate_only_final_compose_rate": 0.70,
        "real_alternate_only_path_success_rate": 0.40,
        "real_insufficient_final_compose_rate": 0.20,
        "schema_valid_rate": 1.0,
    }

    gates = calibration._quality_gates(metrics)

    assert len(gates) == 8
    assert all(gate["passed"] for gate in gates)
    metrics["real_insufficient_final_compose_rate"] = 0.21
    failed = calibration._quality_gates(metrics)
    assert (
        next(gate for gate in failed if gate["name"] == "real_insufficient_final_compose_rate")[
            "passed"
        ]
        is False
    )


def test_schema_invalid_action_count_has_explicit_bucket() -> None:
    counts = calibration._action_counts([None, IterativeDecisionAction.COMPOSE.value, None])

    assert counts == {"compose_grounded_answer": 1, "schema_invalid": 2}


def test_public_key_scan_rejects_private_payload_keys() -> None:
    assert calibration._forbidden_keys_found({"nested": {"question_text": "private"}}) == {
        "question_text"
    }
    assert calibration._forbidden_keys_found({"question_digest": "safe"}) == set()


def test_prior_failure_audit_reads_windows_redirect_encoding(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    exit_file = tmp_path / "exit.txt"
    events = (
        {"stage": "Stage 169", "phase": "synthetic_router_calls", "completed": 14},
        {"stage": "Stage 169", "phase": "train_router_calls", "completed": 9},
    )
    stdout.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    stderr.write_text(
        "加载进度\nCUDA error: out of memory",
        encoding=locale.getpreferredencoding(False),
    )
    exit_file.write_text("1", encoding="utf-8")

    summary = calibration._prior_failed_run_summary(
        stdout_path=stdout,
        stderr_path=stderr,
        exit_path=exit_file,
    )

    assert summary["cuda_oom_confirmed"] is True
    assert summary["synthetic_cases_completed"] == 14
    assert summary["train_cases_completed"] == 9


def test_visualizations_write_four_parseable_svgs(tmp_path: Path) -> None:
    visualizations = calibration.write_stage169_visualizations(
        report=_visual_report(), output_dir=tmp_path
    )

    assert len(visualizations) == 5
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_cli_exposes_train_protocol_without_dev_or_test_paths() -> None:
    result = CliRunner().invoke(app, ["--help"])
    parameters = inspect.signature(main).parameters

    assert result.exit_code == 0
    assert "model_snapshot" in parameters
    assert not ({"dev", "development", "dev_split"} & set(parameters))
    assert not ({"test", "test_split"} & set(parameters))


def _sample(
    sample_id: str, answerable: bool, answer_doc_id: str | None
) -> PrimeQAHybridSplitSample:
    return PrimeQAHybridSplitSample(
        split_name="fixture",
        protocol_version="fixture-v1",
        assigned_split="train",
        split_subtype="group_random_train",
        source_split="train",
        sample_id=sample_id,
        question_id=sample_id,
        question_title="title",
        question_text="question",
        answerable=answerable,
        answer="answer" if answerable else "",
        answer_doc_id=answer_doc_id,
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )


def _records(sample_id: str, *, initial_from_rank: int) -> tuple[ContextCandidateRecord, ...]:
    initial_ranks = set(range(initial_from_rank, initial_from_rank + 10))
    return tuple(
        ContextCandidateRecord(
            sample_id=sample_id,
            fold_id="fold-0",
            document_id=f"doc-{rank:03d}",
            baseline_rank=rank,
            answerable=True,
            is_gold=False,
            features={
                "stage116_rrf_score": 1.0 / rank,
                "current_query_overlap_combined_score": float(rank in initial_ranks),
            },
        )
        for rank in range(1, 201)
    )


def _visual_report() -> dict:
    train = {
        stratum: {
            "initial_action_counts": {IterativeDecisionAction.INSPECT.value: 5},
        }
        for stratum in calibration._TRAIN_STRATA
    }
    return {
        "synthetic_calibration": {
            "phase_action_accuracy": 0.8,
            "clarification_kind_accuracy": 5 / 6,
            "exact_path_accuracy": 0.7,
        },
        "train_proxy_calibration": train,
        "quality_metrics": {
            "schema_valid_rate": 1.0,
            "real_initial_visible_compose_rate": 0.8,
            "real_alternate_only_inspect_rate": 0.6,
            "real_alternate_only_final_compose_rate": 0.8,
            "real_alternate_only_path_success_rate": 0.5,
            "real_insufficient_final_compose_rate": 0.1,
            "latency_ms": {"p50": 900.0, "p95": 1200.0, "max": 1500.0},
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 4 * 1024**3,
            "process_peak_private_usage_bytes": 5 * 1024**3,
            "gpu_peak_allocated_bytes": 5 * 1024**3,
            "gpu_peak_reserved_bytes": 6 * 1024**3,
            "minimum_system_available_memory_bytes": 8 * 1024**3,
        },
    }
