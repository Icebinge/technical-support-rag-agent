from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol as protocol,
)
from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_validation as validation,
)
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_stage160_runtime_observer_captures_hashed_gold_diagnostics() -> None:
    sample = _sample()
    runtime = _FakeRuntime(_fake_run())
    observer = validation.Stage160DiagnosticRuntimeObserver(runtime, [sample])

    observer.open_thread("thread-1")
    result = observer.run_turn(
        opaque_thread_handle="thread-1",
        question=PrimeQARuntimeQuery(id="runtime", title="Title", text="Question"),
    )
    observer.close_thread("thread-1")

    assert result is runtime.run
    assert len(observer.captures) == 1
    capture = observer.captures[0]
    assert capture.private_identity_sha256 == sample.private_identity_sha256
    assert capture.gold_candidate_rank == 2
    assert capture.gold_generation_rank == 2
    assert capture.gold_cited is True
    assert capture.answer_token_f1 == 1.0
    assert capture.router_input_token_count == 2500


def test_stage160_runtime_observer_rejects_query_order_mismatch() -> None:
    observer = validation.Stage160DiagnosticRuntimeObserver(_FakeRuntime(_fake_run()), [_sample()])

    with pytest.raises(RuntimeError, match="query order mismatch"):
        observer.run_turn(
            opaque_thread_handle="thread-1",
            question=PrimeQARuntimeQuery(id="runtime", title="Wrong", text="Question"),
        )


def test_stage160_guard_fixture_passes_all_checks() -> None:
    report = _guard_report()

    checks = validation._guard_checks(report)

    assert len(checks) >= 55
    assert all(check["passed"] for check in checks)


def test_stage160_guards_reject_test_access_and_dev_tuning() -> None:
    report = _guard_report()
    report["closed_boundaries"]["test_split_loaded"] = True
    report["closed_boundaries"]["dev_used_for_threshold_tuning"] = True

    failed = {
        check["name"] for check in validation._guard_checks(report) if check["passed"] is False
    }

    assert "test_split_closed" in failed
    assert "dev_no_fit_selection_tuning" in failed


def test_stage160_decision_identifies_dominant_refusal_mechanism() -> None:
    report = _guard_report()
    report["aggregate_diagnostics"]["answerable_refusal_flow"].update(
        {
            "gold_absent_candidate_pool_refusal_count": 4,
            "gold_lost_before_generation_refusal_count": 10,
            "gold_visible_model_refusal_count": 35,
            "post_compose_refusal_count": 0,
        }
    )

    decision = validation._decision(report=report, passed=True)

    assert decision["dominant_answerable_refusal_mechanism"] == ("gold_visible_model_refusal")
    assert decision["diagnostic_only"] is True
    assert decision["policy_selected"] is False
    assert decision["test_gate_opened"] is False


def test_stage160_visualizations_write_ten_parseable_svgs(tmp_path) -> None:
    report = _guard_report()

    visualizations = validation.write_stage160_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    assert all((tmp_path / visualization.name).is_file() for visualization in visualizations)


@dataclass
class _FakeSummary:
    completed_turn_count: int = 0
    retained_state_bytes: int = 0


class _FakeRuntime:
    def __init__(self, run) -> None:
        self.run = run
        self.last_public_trace = run.public_safe_trace

    def topology(self):
        return {"node_count": 9}

    def open_thread(self, opaque_thread_handle: str):
        return _FakeSummary()

    def close_thread(self, opaque_thread_handle: str):
        return _FakeSummary()

    def thread_summary(self, opaque_thread_handle: str):
        return _FakeSummary(completed_turn_count=1, retained_state_bytes=100)

    def run_turn(self, *, opaque_thread_handle: str, question):
        return self.run


def _sample() -> protocol.Stage160DiagnosticSample:
    return protocol.Stage160DiagnosticSample(
        private_identity_sha256="1" * 64,
        query_digest_sha256=protocol.query_digest("Title", "Question"),
        diagnostic_group_sha256="2" * 64,
        answerable=True,
        gold_answer="Run the installer",
        gold_document_id="gold-doc",
        gold_document_sha256="3" * 64,
        question_route="how_to_or_lookup",
        split_subtype="random_grouped",
        runtime_query=PrimeQARuntimeQuery(
            id="stage160-dev-test",
            title="Title",
            text="Question",
        ),
    )


def _fake_run():
    wrong = RetrievalResult(
        document=PrimeQADocument(id="wrong", title="Wrong", text="Wrong content"),
        score=1.0,
        rank=1,
    )
    gold = RetrievalResult(
        document=PrimeQADocument(
            id="gold-doc",
            title="Gold",
            text="Run the installer",
        ),
        score=0.9,
        rank=2,
    )
    answer = GeneratedAnswer(
        question_id="runtime",
        answer="Run the installer",
        citations=[
            AnswerCitation(
                document_id="gold-doc",
                title="Gold",
                retrieval_rank=2,
                evidence_score=0.9,
            )
        ],
        refused=False,
    )
    metrics = SimpleNamespace(
        input_token_count=2500,
        output_token_count=9,
        generation_latency_ms=1200.0,
    )
    workflow = SimpleNamespace(
        final_state={
            "generation_context_results": (wrong, gold),
            "verification_context_results": (wrong, gold),
        },
        router_metrics=metrics,
    )
    return SimpleNamespace(
        workflow_run=workflow,
        candidate_pool_results=(wrong, gold),
        verified_answer=answer,
        public_safe_trace=SimpleNamespace(
            selected_action="compose_grounded_answer",
            terminal_state="complete",
        ),
    )


def _guard_report() -> dict:
    return {
        "user_confirmation": {
            "dev_gold_diagnostics_confirmed": True,
            "grouped_five_fold_stability_confirmed": True,
        },
        "stage159_authorization": {
            "artifact_identity_exact": True,
            "guard_count": 65,
            "all_guards_passed": True,
            "source_fingerprint_match_count": 4,
            "test_gate_opened": False,
        },
        "source_unchanged_after_validation": True,
        "stage159_artifact_unchanged_after_validation": True,
        "dev_source_unchanged_after_validation": True,
        "dev_diagnostic_protocol": {
            "source_sha256": ("071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f"),
            "dev_row_count": 121,
            "answerable_count": 76,
            "unanswerable_count": 45,
            "stable_order_sha256": (
                "3b8a39cae397db4402080a2780178ade0fd4fc3a9ba5facb25d041510e8b69b7"
            ),
            "gold_fields_used_for_diagnosis": True,
            "gold_fields_projected_into_runtime": False,
            "gold_fields_used_for_selection_or_tuning": False,
        },
        "workload_plan": {
            "grouping_sha256": ("7aa271a775c2926b32226e0a4fccc96cff3a7bf98fc90246c8002d79561fd6d0"),
            "turn_count": 121,
            "thread_count": 31,
        },
        "grouped_fold_protocol": {
            "fold_count": 5,
            "row_counts": {"0": 25, "1": 24, "2": 24, "3": 24, "4": 24},
            "fit_models": False,
            "select_policy": False,
            "tune_thresholds": False,
        },
        "startup": {
            "resource_factory_build_count": 1,
            "model_generation_call_count": 122,
            "peak_gpu_memory_bytes": 1,
        },
        "real_service": {
            "server_started": True,
            "server_thread_alive_after_shutdown": False,
            "port_rebind_after_shutdown": True,
            "health_status": {"live": 200, "ready": 200},
            "dev_http": {
                "open_http_status_counts": {"201": 31},
                "turn_http_status_counts": {"200": 121},
                "close_http_status_counts": {"200": 31},
            },
            "coordinator_counters_after_shutdown": {
                "admitted_turn_count": 121,
                "completed_turn_count": 121,
                "failed_turn_count": 0,
                "opened_thread_count": 0,
            },
        },
        "aggregate_diagnostics": {
            "overview": {
                "case_count": 121,
                "answerable_count": 76,
                "unanswerable_count": 45,
                "refusal_count": 87,
                "selected_action_counts": {
                    "compose_grounded_answer": 34,
                    "refuse_insufficient_evidence": 87,
                },
            },
            "failure_bucket_counts": {
                "answerable_refusal_gold_visible_model_refused": 50,
                "unanswerable_correct_refusal": 37,
                "unanswerable_false_answer": 8,
                "answerable_answer_gold_cited": 26,
            },
            "quality_diagnostics": {
                "answerable_refusal_rate": 0.6579,
                "unanswerable_false_answer_rate": 0.1778,
                "answerable_gold_candidate_pool_hit_rate": 0.9,
            },
            "answerable_refusal_flow": {
                "answerable_count": 76,
                "answerable_refusal_count": 50,
                "gold_absent_candidate_pool_refusal_count": 5,
                "gold_lost_before_generation_refusal_count": 5,
                "gold_visible_model_refusal_count": 40,
                "post_compose_refusal_count": 0,
            },
            "by_turn_position": {
                str(position): {
                    "end_to_end_latency_ms": {"average": 1000 * position},
                    "router_generation_latency_ms": {"average": 900 * position},
                }
                for position in range(1, 5)
            },
            "by_selected_action": {
                "compose_grounded_answer": {"router_generation_latency_ms": {"average": 1200}},
                "refuse_insufficient_evidence": {"router_generation_latency_ms": {"average": 1800}},
            },
            "latency_diagnostics": {
                "generation_share_of_total_average": 0.9,
                "spearman_correlations_with_generation_latency": {
                    "turn_position": 0.5,
                    "router_input_token_count": 0.6,
                },
            },
            "fold_diagnostic_stability": {
                "fold_count": 5,
                "fit_models": False,
                "folds": {
                    str(fold): {"answerable_refusal_rate": 0.5 + fold * 0.01} for fold in range(5)
                },
            },
        },
        "private_diagnostic_artifact_contract": {
            "row_count": 121,
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "public_report_contains_case_rows": False,
        },
        "closed_boundaries": {
            "test_split_loaded": False,
            "test_metrics_run": False,
            "dev_used_for_model_fit": False,
            "dev_used_for_policy_selection": False,
            "dev_used_for_threshold_tuning": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "persistent_state_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
            "queue_action_count": 0,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
        "guard_checks": [],
    }
