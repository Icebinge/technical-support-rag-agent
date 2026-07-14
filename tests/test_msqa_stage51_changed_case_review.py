from __future__ import annotations

import json
from pathlib import Path

import pytest

from ts_rag_agent.application import msqa_stage51_adapter_comparison
from ts_rag_agent.application.answer_composition import AnswerCompositionDecision
from ts_rag_agent.application.candidate_score_guarded_composition_policy import (
    RuntimeCandidateRerankerDecisionTrace,
)
from ts_rag_agent.application.evidence_selection import classify_question_route
from ts_rag_agent.application.msqa_stage51_adapter_comparison import (
    compare_msqa_stage51_capped_adapter,
)
from ts_rag_agent.application.msqa_stage51_changed_case_review import (
    review_msqa_stage51_changed_cases,
    write_msqa_stage51_changed_case_review_visualizations,
)


def test_msqa_stage51_changed_case_review_rebuilds_stage64_cases(tmp_path, monkeypatch):
    paths = _write_regressed_fixture(tmp_path)
    _patch_rank2_policy(monkeypatch)
    stage64_report = _write_stage64_report(paths, tmp_path)

    report = review_msqa_stage51_changed_cases(
        stage64_report_path=stage64_report,
        split_jsonl_path=paths["split"],
        candidate_jsonl_path=paths["candidates"],
        adapter_report_path=paths["adapter_report"],
        distribution_report_path=paths["distribution_report"],
        candidate_reranker_dataset_path=paths["reranker_dataset"],
        stage31_summary_path=paths["stage31_summary"],
        max_answer_candidates=1,
        max_citation_rank=1,
        sample_limit=1,
    )

    assert report["stage"] == "Stage 65"
    assert report["rebuild_contract"]["candidate_pool_rebuilt"] is False
    assert all(check["passed"] for check in report["consistency_checks"])
    assert report["changed_case_summary"]["question_count"] == 1
    assert report["changed_case_summary"]["changed_answer_count"] == 1
    assert report["changed_case_summary"]["top3_regression_count"] == 1
    assert report["changed_case_summary"]["citation_delta"] == 0
    assert report["decision"]["status"] == (
        "msqa_stage51_changed_case_review_blocks_defaultization"
    )

    artifacts = write_msqa_stage51_changed_case_review_visualizations(
        report,
        tmp_path / "visuals",
    )
    assert {artifact.name for artifact in artifacts} == {
        "stage65_msqa_changed_outcomes.svg",
        "stage65_msqa_regressions_by_route.svg",
        "stage65_msqa_changed_by_selected_rank.svg",
        "stage65_msqa_source_transitions.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def test_msqa_stage51_changed_case_review_flags_metric_mismatch(tmp_path, monkeypatch):
    paths = _write_regressed_fixture(tmp_path)
    _patch_rank2_policy(monkeypatch)
    stage64_report = _write_stage64_report(paths, tmp_path)
    report_json = json.loads(stage64_report.read_text(encoding="utf-8"))
    report_json["metrics"]["changed_answer_count"] = 99
    stage64_report.write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = review_msqa_stage51_changed_cases(
        stage64_report_path=stage64_report,
        split_jsonl_path=paths["split"],
        candidate_jsonl_path=paths["candidates"],
        adapter_report_path=paths["adapter_report"],
        distribution_report_path=paths["distribution_report"],
        candidate_reranker_dataset_path=paths["reranker_dataset"],
        stage31_summary_path=paths["stage31_summary"],
        max_answer_candidates=1,
        max_citation_rank=1,
    )

    failed_checks = [
        check["name"] for check in report["consistency_checks"] if not check["passed"]
    ]
    assert failed_checks == ["changed_answer_count_matches_stage64"]
    assert report["decision"]["status"] == (
        "msqa_stage51_changed_case_review_blocked_by_inconsistent_rebuild"
    )


def _patch_rank2_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        msqa_stage51_adapter_comparison,
        "fit_candidate_score_guarded_reranker_composition_policy",
        _fake_fit_rank2_policy,
    )


def _fake_fit_rank2_policy(*args, **kwargs) -> _Rank2Policy:
    return _Rank2Policy()


class _Rank2Policy:
    name = "candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker"

    def __init__(self) -> None:
        self._last_trace: RuntimeCandidateRerankerDecisionTrace | None = None

    @property
    def last_trace(self) -> RuntimeCandidateRerankerDecisionTrace | None:
        return self._last_trace

    def select(self, question, candidates, max_sentences):
        selected = candidates[1:2]
        self._last_trace = RuntimeCandidateRerankerDecisionTrace(
            action="replace_with_model_candidate",
            reason="candidate_score_gte_60_accepted",
            selected_candidate_rank=2,
            selected_candidate_score=round(candidates[1].score, 4),
            model_score_margin_vs_top_candidate=0.5,
            proposed_worst_retrieval_rank=1,
            rank_contained_max_retrieval_rank=1,
            preserve_baseline_out_of_rank_docs=True,
        )
        return AnswerCompositionDecision(
            selected_candidates=selected[:max_sentences],
            question_route=classify_question_route(question),
            strategy=self.name,
            reason="candidate_score_gte_60_accepted",
        )


def _write_stage64_report(paths: dict[str, Path], tmp_path: Path) -> Path:
    report = compare_msqa_stage51_capped_adapter(
        split_jsonl_path=paths["split"],
        candidate_jsonl_path=paths["candidates"],
        adapter_report_path=paths["adapter_report"],
        distribution_report_path=paths["distribution_report"],
        candidate_reranker_dataset_path=paths["reranker_dataset"],
        stage31_summary_path=paths["stage31_summary"],
        max_answer_candidates=1,
        max_citation_rank=1,
        sample_limit=1,
    )
    assert report["decision"]["status"] == (
        "msqa_stage51_capped_adapter_comparison_f1_regressed"
    )
    stage64_report = tmp_path / "stage64_report.json"
    stage64_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stage64_report


def _write_regressed_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "split": tmp_path / "split.jsonl",
        "candidates": tmp_path / "candidates.jsonl",
        "adapter_report": tmp_path / "adapter_report.json",
        "distribution_report": tmp_path / "distribution_report.json",
        "reranker_dataset": tmp_path / "reranker_dataset.jsonl",
        "stage31_summary": tmp_path / "stage31_summary.json",
    }
    _write_jsonl(
        paths["split"],
        [
            {
                "dataset": "microsoft_msqa",
                "split": "msqa_stage57_project_eval_v1",
                "adapter_contract_version": "msqa_eval_adapter_v1",
                "question_id": "q1",
                "answer_id": "a1",
                "question": "How do I reset an Azure password?",
                "answer": "Reset the password in the Azure portal.",
                "source_url": "https://learn.microsoft.com/q1",
                "metadata": {},
            }
        ],
    )
    _write_jsonl(
        paths["candidates"],
        [
            _candidate_row(
                candidate_id="q1::good",
                sentence="Reset the password in the Azure portal.",
                candidate_score=90.0,
            ),
            _candidate_row(
                candidate_id="q1::bad",
                sentence="Open unrelated billing logs for review.",
                candidate_score=80.0,
            ),
        ],
    )
    _write_adapter_report(paths["adapter_report"])
    _write_distribution_report(paths["distribution_report"])
    _write_jsonl(paths["reranker_dataset"], [_reranker_row()])
    _write_stage31_summary(paths["stage31_summary"])
    return paths


def _candidate_row(
    *,
    candidate_id: str,
    sentence: str,
    candidate_score: float,
) -> dict:
    return {
        "query_question_id": "q1",
        "query_answer_id": "a1",
        "gold_source_row_id": "q1",
        "gold_source_url": "https://learn.microsoft.com/q1",
        "question_id": "q1",
        "answer_id": f"a1-{candidate_id}",
        "source_url": "https://learn.microsoft.com/q1",
        "source_row_id": "q1",
        "candidate_id": candidate_id,
        "candidate_row_id": f"q1::{candidate_id}",
        "candidate_sentence": sentence,
        "retrieval_rank": 1,
        "retrieval_score": 100.0,
        "candidate_score": candidate_score,
        "overlap_terms": ["azure", "password"],
    }


def _write_adapter_report(path: Path) -> None:
    report = {
        "stage": "Stage 63",
        "adapter_contract": {
            "top_k": 5,
            "max_candidates_per_source_row": 3,
            "effective_candidate_pool_cap": 15,
        },
        "decision": {
            "status": "msqa_stage31_aligned_candidate_adapter_dry_run_passed",
            "stage51_candidate_run_performed": False,
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_distribution_report(path: Path) -> None:
    report = {
        "stage": "Stage 63",
        "adapter_candidate_distribution": {
            "gold_source_candidate_rate": 1.0,
        },
        "stage31_candidate_distribution": {
            "gold_document_candidate_rate": 1.0,
        },
        "candidate_pool_comparison": {
            "gold_candidate_rate_delta_adapter_minus_stage31": 0.0,
        },
        "decision": {
            "status": "msqa_stage51_adapter_comparison_ready_for_user_confirmation",
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _reranker_row() -> dict:
    return {
        "split": "train",
        "question_id": "t1",
        "candidate_id": "t1::candidate_001",
        "candidate_rank": 1,
        "runtime_features": {
            "selector_name": "test_selector",
            "question_route": "install_upgrade_config",
            "retrieval_rank": 1,
            "retrieval_score": 100.0,
            "candidate_score": 90.0,
            "candidate_token_count": 6,
            "candidate_sentence_count": 1,
            "question_token_count": 6,
            "query_term_count": 3,
            "query_overlap_count": 2,
            "query_overlap_ratio": 0.66,
            "candidate_query_coverage_ratio": 0.5,
            "title_query_overlap_count": 0,
            "title_query_overlap_ratio": 0.0,
            "answer_signal_score": 1.0,
            "problem_noise_score": 0.0,
            "has_answer_heading": False,
            "has_problem_heading": False,
            "has_question_heading": False,
            "has_url": False,
            "has_trace_noise": False,
            "symbol_ratio": 0.0,
        },
        "gold_labels": {
            "candidate_token_f1": 1.0,
            "is_best_candidate_for_question": True,
            "is_gold_document": True,
        },
        "metadata": {},
    }


def _write_stage31_summary(path: Path) -> None:
    report = {
        "dataset": "PrimeQA/TechQA",
        "build_config": {
            "retrieval_top_k": 5,
            "max_candidates_per_document": 3,
            "candidate_limit": 25,
            "evidence_selector": "test_selector",
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
