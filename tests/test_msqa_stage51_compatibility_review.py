import json
from pathlib import Path

import pytest

from ts_rag_agent.application.msqa_stage51_compatibility_review import (
    review_msqa_stage51_compatibility,
    write_msqa_stage51_compatibility_visualizations,
)


def test_msqa_stage51_compatibility_review_blocks_direct_candidate_run(tmp_path):
    stage58_report = tmp_path / "stage58.json"
    _write_stage58_report(stage58_report)

    report = review_msqa_stage51_compatibility(stage58_report)

    assert report["stage"] == "Stage 59"
    assert report["decision"]["status"] == "stage51_msqa_compatibility_blocked"
    assert report["decision"]["can_run_stage51_candidate_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    gate = report["compatibility_gate"]["summary"]
    assert gate["status_counts"] == {"blocked": 5, "pass": 2}
    assert gate["blocker_count"] == 5
    assert "stage51_task_semantics_match_msqa" in gate["blocker_checks"]


def test_msqa_stage51_compatibility_review_summarizes_failures_and_gaps(tmp_path):
    stage58_report = tmp_path / "stage58.json"
    _write_stage58_report(stage58_report)

    report = review_msqa_stage51_compatibility(stage58_report)

    assert report["stage58_baseline_summary"]["max_k"] == 10
    assert report["failure_mode_review"]["primary_failure_rates"] == {
        "gold_source_missing_at_10": 0.4,
        "top1_wrong_source": 0.5,
        "top1_token_f1_below_0_3": 0.3,
    }
    assert report["failure_mode_review"]["primary_vs_diagnostic_gap"] == {
        "hit@1": 0.5,
        "hit@10": 0.4,
        "mrr": 0.45,
        "average_top1_token_f1": 0.48,
    }


def test_msqa_stage51_compatibility_visualizations_are_written(tmp_path):
    stage58_report = tmp_path / "stage58.json"
    _write_stage58_report(stage58_report)
    report = review_msqa_stage51_compatibility(stage58_report)

    artifacts = write_msqa_stage51_compatibility_visualizations(
        report,
        tmp_path / "visuals",
    )

    assert {artifact.name for artifact in artifacts} == {
        "stage59_msqa_stage51_gate_checks.svg",
        "stage59_msqa_answer_only_failure_modes.svg",
        "stage59_msqa_variant_metric_comparison.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")


def test_msqa_stage51_compatibility_requires_stage58_report(tmp_path):
    not_stage58 = tmp_path / "not_stage58.json"
    not_stage58.write_text(json.dumps({"stage": "Stage 57"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a Stage 58 report"):
        review_msqa_stage51_compatibility(not_stage58)


def _write_stage58_report(path: Path) -> None:
    report = {
        "stage": "Stage 58",
        "created_at": "2026-07-14",
        "input_contract": {
            "split_name": "msqa_stage57_project_eval_v1",
            "adapter_contract_version": "msqa_eval_adapter_v1",
        },
        "baseline_definition": {
            "primary_variant": "answer_only",
            "diagnostic_variant": "question_answer_page_text",
            "corpus_scope": "frozen_split_only",
        },
        "data": {
            "corpus_contract_rows": 10,
            "rejected_contract_rows": {},
            "frozen_split_samples": 10,
        },
        "variants": [
            {
                "corpus_mode": "answer_only",
                "retrieval_metrics": {
                    "evaluated_questions": 10,
                    "hit_at_k": {"hit@1": 0.5, "hit@10": 0.6},
                    "mrr": 0.55,
                    "gold_source_not_found_at_max_k": 4,
                },
                "answer_metrics": {
                    "average_top1_token_f1": 0.52,
                    "oracle_answer_token_f1_at_k": {"oracle@10": 0.7},
                    "top1_low_f1_threshold": 0.3,
                    "top1_low_f1_count": 3,
                },
                "failure_mode_counts": {
                    "gold_source_missing_at_10": 4,
                    "top1_wrong_source": 5,
                    "top1_token_f1_below_0_3": 3,
                },
            },
            {
                "corpus_mode": "question_answer_page_text",
                "retrieval_metrics": {
                    "evaluated_questions": 10,
                    "hit_at_k": {"hit@1": 1.0, "hit@10": 1.0},
                    "mrr": 1.0,
                    "gold_source_not_found_at_max_k": 0,
                },
                "answer_metrics": {
                    "average_top1_token_f1": 1.0,
                    "oracle_answer_token_f1_at_k": {"oracle@10": 1.0},
                    "top1_low_f1_threshold": 0.3,
                    "top1_low_f1_count": 0,
                },
                "failure_mode_counts": {
                    "gold_source_missing_at_10": 0,
                    "top1_wrong_source": 0,
                    "top1_token_f1_below_0_3": 0,
                },
            },
        ],
        "decision": {
            "status": "msqa_topk_baseline_recorded",
            "can_run_stage51_candidate_now": False,
            "can_defaultize_runtime_now": False,
            "default_runtime_policy": "unchanged",
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
