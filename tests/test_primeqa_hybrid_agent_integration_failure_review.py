from __future__ import annotations

import json

from ts_rag_agent.application.primeqa_hybrid_agent_integration_failure_review import (
    _action_boundary,
    _failure_patterns,
    _public_safe_contract,
    _region_shift,
    _split_deltas,
    review_primeqa_hybrid_agent_integration_failure_patterns,
)


def test_split_deltas_capture_recall_gain_and_citation_loss() -> None:
    candidate = _split_report(f1=0.1949, citation_count=150, hit_count=354)
    control = _split_report(f1=0.1946, citation_count=151, hit_count=345)

    deltas = _split_deltas(candidate=candidate, control=control)

    assert deltas["verified_average_token_f1_delta"] == 0.0003
    assert deltas["verified_gold_citation_count_delta"] == -1
    assert deltas["gold_hit_count_at_profile_depth_delta"] == 9


def test_region_shift_detects_append_displacement() -> None:
    shift = _region_shift(
        control_regions={
            "rank_001_010": 1043,
            "stage116_immutable_prefix_011_200": 637,
        },
        candidate_regions={
            "rank_001_010": 1027,
            "stage116_immutable_prefix_011_200": 611,
            "stage128_append_expansion_201_400": 42,
        },
    )

    assert shift["append_region_selected_citation_count"] == 42
    assert shift["prefix_like_selected_citation_delta"] == -42
    assert shift["append_displacement_balance"] == 0


def test_failure_patterns_block_direct_stage128_path() -> None:
    train_review = _review(split="train_cv", rows=562, changed=221)
    dev_review = _review(split="dev", rows=121, changed=50)

    patterns = _failure_patterns(train_review=train_review, dev_review=dev_review)
    action = _action_boundary(patterns)

    assert patterns[0]["pattern_id"] == "recall_gain_not_citation_safe"
    assert patterns[0]["severity"] == "blocking"
    assert action["stage128_direct_agent_integration_path_blocked"] is True
    assert action["stage128_final_test_gate_allowed_now"] is False


def test_public_safe_contract_flags_forbidden_keys() -> None:
    public_safe = _public_safe_contract({"unsafe": {"document_id": "private-doc"}})

    assert public_safe["forbidden_keys_found"] == ["document_id"]
    assert public_safe["test_split_loaded"] is False
    assert public_safe["final_test_metrics_run"] is False


def test_review_blocks_without_confirmation(tmp_path) -> None:
    path = tmp_path / "stage129.json"
    path.write_text(json.dumps(_stage129_report()), encoding="utf-8")

    report = review_primeqa_hybrid_agent_integration_failure_patterns(
        stage129_report_path=path,
        user_confirmed_review=False,
        confirmation_note="not confirmed",
    )

    failed = [check["name"] for check in report["guard_checks"] if not check["passed"]]
    assert failed == ["user_confirmed_stage130_review"]
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage129_agent_integration_failure_review_blocked"
    )


def test_review_completes_with_expected_stage129_report(tmp_path) -> None:
    path = tmp_path / "stage129.json"
    path.write_text(json.dumps(_stage129_report()), encoding="utf-8")

    report = review_primeqa_hybrid_agent_integration_failure_patterns(
        stage129_report_path=path,
        user_confirmed_review=True,
        confirmation_note="user confirmed Stage130 review",
    )

    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage129_agent_integration_failure_review_completed"
    )
    assert report["decision"]["recommended_next_direction"] == (
        "freeze_append_candidate_evidence_shortlist_redesign_protocol"
    )
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])


def _review(*, split: str, rows: int, changed: int) -> dict:
    candidate = _split_report(f1=0.1949, citation_count=150, hit_count=354)
    control = _split_report(f1=0.1946, citation_count=151, hit_count=345)
    deltas = _split_deltas(candidate=candidate, control=control)
    shift = _region_shift(
        control_regions={
            "rank_001_010": 1043,
            "stage116_immutable_prefix_011_200": 637,
        },
        candidate_regions={
            "rank_001_010": 1027,
            "stage116_immutable_prefix_011_200": 611,
            "stage128_append_expansion_201_400": 42,
        },
    )
    return {
        "split": split,
        "row_count": rows,
        "candidate_vs_control_deltas": deltas,
        "selected_citation_region_shift": shift,
        "changed_verified_answers_vs_control": changed,
        "changed_verified_answer_rate_vs_control": round(changed / rows, 4),
    }


def _split_report(*, f1: float, citation_count: int, hit_count: int) -> dict:
    return {
        "verified_metrics": {
            "average_token_f1": f1,
            "gold_doc_citation_rate": 0.4,
            "answerable_refusal_rate": 0.0,
            "unanswerable_refusal_rate": 0.0,
        },
        "retrieval_summary": {
            "gold_hit_count_at_profile_depth": hit_count,
            "gold_hit_rate_at_profile_depth": round(hit_count / 370, 4),
            "gold_miss_count_at_profile_depth": 370 - hit_count,
        },
        "selected_evidence_summary": {
            "gold_citation_count": citation_count,
            "citation_count": 1680,
            "answered_count": 560,
            "rank_region_counts": {
                "rank_001_010": 1027,
                "stage116_immutable_prefix_011_200": 611,
                "stage128_append_expansion_201_400": 42,
            },
        },
        "row_count": 562,
        "changed_verified_answers_vs_stage116_control": 221,
    }


def _stage129_report() -> dict:
    control_train = _split_report(f1=0.1946, citation_count=151, hit_count=345)
    candidate_train = _split_report(f1=0.1949, citation_count=150, hit_count=354)
    control_dev = _split_report(f1=0.1873, citation_count=33, hit_count=69)
    candidate_dev = _split_report(f1=0.1837, citation_count=31, hit_count=70)
    control_train["selected_evidence_summary"]["rank_region_counts"] = {
        "rank_001_010": 1043,
        "stage116_immutable_prefix_011_200": 637,
    }
    control_dev["selected_evidence_summary"]["rank_region_counts"] = {
        "rank_001_010": 220,
        "stage116_immutable_prefix_011_200": 143,
    }
    candidate_dev["selected_evidence_summary"]["rank_region_counts"] = {
        "rank_001_010": 210,
        "stage116_immutable_prefix_011_200": 141,
        "stage128_append_expansion_201_400": 12,
    }
    candidate_dev["changed_verified_answers_vs_stage116_control"] = 50
    return {
        "stage": "Stage 129",
        "analysis_id": "primeqa_hybrid_agent_retrieval_integration_validation_v1",
        "decision": {
            "status": (
                "primeqa_hybrid_agent_retrieval_integration_validation_blocked_or_failed"
            ),
            "recommended_next_direction": (
                "review_stage129_agent_integration_failure_patterns"
            ),
            "selected_profile_id": "stage128_prefix_append_top400_agent_pool",
            "train_cv_validation_passed": False,
            "train_cv_failed_checks": [
                "gold_citation_count_delta_vs_stage116_non_negative"
            ],
            "failed_checks": ["stage129_agent_answer_quality_train_cv_guard"],
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
            "dev_gate_status": "report_only_no_runtime_or_test_gate",
        },
        "train_cv_validation": {
            "selection_mode": "train_grouped_cross_validation_agent_integration_validation",
            "selected_profile_id": "stage128_prefix_append_top400_agent_pool",
        },
        "guard_checks": [
            {"name": f"check_{index}", "passed": index < 20}
            for index in range(21)
        ],
        "profile_reports": {
            "stage116_top200_agent_pool_control": {
                "split_reports": {"train_cv": control_train, "dev": control_dev}
            },
            "stage128_prefix_append_top400_agent_pool": {
                "split_reports": {"train_cv": candidate_train, "dev": candidate_dev}
            },
        },
        "public_safe_contract": {"forbidden_keys_found": []},
    }
