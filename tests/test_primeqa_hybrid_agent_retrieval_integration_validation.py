from __future__ import annotations

from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _candidate_pool_summary,
    _decision,
    _profile_configs,
    _public_safe_contract,
    _selected_append_config,
    _split_deltas,
    _stage128_summary,
)


def test_stage128_summary_and_selected_config_are_joined_from_protocols() -> None:
    summary = _stage128_summary(_stage128_protocol())
    selected = _selected_append_config(
        stage125_protocol=_stage125_protocol(),
        stage128_summary=summary,
    )

    assert summary["status"] == "primeqa_hybrid_agent_retrieval_integration_protocol_frozen"
    assert summary["recommended_next_direction"] == (
        "run_agent_retrieval_integration_train_cv_dev_validation"
    )
    assert selected is not None
    assert selected["config_id"] == "prefix_existing_dense_broad_append200_v1"
    assert selected["append_generation"]["rrf_k"] == 60


def test_candidate_pool_summary_detects_prefix_and_append_guard_failures() -> None:
    summary = _candidate_pool_summary(
        {
            "train": {
                "sample-1": {
                    "prefix_identity_violation_count": 0,
                    "append_count": 200,
                },
                "sample-2": {
                    "prefix_identity_violation_count": 2,
                    "append_count": 201,
                },
            },
            "dev": {
                "sample-3": {
                    "prefix_identity_violation_count": 0,
                    "append_count": 199,
                }
            },
        }
    )

    assert summary["all_splits_prefix_identity_violation_count"] == 2
    assert summary["all_splits_append_budget_exceeded_count"] == 1
    assert summary["splits"]["train"]["append_count"]["max"] == 201


def test_split_deltas_use_verified_answer_and_retrieval_metrics() -> None:
    candidate = _split_report(
        f1=0.35,
        citation_rate=0.6,
        refusal_rate=0.1,
        unanswerable_refusal_rate=0.9,
        gold_citation_count=8,
        hit_count=12,
        hit_rate=0.8,
        changed=3,
    )
    baseline = _split_report(
        f1=0.3,
        citation_rate=0.5,
        refusal_rate=0.2,
        unanswerable_refusal_rate=0.85,
        gold_citation_count=7,
        hit_count=10,
        hit_rate=0.7,
        changed=0,
    )

    deltas = _split_deltas(
        candidate=candidate,
        baseline=baseline,
        changed_answer_count=3,
    )

    assert deltas["verified_average_token_f1_delta"] == 0.05
    assert deltas["verified_gold_doc_citation_rate_delta"] == 0.1
    assert deltas["verified_gold_citation_count_delta"] == 1
    assert deltas["answerable_refusal_rate_delta"] == -0.1
    assert deltas["gold_hit_count_at_profile_depth_delta"] == 2
    assert deltas["changed_verified_answers"] == 3


def test_decision_blocks_runtime_default_even_when_validation_passes() -> None:
    decision = _decision(
        guard_checks=[{"name": "all_good", "passed": True}],
        train_cv_validation={"passed": True, "failed_checks": []},
        dev_report={
            "status": "reported_not_used_for_selection",
            "dev_gate_status": "report_only_no_runtime_or_test_gate",
        },
    )

    assert decision["status"] == (
        "primeqa_hybrid_agent_retrieval_integration_validation_completed"
    )
    assert decision["can_run_final_test_metrics_now"] is False
    assert decision["runtime_defaultization_allowed_now"] is False
    assert decision["default_runtime_policy"] == "unchanged"


def test_profile_configs_include_baseline_stage116_and_stage128_profiles() -> None:
    profiles = _profile_configs(_selected_config())

    assert [profile.profile_id for profile in profiles] == [
        "stage102_bm25_top10_verified_baseline",
        "stage116_top200_agent_pool_control",
        "stage128_prefix_append_top400_agent_pool",
    ]
    assert profiles[0].verifier_max_citation_rank == 3
    assert profiles[1].verifier_max_citation_rank == 200
    assert profiles[2].verifier_max_citation_rank == 400


def test_public_safe_contract_flags_forbidden_keys() -> None:
    report = {
        "safe_summary": {"count": 1},
        "unsafe": {"question_text": "raw question"},
    }

    public_safe = _public_safe_contract(report)

    assert public_safe["forbidden_keys_found"] == ["question_text"]
    assert public_safe["test_split_loaded"] is False
    assert public_safe["final_test_metrics_run"] is False


def _split_report(
    *,
    f1: float,
    citation_rate: float,
    refusal_rate: float,
    unanswerable_refusal_rate: float,
    gold_citation_count: int,
    hit_count: int,
    hit_rate: float,
    changed: int,
) -> dict:
    return {
        "verified_metrics": {
            "average_token_f1": f1,
            "gold_doc_citation_rate": citation_rate,
            "answerable_refusal_rate": refusal_rate,
            "unanswerable_refusal_rate": unanswerable_refusal_rate,
        },
        "retrieval_summary": {
            "gold_hit_count_at_profile_depth": hit_count,
            "gold_hit_rate_at_profile_depth": hit_rate,
        },
        "selected_evidence_summary": {
            "gold_citation_count": gold_citation_count,
        },
        "changed_verified_answers_vs_stage116_control": changed,
    }


def _stage128_protocol() -> dict:
    return {
        "stage": "Stage 128",
        "protocol_id": "primeqa_hybrid_agent_retrieval_integration_protocol_v1",
        "decision": {
            "status": "primeqa_hybrid_agent_retrieval_integration_protocol_frozen",
            "recommended_next_direction": (
                "run_agent_retrieval_integration_train_cv_dev_validation"
            ),
            "selected_config_id": "prefix_existing_dense_broad_append200_v1",
            "selected_family_id": "stage116_prefix_existing_dense_append_family_v1",
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "frozen_protocol": {
            "selected_retrieval_config": {
                "config_id": "prefix_existing_dense_broad_append200_v1",
                "family_id": "stage116_prefix_existing_dense_append_family_v1",
            },
            "agent_retrieval_contract": {
                "candidate_pool_output_depth": 400,
                "candidate_pool_is_not_automatic_answer_context": True,
                "answer_context_policy": "unchanged_until_stage129_validation",
            },
            "validation_plan": {
                "next_stage": "Stage129",
                "action": "run_agent_retrieval_integration_train_cv_dev_validation",
            },
        },
        "public_safe_contract": {"forbidden_keys_found": []},
    }


def _stage125_protocol() -> dict:
    return {
        "frozen_protocol": {
            "candidate_configs": [
                _selected_config(),
                {
                    "config_id": "other",
                    "family_id": "other_family",
                    "append_generation": {},
                },
            ]
        }
    }


def _selected_config() -> dict:
    return {
        "config_id": "prefix_existing_dense_broad_append200_v1",
        "family_id": "stage116_prefix_existing_dense_append_family_v1",
        "source_stage124_config_id": "existing_dense_cache_broad_union_top400_v1",
        "append_generation": {
            "append_source_algorithm": "cached_dense_plus_lexical_rrf",
            "route_set": "stage116_lexical_routes_plus_existing_dense_cache_routes",
            "channel_top_k": 400,
            "rrf_k": 60,
            "append_start_rank": 201,
            "append_budget": 200,
            "target_pool_depth": 400,
        },
    }
