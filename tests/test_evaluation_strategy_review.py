from ts_rag_agent.application.evaluation_strategy_review import (
    review_evaluation_strategy,
    write_evaluation_strategy_visualizations,
)


def test_evaluation_strategy_review_rejects_leaking_nvidia_and_requires_choice():
    review = review_evaluation_strategy(
        readiness_review=_readiness_review(),
        leakage_report=_blocked_leakage_report(),
    )

    assert review["current_facts"]["nvidia_train_json_blocked_as_heldout"] is True
    assert review["current_facts"]["default_runtime_policy"] == "unchanged"
    assert review["rejected_paths"][0]["label"] == "use_nvidia_train_json_as_current_heldout"
    assert review["decision_required"]["requires_user_confirmation"] is True
    assert (
        review["decision_required"]["recommended_for_confirmation"]
        == "external_independent_eval_set"
    )
    assert all(
        option["status"] == "available_after_user_confirmation"
        for option in review["strategy_options"]
    )


def test_evaluation_strategy_review_option_properties_are_distinct():
    review = review_evaluation_strategy(
        readiness_review=_readiness_review(),
        leakage_report=_blocked_leakage_report(),
    )
    options = {option["label"]: option for option in review["strategy_options"]}

    assert options["external_independent_eval_set"]["can_support_defaultization"] is True
    assert options["external_independent_eval_set"]["keeps_stage51_candidate_frozen"] is True
    assert options["rebuild_leak_safe_primeqa_split"]["requires_full_pipeline_rerun"] is True
    assert options["rebuild_leak_safe_primeqa_split"]["keeps_stage51_candidate_frozen"] is False
    assert options["freeze_without_defaultization"]["can_support_defaultization"] is False


def test_evaluation_strategy_visualizations_are_written(tmp_path):
    review = review_evaluation_strategy(
        readiness_review=_readiness_review(),
        leakage_report=_blocked_leakage_report(),
    )

    artifacts = write_evaluation_strategy_visualizations(review, tmp_path)

    assert {artifact.name for artifact in artifacts} == {
        "stage54_option_validity_score.svg",
        "stage54_option_effort_score.svg",
        "stage54_option_defaultization_support.svg",
        "stage54_blocked_nvidia_overlap.svg",
    }
    for artifact in artifacts:
        assert (tmp_path / artifact.name).read_text(encoding="utf-8").startswith("<svg")


def _readiness_review() -> dict:
    return {
        "candidate_policy": (
            "candidate_score_gte_60_rank_contained_"
            "preserve_baseline_out_of_rank_guarded_reranker"
        ),
        "overall_decision": {
            "candidate_passes_dev_train_readiness": True,
        },
    }


def _blocked_leakage_report() -> dict:
    return {
        "heldout_usable_without_exclusions": False,
        "counts": {
            "heldout_questions": 910,
            "exact_overlap_count": 910,
            "unhandled_overlap_count": 910,
        },
    }
