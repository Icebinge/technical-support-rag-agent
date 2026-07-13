from ts_rag_agent.application.answer_gap_priority_analysis import (
    analyze_answer_gap_priorities,
    answer_gap_priority_analysis_to_dict,
)


def test_answer_gap_priority_analysis_ranks_route_bucket_issues():
    report = {
        "cases": [
            _case(
                question_id="q1",
                route="install_upgrade_config",
                bucket="gold_in_context_not_selected",
                selected_f1=0.1,
                best_window_f1=0.8,
                gold_rank=1,
                selected_gold_count=0,
            ),
            _case(
                question_id="q2",
                route="install_upgrade_config",
                bucket="gold_window_beats_selected_answer",
                selected_f1=0.2,
                best_window_f1=0.9,
                gold_rank=2,
                selected_gold_count=1,
            ),
            _case(
                question_id="q3",
                route="other",
                bucket="selected_answer_reasonable_overlap",
                selected_f1=0.5,
                best_window_f1=0.6,
                gold_rank=1,
                selected_gold_count=1,
            ),
        ]
    }

    analysis = analyze_answer_gap_priorities([report], min_cases=1)
    result_dict = answer_gap_priority_analysis_to_dict(analysis)

    assert analysis.total_cases == 3
    assert analysis.route_priorities[0].group_key == "install_upgrade_config"
    assert analysis.route_priorities[0].sample_cases[0].question_id == "q1"
    assert analysis.route_bucket_priorities[0].group_key.startswith(
        "install_upgrade_config::"
    )
    assert result_dict["priority_score_note"].startswith("Heuristic only")


def test_answer_gap_priority_analysis_rejects_missing_cases():
    try:
        analyze_answer_gap_priorities([{"summary": {}}])
    except ValueError as exc:
        assert "cases" in str(exc)
    else:
        raise AssertionError("report without cases should fail")


def _case(
    question_id: str,
    route: str,
    bucket: str,
    selected_f1: float,
    best_window_f1: float,
    gold_rank: int | None,
    selected_gold_count: int,
) -> dict:
    return {
        "question_id": question_id,
        "question_route": route,
        "bucket": bucket,
        "selected_answer_token_f1": selected_f1,
        "gold_retrieval_rank": gold_rank,
        "selected_gold_candidate_count": selected_gold_count,
        "best_gold_window": {"token_f1": best_window_f1},
    }
