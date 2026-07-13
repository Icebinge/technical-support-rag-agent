from ts_rag_agent.application.other_route_window_outcome_analysis import (
    analyze_other_route_window_outcomes,
    classify_other_route_subtype,
    other_route_window_outcome_analysis_to_dict,
)


def test_classify_other_route_subtype_uses_runtime_visible_text():
    assert (
        classify_other_route_subtype("Is there a support's guide to the MTTrapd probe?")
        == "support_or_download"
    )
    assert (
        classify_other_route_subtype("Can I hide left menu pane for specific application?")
        == "procedure_or_change"
    )
    assert (
        classify_other_route_subtype("Is WebSphere Extreme Scale leap year compliant?")
        == "capability_or_support"
    )


def test_other_route_window_outcome_analysis_requires_cross_source_stability():
    baseline_reports = [
        {
            "cases": [
                _case(
                    question_id="dev-q1",
                    title="Can I hide left menu pane?",
                    route="other",
                    f1=0.7,
                    gold_cited=True,
                ),
                _case(
                    question_id="dev-q2",
                    title="Can I change the default pane?",
                    route="other",
                    f1=0.6,
                    gold_cited=True,
                ),
                _case(
                    question_id="dev-q3",
                    title="Can I migrate search templates?",
                    route="other",
                    f1=0.5,
                    gold_cited=True,
                ),
            ]
        },
        {
            "cases": [
                _case(
                    question_id="train-q1",
                    title="Can I hide left menu pane?",
                    route="other",
                    f1=0.2,
                    gold_cited=True,
                ),
                _case(
                    question_id="train-q2",
                    title="Can I change the default pane?",
                    route="other",
                    f1=0.3,
                    gold_cited=True,
                ),
                _case(
                    question_id="train-q3",
                    title="Can I migrate search templates?",
                    route="other",
                    f1=0.4,
                    gold_cited=True,
                ),
            ]
        },
    ]
    challenger_reports = [
        {
            "cases": [
                _case(
                    question_id="dev-q1",
                    title="Can I hide left menu pane?",
                    route="other",
                    f1=0.3,
                    gold_cited=True,
                ),
                _case(
                    question_id="dev-q2",
                    title="Can I change the default pane?",
                    route="other",
                    f1=0.5,
                    gold_cited=True,
                ),
                _case(
                    question_id="dev-q3",
                    title="Can I migrate search templates?",
                    route="other",
                    f1=0.4,
                    gold_cited=True,
                ),
            ]
        },
        {
            "cases": [
                _case(
                    question_id="train-q1",
                    title="Can I hide left menu pane?",
                    route="other",
                    f1=0.6,
                    gold_cited=True,
                ),
                _case(
                    question_id="train-q2",
                    title="Can I change the default pane?",
                    route="other",
                    f1=0.7,
                    gold_cited=True,
                ),
                _case(
                    question_id="train-q3",
                    title="Can I migrate search templates?",
                    route="other",
                    f1=0.8,
                    gold_cited=True,
                ),
            ]
        },
    ]

    analysis = analyze_other_route_window_outcomes(
        baseline_reports=baseline_reports,
        challenger_reports=challenger_reports,
        source_labels=["dev", "train"],
        min_cases=3,
    )
    result_dict = other_route_window_outcome_analysis_to_dict(analysis)

    assert analysis.total_cases == 6
    assert analysis.stable_answer_window_subtypes == []
    assert analysis.mixed_subtypes == ["procedure_or_change"]
    assert result_dict["recommendation_note"].startswith("Heuristic only")


def test_other_route_window_outcome_analysis_detects_stable_candidate():
    baseline_reports = [
        {
            "cases": [
                _case("dev-q1", "Support guide for probe A", "other", 0.1, False),
                _case("dev-q2", "Support guide for probe B", "other", 0.2, True),
                _case("dev-q3", "Support guide for probe C", "other", 0.2, True),
            ]
        },
        {
            "cases": [
                _case("train-q1", "Support guide for probe A", "other", 0.1, False),
                _case("train-q2", "Support guide for probe B", "other", 0.2, True),
                _case("train-q3", "Support guide for probe C", "other", 0.2, True),
            ]
        },
    ]
    challenger_reports = [
        {
            "cases": [
                _case("dev-q1", "Support guide for probe A", "other", 0.6, True),
                _case("dev-q2", "Support guide for probe B", "other", 0.5, True),
                _case("dev-q3", "Support guide for probe C", "other", 0.5, True),
            ]
        },
        {
            "cases": [
                _case("train-q1", "Support guide for probe A", "other", 0.6, True),
                _case("train-q2", "Support guide for probe B", "other", 0.5, True),
                _case("train-q3", "Support guide for probe C", "other", 0.5, True),
            ]
        },
    ]

    analysis = analyze_other_route_window_outcomes(
        baseline_reports=baseline_reports,
        challenger_reports=challenger_reports,
        source_labels=["dev", "train"],
        min_cases=3,
    )

    assert analysis.stable_answer_window_subtypes == ["support_or_download"]
    assert analysis.overall_subtype_summaries[0].recommendation == (
        "candidate_answer_window"
    )


def _case(
    question_id: str,
    title: str,
    route: str,
    f1: float,
    gold_cited: bool,
) -> dict:
    return {
        "question_id": question_id,
        "question_title": title,
        "question_text": "",
        "question_route": route,
        "selected_answer_token_f1": f1,
        "selected_gold_candidate_count": 1 if gold_cited else 0,
    }
