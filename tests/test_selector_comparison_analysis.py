from ts_rag_agent.application.selector_comparison_analysis import (
    classify_question_type,
    compare_selector_reports,
)


def test_classify_question_type_detects_security_bulletin():
    question_type = classify_question_type(
        question_title="Security Bulletin for Java SDK CVE-2020-1234",
        question_text="",
        gold_answer="CVSS Base Score: 5.9",
    )

    assert question_type == "security_bulletin"


def test_compare_selector_reports_counts_winners_and_types():
    baseline_report = {
        "cases": [
            _case(
                question_id="q1",
                title="Security Bulletin CVE-2020-1234",
                f1=0.2,
                gold_cited=True,
                answer="CVEID: CVE-2020-1234 CVSS Base Score: 5.",
            ),
            _case(
                question_id="q2",
                title="How do I configure service A?",
                f1=0.6,
                gold_cited=True,
                answer="Configure service A from the admin panel.",
            ),
            _case(
                question_id="q3",
                title="Known limitation for product B",
                f1=0.4,
                gold_cited=False,
                answer="This is a known current limitation.",
            ),
        ]
    }
    challenger_report = {
        "cases": [
            _case(
                question_id="q1",
                title="Security Bulletin CVE-2020-1234",
                f1=0.5,
                gold_cited=True,
                answer="CVEID: CVE-2020-1234 CVSS Base Score: 5.",
            ),
            _case(
                question_id="q2",
                title="How do I configure service A?",
                f1=0.56,
                gold_cited=True,
                answer="Configure service A from the admin panel.",
            ),
            _case(
                question_id="q3",
                title="Known limitation for product B",
                f1=0.41,
                gold_cited=True,
                answer="This is a known current limitation.",
            ),
        ]
    }

    result = compare_selector_reports(
        baseline_report=baseline_report,
        challenger_report=challenger_report,
        baseline_label="answer-aware",
        challenger_label="section-span",
        f1_win_margin=0.03,
    )

    assert result.summary.total_compared == 3
    assert result.summary.challenger_wins == 2
    assert result.summary.baseline_wins == 1
    assert result.summary.question_type_counts["security_bulletin"] == 1
    assert result.summary.challenger_question_route_counts["security_bulletin"] == 1
    assert result.summary.challenger_selected_selector_counts[
        "section_span_bm25_sentence"
    ] == 1
    assert result.challenger_win_cases[0].question_id == "q1"
    assert result.challenger_win_cases[0].challenger_question_route == "security_bulletin"


def _case(
    question_id: str,
    title: str,
    f1: float,
    gold_cited: bool,
    answer: str,
) -> dict:
    return {
        "question_id": question_id,
        "question_title": title,
        "question_text": "",
        "gold_answer": answer,
        "bucket": "gold_window_beats_selected_answer",
        "selected_answer_token_f1": f1,
        "selected_gold_candidate_count": 1 if gold_cited else 0,
        "question_route": classify_route_from_title(title),
        "selected_selector_name": (
            "section_span_bm25_sentence"
            if "Security Bulletin" in title
            else "answer_aware_bm25_sentence"
        ),
        "selected_candidates": [
            {
                "sentence": answer,
            }
        ],
    }


def classify_route_from_title(title: str) -> str:
    if "Security Bulletin" in title:
        return "security_bulletin"
    if "limitation" in title:
        return "limitation_or_restriction"
    return "other"
