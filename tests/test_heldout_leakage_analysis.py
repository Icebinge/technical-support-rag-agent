import pytest

from ts_rag_agent.application.heldout_leakage_analysis import (
    LeakageQuestion,
    analyze_heldout_leakage,
    normalize_question_text,
    write_leakage_visualizations,
)


def test_normalize_question_text_keeps_ascii_tokens_only():
    assert normalize_question_text(" How do I fix SSL-Key error? ") == (
        "how do i fix ssl key error"
    )


def test_analyze_heldout_leakage_blocks_exact_and_near_overlaps():
    report = analyze_heldout_leakage(
        heldout_questions=[
            _question("heldout", "h1", "How do I fix SSL key error?"),
            _question("heldout", "h2", "How can I configure portal database transfer"),
            _question("heldout", "h3", "A completely new heldout question"),
        ],
        development_questions=[
            _question("development", "d1", "How do I fix SSL key error"),
            _question("development", "d2", "How do I configure portal database transfer"),
        ],
        near_duplicate_threshold=0.7,
        sample_limit=10,
    )

    assert report["counts"]["heldout_questions"] == 3
    assert report["counts"]["exact_overlap_count"] == 1
    assert report["counts"]["exact_overlap_pair_count"] == 1
    assert report["counts"]["near_duplicate_overlap_count"] == 1
    assert report["counts"]["near_duplicate_overlap_pair_count"] == 1
    assert report["counts"]["unhandled_overlap_count"] == 2
    assert report["counts"]["heldout_questions_without_detected_overlap"] == 1
    assert report["heldout_usable_without_exclusions"] is False
    assert report["decision"].startswith("blocked:")
    assert report["exact_overlap_samples"][0]["heldout_question_id"] == "h1"
    assert report["near_duplicate_overlap_samples"][0]["heldout_question_id"] == "h2"


def test_analyze_heldout_leakage_separates_question_count_from_pair_count():
    report = analyze_heldout_leakage(
        heldout_questions=[_question("heldout", "h1", "How do I fix SSL key error?")],
        development_questions=[
            _question("development", "d1", "How do I fix SSL key error"),
            _question("development", "d2", "How do I fix SSL key error"),
        ],
        sample_limit=10,
    )

    assert report["counts"]["exact_overlap_count"] == 1
    assert report["counts"]["exact_overlap_pair_count"] == 2
    assert report["counts"]["unhandled_overlap_count"] == 1
    assert report["counts"]["heldout_questions_without_detected_overlap"] == 0
    assert len(report["exact_overlap_samples"]) == 2


def test_analyze_heldout_leakage_passes_without_detected_overlap():
    report = analyze_heldout_leakage(
        heldout_questions=[_question("heldout", "h1", "A new heldout question")],
        development_questions=[_question("development", "d1", "Existing train case")],
        near_duplicate_threshold=0.9,
    )

    assert report["heldout_usable_without_exclusions"] is True
    assert report["decision"].startswith("passed:")


def test_analyze_heldout_leakage_validates_inputs():
    with pytest.raises(ValueError, match="heldout_questions must not be empty"):
        analyze_heldout_leakage(
            heldout_questions=[],
            development_questions=[_question("development", "d1", "Existing")],
        )

    with pytest.raises(ValueError, match="near_duplicate_threshold"):
        analyze_heldout_leakage(
            heldout_questions=[_question("heldout", "h1", "New")],
            development_questions=[_question("development", "d1", "Existing")],
            near_duplicate_threshold=0,
        )


def test_write_leakage_visualizations(tmp_path):
    report = analyze_heldout_leakage(
        heldout_questions=[_question("heldout", "h1", "How do I fix SSL key error?")],
        development_questions=[_question("development", "d1", "How do I fix SSL key error")],
    )

    artifacts = write_leakage_visualizations(report, tmp_path)

    assert {artifact.name for artifact in artifacts} == {
        "stage53_heldout_overlap_counts.svg",
        "stage53_development_source_counts.svg",
    }
    for artifact in artifacts:
        assert (tmp_path / artifact.name).read_text(encoding="utf-8").startswith("<svg")


def _question(source: str, question_id: str, text: str) -> LeakageQuestion:
    return LeakageQuestion(
        source=source,
        split="test",
        question_id=question_id,
        question_text=text,
    )
