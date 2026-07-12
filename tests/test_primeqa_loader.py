import json

import pytest

from ts_rag_agent.infrastructure.primeqa_loader import (
    compute_primeqa_stats,
    load_primeqa_documents,
    load_primeqa_questions,
)


def test_load_primeqa_questions_and_documents(tmp_path):
    questions_json = tmp_path / "training_Q_A.json"
    documents_json = tmp_path / "training_dev_technotes.sections.json"

    questions_json.write_text(
        json.dumps(
            [
                {
                    "QUESTION_ID": "TRAIN_Q000",
                    "QUESTION_TITLE": "How do I restart service A?",
                    "QUESTION_TEXT": "The service stopped after an upgrade.",
                    "DOCUMENT": "doc-a",
                    "ANSWER": "Restart service A from the control panel.",
                    "START_OFFSET": "10",
                    "END_OFFSET": "55",
                    "ANSWERABLE": "Y",
                    "DOC_IDS": ["doc-a", "doc-b"],
                },
                {
                    "QUESTION_ID": "TRAIN_Q001",
                    "QUESTION_TITLE": "What is the admin password?",
                    "QUESTION_TEXT": "",
                    "DOCUMENT": "",
                    "ANSWER": "",
                    "START_OFFSET": "-",
                    "END_OFFSET": "-",
                    "ANSWERABLE": "N",
                    "DOC_IDS": ["missing-doc"],
                },
            ]
        ),
        encoding="utf-8",
    )
    documents_json.write_text(
        json.dumps(
            {
                "doc-a": {
                    "id": "doc-a",
                    "title": "Restart service A",
                    "text": "Use the control panel to restart service A.",
                    "sections": [],
                },
                "doc-b": {
                    "id": "doc-b",
                    "title": "Service A upgrade notes",
                    "text": "Service A may stop after some upgrades.",
                    "sections": [],
                },
            }
        ),
        encoding="utf-8",
    )

    questions = load_primeqa_questions(questions_json)
    documents = load_primeqa_documents(documents_json)
    stats = compute_primeqa_stats(questions, documents)

    assert len(questions) == 2
    assert questions[0].id == "TRAIN_Q000"
    assert questions[0].answerable is True
    assert questions[0].answer_doc_id == "doc-a"
    assert questions[0].start_offset == 10
    assert questions[1].start_offset is None
    assert questions[1].end_offset is None
    assert "How do I restart service A?" in questions[0].full_question
    assert len(documents) == 2
    assert documents["doc-a"].text.startswith("Use the control panel")
    assert stats.answerable_questions == 1
    assert stats.unanswerable_questions == 1
    assert stats.missing_candidate_doc_ids == 1
    assert stats.missing_answer_doc_ids == 0


def test_rejects_unknown_answerable_value(tmp_path):
    questions_json = tmp_path / "bad_Q_A.json"
    questions_json.write_text(
        json.dumps(
            [
                {
                    "QUESTION_ID": "TRAIN_Q999",
                    "QUESTION_TITLE": "Bad row",
                    "QUESTION_TEXT": "",
                    "DOCUMENT": "",
                    "ANSWER": "",
                    "START_OFFSET": "",
                    "END_OFFSET": "",
                    "ANSWERABLE": "MAYBE",
                    "DOC_IDS": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ANSWERABLE"):
        load_primeqa_questions(questions_json)
