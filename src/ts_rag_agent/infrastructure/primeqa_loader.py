from __future__ import annotations

import json
from pathlib import Path

from ts_rag_agent.domain.dataset import (
    PrimeQADocument,
    PrimeQADocumentSection,
    PrimeQAQuestion,
    PrimeQAStats,
)


def load_primeqa_questions(questions_json_path: Path) -> list[PrimeQAQuestion]:
    """读取 PrimeQA 训练集或开发集问答数据并转换为类型化对象。"""

    rows = json.loads(questions_json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of question rows: {questions_json_path}")

    return [_question_from_row(row) for row in rows]


def load_primeqa_documents(documents_json_path: Path) -> dict[str, PrimeQADocument]:
    """从清洗后的 sections JSON 文件中读取 PrimeQA 技术文档。"""

    rows = json.loads(documents_json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, dict):
        raise ValueError(f"Expected a document-id keyed object: {documents_json_path}")

    documents = {}
    for doc_id, row in rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"Expected document row object for id {doc_id}")

        document = PrimeQADocument(
            id=str(row.get("id") or doc_id),
            title=str(row.get("title") or ""),
            text=str(row.get("text") or ""),
        )
        documents[document.id] = document

    return documents


def load_primeqa_document_sections(
    documents_json_path: Path,
) -> dict[str, list[PrimeQADocumentSection]]:
    """从清洗后的 sections JSON 文件中读取每篇文档的 section。"""

    rows = json.loads(documents_json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, dict):
        raise ValueError(f"Expected a document-id keyed object: {documents_json_path}")

    sections_by_document = {}
    for doc_id, row in rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"Expected document row object for id {doc_id}")

        sections_by_document[str(doc_id)] = [
            _section_from_row(document_id=str(doc_id), row=section)
            for section in row.get("sections", [])
        ]

    return sections_by_document


def compute_primeqa_stats(
    questions: list[PrimeQAQuestion],
    documents: dict[str, PrimeQADocument],
) -> PrimeQAStats:
    """统计问题、文档以及引用覆盖情况。"""

    document_ids = set(documents)
    candidate_doc_ids = {doc_id for question in questions for doc_id in question.doc_ids}
    answer_doc_ids = {
        question.answer_doc_id
        for question in questions
        if question.answerable and question.answer_doc_id
    }
    candidate_counts = [len(question.doc_ids) for question in questions]
    answerable_questions = sum(1 for question in questions if question.answerable)

    return PrimeQAStats(
        total_questions=len(questions),
        answerable_questions=answerable_questions,
        unanswerable_questions=len(questions) - answerable_questions,
        total_documents=len(documents),
        unique_candidate_doc_ids=len(candidate_doc_ids),
        missing_candidate_doc_ids=len(candidate_doc_ids - document_ids),
        missing_answer_doc_ids=len(answer_doc_ids - document_ids),
        avg_candidate_doc_ids=round(sum(candidate_counts) / len(candidate_counts), 3)
        if candidate_counts
        else 0.0,
    )


def _question_from_row(row: dict) -> PrimeQAQuestion:
    answerable = _parse_answerable(row["ANSWERABLE"])

    return PrimeQAQuestion(
        id=str(row["QUESTION_ID"]),
        title=str(row.get("QUESTION_TITLE") or ""),
        text=str(row.get("QUESTION_TEXT") or ""),
        answer=str(row.get("ANSWER") or ""),
        answerable=answerable,
        answer_doc_id=str(row["DOCUMENT"]) if row.get("DOCUMENT") else None,
        doc_ids=[str(doc_id) for doc_id in row.get("DOC_IDS", [])],
        start_offset=_to_optional_int(row.get("START_OFFSET")),
        end_offset=_to_optional_int(row.get("END_OFFSET")),
    )


def _section_from_row(document_id: str, row: dict) -> PrimeQADocumentSection:
    return PrimeQADocumentSection(
        document_id=document_id,
        section_id=str(row.get("id") or ""),
        text=str(row.get("text") or ""),
        start_offset=_to_optional_int(row.get("start")),
        end_offset=_to_optional_int(row.get("end")),
    )


def _parse_answerable(value: str) -> bool:
    normalized = str(value).strip().upper()
    if normalized == "Y":
        return True
    if normalized == "N":
        return False

    raise ValueError(f"Unsupported PrimeQA ANSWERABLE value: {value!r}")


def _to_optional_int(value: object) -> int | None:
    if value in (None, "", "-"):
        return None
    return int(value)
