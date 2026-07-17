from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class EvidenceContext(BaseModel):
    """检索得到或数据集中标注的证据文档片段。"""

    filename: str
    text: str


class TechQASample(BaseModel):
    """一条技术支持问答样本。"""

    id: str
    question: str
    answer: str
    is_impossible: bool
    contexts: list[EvidenceContext] = Field(default_factory=list)

    @property
    def is_answerable(self) -> bool:
        return not self.is_impossible


class DatasetStats(BaseModel):
    """用于验证报告的数据集统计信息。"""

    total_rows: int
    answerable_rows: int
    impossible_rows: int
    unique_referenced_files: int
    missing_referenced_files: int
    corpus_files: int
    min_contexts: int
    max_contexts: int
    avg_contexts: float


class PrimeQAQuestion(BaseModel):
    """一条 PrimeQA TechQA 问答样本。"""

    id: str
    title: str
    text: str
    answer: str
    answerable: bool
    answer_doc_id: str | None
    doc_ids: list[str] = Field(default_factory=list)
    start_offset: int | None = None
    end_offset: int | None = None

    @property
    def full_question(self) -> str:
        parts = [self.title.strip(), self.text.strip()]
        return "\n\n".join(part for part in parts if part)


class PrimeQAQuery(Protocol):
    """Label-free structural contract consumed by the online RAG path."""

    id: str
    title: str
    text: str

    @property
    def full_question(self) -> str: ...


class PrimeQARuntimeQuery(BaseModel):
    """Serving query carrying no gold answer, label, or document membership."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str = ""
    text: str

    @property
    def full_question(self) -> str:
        parts = [self.title.strip(), self.text.strip()]
        return "\n\n".join(part for part in parts if part)


class PrimeQADocument(BaseModel):
    """一篇用于检索的 PrimeQA TechQA 技术支持文档。"""

    id: str
    title: str
    text: str


class PrimeQADocumentSection(BaseModel):
    """一篇 PrimeQA 技术文档中的一个 section。"""

    document_id: str
    section_id: str
    text: str
    start_offset: int | None = None
    end_offset: int | None = None


class PrimeQAStats(BaseModel):
    """检索实验前需要确认的 PrimeQA 数据统计信息。"""

    total_questions: int
    answerable_questions: int
    unanswerable_questions: int
    total_documents: int
    unique_candidate_doc_ids: int
    missing_candidate_doc_ids: int
    missing_answer_doc_ids: int
    avg_candidate_doc_ids: float
