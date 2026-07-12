from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerCitation:
    """答案引用的证据文档。"""

    document_id: str
    title: str
    retrieval_rank: int
    evidence_score: float


@dataclass(frozen=True)
class GeneratedAnswer:
    """一次 RAG 回答结果。"""

    question_id: str
    answer: str
    citations: list[AnswerCitation]
    refused: bool


@dataclass(frozen=True)
class AnswerVerificationResult:
    """答案验证结果。"""

    original_answer: GeneratedAnswer
    verified_answer: GeneratedAnswer
    citation_context_valid: bool
    reasons: list[str]


@dataclass(frozen=True)
class AnswerEvaluationMetrics:
    """RAG 回答层评估指标。"""

    total_questions: int
    answerable_questions: int
    unanswerable_questions: int
    generated_answerable_questions: int
    refused_answerable_questions: int
    refused_unanswerable_questions: int
    gold_doc_citation_rate: float
    answerable_refusal_rate: float
    unanswerable_refusal_rate: float
    average_token_f1: float
