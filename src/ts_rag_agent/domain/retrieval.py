from dataclasses import dataclass

from ts_rag_agent.domain.dataset import PrimeQADocument


@dataclass(frozen=True)
class RetrievalResult:
    """单条检索结果。"""

    document: PrimeQADocument
    score: float
    rank: int


@dataclass(frozen=True)
class RetrievalMetrics:
    """检索评估指标。"""

    total_questions: int
    evaluated_questions: int
    hit_at_k: dict[int, float]
    mrr: float
