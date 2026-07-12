from pydantic import BaseModel, Field


class EvidenceContext(BaseModel):
    """A retrieved or labeled support document context."""

    filename: str
    text: str


class TechQASample(BaseModel):
    """One technical support question-answer example."""

    id: str
    question: str
    answer: str
    is_impossible: bool
    contexts: list[EvidenceContext] = Field(default_factory=list)

    @property
    def is_answerable(self) -> bool:
        return not self.is_impossible


class DatasetStats(BaseModel):
    """High-level dataset statistics used in verification reports."""

    total_rows: int
    answerable_rows: int
    impossible_rows: int
    unique_referenced_files: int
    missing_referenced_files: int
    corpus_files: int
    min_contexts: int
    max_contexts: int
    avg_contexts: float
