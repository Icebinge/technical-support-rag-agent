from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    CandidateScoringPolicy,
    PrimaryContextSelectionPolicy,
)
from ts_rag_agent.domain.dataset import PrimeQAQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

LISTWISE_CHECKPOINT_FORMAT_ID = "primeqa_hybrid_listwise_reranker_checkpoint_v1"
LISTWISE_RUNTIME_POLICY_ID = "query_overlap_rrf_union_listwise_top10_v1"
LISTWISE_TRAINING_FAMILY = "listwise_none"
LISTWISE_CONTEXT_DEPTH = 10


@dataclass(frozen=True)
class ListwiseScoringCounters:
    call_count: int
    pair_count: int
    total_seconds: float
    maximum_call_seconds: float

    @property
    def mean_call_seconds(self) -> float:
        return self.total_seconds / self.call_count if self.call_count else 0.0


class ListwiseScoreProvider(Protocol):
    def score(
        self,
        *,
        question: PrimeQAQuery,
        candidates: Sequence[RetrievalResult],
    ) -> Mapping[str, float]: ...

    def counters(self) -> ListwiseScoringCounters: ...


class _MeasuredScoreProvider:
    def __init__(self) -> None:
        self._counter_lock = Lock()
        self._call_count = 0
        self._pair_count = 0
        self._total_seconds = 0.0
        self._maximum_call_seconds = 0.0

    def counters(self) -> ListwiseScoringCounters:
        with self._counter_lock:
            return ListwiseScoringCounters(
                call_count=self._call_count,
                pair_count=self._pair_count,
                total_seconds=round(self._total_seconds, 6),
                maximum_call_seconds=round(self._maximum_call_seconds, 6),
            )

    def _record(self, *, pair_count: int, seconds: float) -> None:
        with self._counter_lock:
            self._call_count += 1
            self._pair_count += pair_count
            self._total_seconds += seconds
            self._maximum_call_seconds = max(self._maximum_call_seconds, seconds)


class PrecomputedListwiseScoreProvider(_MeasuredScoreProvider):
    """Serve grouped OOF logits without loading a model during quality evaluation."""

    def __init__(self, scores: Mapping[str, float]) -> None:
        super().__init__()
        self._scores = dict(scores)

    def score(
        self,
        *,
        question: PrimeQAQuery,
        candidates: Sequence[RetrievalResult],
    ) -> Mapping[str, float]:
        started_at = time.perf_counter()
        result = {
            candidate.document.id: self._scores[
                stage173._pair_identity(question.id, candidate.document.id)
            ]
            for candidate in candidates
        }
        self._record(pair_count=len(candidates), seconds=time.perf_counter() - started_at)
        return result


class LocalListwiseCheckpointScoreProvider(_MeasuredScoreProvider):
    """Load one authenticated full-train checkpoint for optional CPU runtime use."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("listwise runtime batch and length limits must be positive")
        self._checkpoint_path = checkpoint_path
        self._manifest = load_and_validate_listwise_checkpoint_manifest(checkpoint_path)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = __import__("torch")
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_path),
            local_files_only=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_path),
            local_files_only=True,
        ).to(device)
        self._model.eval()
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._text_policy = stage173.QueryAwareCrossEncoderTextPolicy()

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    def score(
        self,
        *,
        question: PrimeQAQuery,
        candidates: Sequence[RetrievalResult],
    ) -> Mapping[str, float]:
        if not candidates:
            raise ValueError("listwise runtime scoring requires candidates")
        started_at = time.perf_counter()
        scores: dict[str, float] = {}
        with self._torch.inference_mode():
            for start in range(0, len(candidates), self._batch_size):
                batch = candidates[start : start + self._batch_size]
                passages = [
                    self._text_policy.passage(
                        question=question.full_question,
                        document=result.document,
                    )
                    for result in batch
                ]
                encoded = self._tokenizer(
                    [question.full_question] * len(batch),
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                encoded = {name: value.to(self._device) for name, value in encoded.items()}
                logits = self._model(**encoded).logits.reshape(-1).detach().cpu().tolist()
                scores.update(
                    {
                        result.document.id: float(logit)
                        for result, logit in zip(batch, logits, strict=True)
                    }
                )
        if len(scores) != len(candidates):
            raise RuntimeError("listwise runtime score coverage is incomplete")
        self._record(pair_count=len(candidates), seconds=time.perf_counter() - started_at)
        return scores


class ListwiseUnionPrimaryContextSelectionPolicy(PrimaryContextSelectionPolicy):
    """Rerank the union of query-overlap Top10 and original-RRF Top10."""

    def __init__(self, *, score_provider: ListwiseScoreProvider) -> None:
        self._score_provider = score_provider

    def select(
        self,
        *,
        question: PrimeQAQuery,
        query_terms: set[str],
        candidates: Sequence[RetrievalResult],
        scoring_policy: CandidateScoringPolicy,
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        if top_k != LISTWISE_CONTEXT_DEPTH:
            raise ValueError("listwise runtime context depth must remain 10")
        original_rrf = sorted(
            candidates,
            key=lambda result: (result.rank, result.document.id),
        )[:top_k]
        query_overlap = scoring_policy.rank(
            query_terms=query_terms,
            candidates=candidates,
        )[:top_k]
        union: dict[str, RetrievalResult] = {}
        for result in (*query_overlap, *original_rrf):
            union.setdefault(result.document.id, result)
        scores = self._score_provider.score(
            question=question,
            candidates=tuple(union.values()),
        )
        return sorted(
            union.values(),
            key=lambda result: (
                -scores[result.document.id],
                result.rank,
                result.document.id,
            ),
        )[:top_k]


def write_listwise_checkpoint_manifest(
    *,
    checkpoint_path: Path,
    stage177_report_sha256: str,
    train_source_sha256: str,
    training_row_count: int,
    training_pair_count: int,
    optimizer_step_count: int,
    first_epoch_mean_loss: float,
    final_epoch_mean_loss: float,
) -> dict[str, Any]:
    manifest_path = checkpoint_path / "stage178_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"listwise checkpoint manifest already exists: {manifest_path}")
    files = {
        path.name: _sha256_file(path)
        for path in sorted(checkpoint_path.iterdir())
        if path.is_file()
    }
    manifest = {
        "format_id": LISTWISE_CHECKPOINT_FORMAT_ID,
        "training_family": LISTWISE_TRAINING_FAMILY,
        "runtime_policy_id": LISTWISE_RUNTIME_POLICY_ID,
        "stage177_report_sha256": stage177_report_sha256,
        "train_source_sha256": train_source_sha256,
        "training_row_count": training_row_count,
        "training_pair_count": training_pair_count,
        "optimizer_step_count": optimizer_step_count,
        "first_epoch_mean_loss": first_epoch_mean_loss,
        "final_epoch_mean_loss": final_epoch_mean_loss,
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_and_validate_listwise_checkpoint_manifest(
    checkpoint_path: Path,
) -> dict[str, Any]:
    manifest_path = checkpoint_path / "stage178_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("listwise checkpoint manifest must be a JSON object")
    if manifest.get("format_id") != LISTWISE_CHECKPOINT_FORMAT_ID:
        raise ValueError("listwise checkpoint format is not authorized")
    if manifest.get("training_family") != LISTWISE_TRAINING_FAMILY:
        raise ValueError("listwise checkpoint training family is not authorized")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("listwise checkpoint manifest has no file fingerprints")
    for filename, expected in files.items():
        path = checkpoint_path / str(filename)
        if _sha256_file(path) != expected:
            raise ValueError(f"listwise checkpoint file hash mismatch: {filename}")
    return manifest


def checkpoint_public_summary(checkpoint_path: Path) -> dict[str, Any]:
    manifest = load_and_validate_listwise_checkpoint_manifest(checkpoint_path)
    manifest_sha256 = _sha256_file(checkpoint_path / "stage178_manifest.json")
    total_bytes = sum(path.stat().st_size for path in checkpoint_path.iterdir() if path.is_file())
    return {
        **{key: value for key, value in manifest.items() if key != "files"},
        "file_count": len(manifest["files"]) + 1,
        "total_bytes": total_bytes,
        "manifest_sha256": manifest_sha256,
        "all_file_hashes_valid": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def listwise_runtime_policy_contract() -> dict[str, Any]:
    return {
        "policy_id": LISTWISE_RUNTIME_POLICY_ID,
        "candidate_source": "query_overlap_top10_union_original_rrf_top10",
        "maximum_rerank_candidates": 20,
        "generation_context_depth": LISTWISE_CONTEXT_DEPTH,
        "checkpoint_format_id": LISTWISE_CHECKPOINT_FORMAT_ID,
        "training_family": LISTWISE_TRAINING_FAMILY,
        "runtime_device_for_qwen_colocation": "cpu",
        "evidence_sufficiency_gate_enabled": False,
        "fallback_strategy_enabled": False,
        "runtime_registered_as_default": False,
    }
