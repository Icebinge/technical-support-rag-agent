from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.retrieval_evaluation import evaluate_retrieval
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 70"
_CREATED_AT = "2026-07-14"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_RUNTIME_FORBIDDEN_GOLD_KEYS = frozenset(
    {
        "answer",
        "candidate_token_f1",
        "is_gold_document",
        "is_best_candidate_for_question",
        "best_candidate_token_f1_for_question",
        "f1_gap_to_best_candidate",
    }
)


@dataclass(frozen=True)
class PrimeQAHybridDevelopmentChecksVisualization:
    """One generated Stage70 visualization."""

    name: str
    path: str


def run_primeqa_hybrid_development_checks(
    *,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    candidate_dataset_path: Path,
    candidate_summary_path: Path,
    top_k_values: tuple[int, ...] = (1, 5, 10),
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Run train/dev-only baseline and candidate checks for the frozen split."""

    _validate_options(top_k_values=top_k_values, bm25_k1=bm25_k1, bm25_b=bm25_b)
    started_at = time.perf_counter()
    split_samples = {
        "train": load_primeqa_hybrid_split_samples(train_split_path),
        "dev": load_primeqa_hybrid_split_samples(dev_split_path),
    }
    loaded_splits_at = time.perf_counter()
    documents = load_primeqa_documents(documents_path)
    loaded_documents_at = time.perf_counter()
    retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    retriever.fit(documents.values())
    indexed_at = time.perf_counter()
    retrieval_baseline = _retrieval_baseline(
        split_samples=split_samples,
        retriever=retriever,
        top_k_values=top_k_values,
    )
    evaluated_at = time.perf_counter()
    candidate_scan = _scan_candidate_dataset(candidate_dataset_path)
    candidate_summary = _load_candidate_summary(candidate_summary_path)
    candidate_checks = _candidate_checks(
        candidate_scan=candidate_scan,
        candidate_summary=candidate_summary,
    )
    guard_checks = _guard_checks(
        split_samples=split_samples,
        candidate_checks=candidate_checks,
        candidate_scan=candidate_scan,
        candidate_summary=candidate_summary,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only development checks for primeqa_hybrid_stage68_v1. "
            "This stage runs BM25 retrieval baselines on train/dev, audits the "
            "Stage69 train/dev candidate artifact, keeps test locked, does not "
            "run final metrics, and does not change the default runtime."
        ),
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "documents": _fingerprint(documents_path),
            "candidate_dataset": _fingerprint(candidate_dataset_path),
            "candidate_summary": _fingerprint(candidate_summary_path),
        },
        "loaded_split_summary": summarize_primeqa_hybrid_split_samples(split_samples),
        "bm25_baseline": {
            "config": {
                "top_k_values": list(top_k_values),
                "k1": bm25_k1,
                "b": bm25_b,
            },
            "metrics_by_split": retrieval_baseline,
        },
        "candidate_artifact_checks": {
            "scan": candidate_scan,
            "summary": _public_candidate_summary(candidate_summary),
            "checks": candidate_checks,
        },
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents": round(loaded_documents_at - loaded_splits_at, 3),
            "bm25_index": round(indexed_at - loaded_documents_at, 3),
            "bm25_evaluate": round(evaluated_at - indexed_at, 3),
            "candidate_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_development_check_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridDevelopmentChecksVisualization]:
    """Write SVG charts for Stage70 development checks."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage70_primeqa_bm25_hit_at_k.svg": render_horizontal_bar_chart_svg(
            title="Stage70 BM25 train/dev hit@k",
            bars=_bm25_hit_bars(report),
            x_label="hit rate",
            margin_left=220,
        ),
        "stage70_primeqa_bm25_mrr.svg": render_horizontal_bar_chart_svg(
            title="Stage70 BM25 train/dev MRR",
            bars=_bm25_mrr_bars(report),
            x_label="MRR",
            margin_left=180,
        ),
        "stage70_primeqa_candidate_rows_by_split.svg": render_horizontal_bar_chart_svg(
            title="Stage70 candidate rows by split",
            bars=_candidate_rows_by_split_bars(report),
            x_label="candidate row count",
            margin_left=260,
        ),
        "stage70_primeqa_candidate_questions_by_split.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage70 candidate questions by split",
                bars=_candidate_questions_by_split_bars(report),
                x_label="question count",
                margin_left=260,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridDevelopmentChecksVisualization(name=filename, path=str(path))
        )
    return artifacts


def _retrieval_baseline(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    retriever: BM25Retriever,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    metrics_by_split = {}
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        questions = [sample.to_primeqa_question() for sample in split_samples[split]]
        metrics = evaluate_retrieval(
            questions=questions,
            retriever=retriever,
            top_k_values=top_k_values,
        )
        metrics_by_split[split] = {
            "total_questions": metrics.total_questions,
            "evaluated_questions": metrics.evaluated_questions,
            "hit_at_k": {f"hit@{key}": value for key, value in metrics.hit_at_k.items()},
            "mrr": metrics.mrr,
        }
    return metrics_by_split


def _scan_candidate_dataset(candidate_dataset_path: Path) -> dict[str, Any]:
    _ensure_file(candidate_dataset_path)
    rows_by_split: Counter[str] = Counter()
    rows_with_runtime_features = 0
    rows_with_gold_labels = 0
    rows_with_forbidden_runtime_keys = 0
    rows_with_test_split = 0
    question_ids_by_split: dict[str, set[str]] = {}
    total_rows = 0
    with candidate_dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object on line {line_number} in {candidate_dataset_path}"
                )
            total_rows += 1
            split = str(row.get("split") or "")
            question_id = str(row.get("question_id") or "")
            runtime_features = row.get("runtime_features") or {}
            gold_labels = row.get("gold_labels") or {}
            if not isinstance(runtime_features, dict):
                raise ValueError(f"runtime_features must be an object on line {line_number}")
            if not isinstance(gold_labels, dict):
                raise ValueError(f"gold_labels must be an object on line {line_number}")
            rows_by_split[split] += 1
            question_ids_by_split.setdefault(split, set()).add(question_id)
            rows_with_runtime_features += bool(runtime_features)
            rows_with_gold_labels += bool(gold_labels)
            rows_with_test_split += split in _FORBIDDEN_FINAL_SPLITS
            rows_with_forbidden_runtime_keys += bool(
                set(runtime_features).intersection(_RUNTIME_FORBIDDEN_GOLD_KEYS)
            )
    return {
        "row_count": total_rows,
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "question_count_by_split": {
            split: len(question_ids)
            for split, question_ids in sorted(question_ids_by_split.items())
        },
        "rows_with_runtime_features": rows_with_runtime_features,
        "rows_with_gold_labels": rows_with_gold_labels,
        "rows_with_test_split": rows_with_test_split,
        "rows_with_forbidden_runtime_gold_keys": rows_with_forbidden_runtime_keys,
    }


def _load_candidate_summary(candidate_summary_path: Path) -> dict[str, Any]:
    _ensure_file(candidate_summary_path)
    summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Expected object summary in {candidate_summary_path}")
    artifact_summary = summary.get("candidate_artifact_summary")
    if not isinstance(artifact_summary, dict):
        raise ValueError("candidate_artifact_summary is missing")
    public_summary = artifact_summary.get("summary")
    if not isinstance(public_summary, dict):
        raise ValueError("candidate_artifact_summary.summary is missing")
    return summary


def _candidate_checks(
    *,
    candidate_scan: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summary = candidate_summary["candidate_artifact_summary"]["summary"]
    expected_rows_by_split = {
        str(split): int(count) for split, count in summary["rows_by_split"].items()
    }
    observed_rows_by_split = {
        str(split): int(count) for split, count in candidate_scan["rows_by_split"].items()
    }
    expected_questions_by_split = {
        str(split): int(count)
        for split, count in summary["questions_by_split"].items()
    }
    observed_questions_by_split = {
        str(split): int(count)
        for split, count in candidate_scan["question_count_by_split"].items()
    }
    return [
        _check(
            name="candidate_dataset_row_count_matches_summary",
            passed=int(candidate_scan["row_count"]) == int(summary["total_rows"]),
            observed=int(candidate_scan["row_count"]),
            expected=int(summary["total_rows"]),
        ),
        _check(
            name="candidate_rows_by_split_match_summary",
            passed=observed_rows_by_split == expected_rows_by_split,
            observed=observed_rows_by_split,
            expected=expected_rows_by_split,
        ),
        _check(
            name="candidate_questions_by_split_match_summary",
            passed=observed_questions_by_split == expected_questions_by_split,
            observed=observed_questions_by_split,
            expected=expected_questions_by_split,
        ),
        _check(
            name="candidate_dataset_contains_runtime_features",
            passed=candidate_scan["rows_with_runtime_features"] == candidate_scan["row_count"],
            observed=candidate_scan["rows_with_runtime_features"],
            expected=candidate_scan["row_count"],
        ),
        _check(
            name="candidate_dataset_contains_offline_gold_labels",
            passed=candidate_scan["rows_with_gold_labels"] == candidate_scan["row_count"],
            observed=candidate_scan["rows_with_gold_labels"],
            expected=candidate_scan["row_count"],
        ),
        _check(
            name="runtime_features_exclude_gold_label_keys",
            passed=candidate_scan["rows_with_forbidden_runtime_gold_keys"] == 0,
            observed=candidate_scan["rows_with_forbidden_runtime_gold_keys"],
            expected=0,
        ),
    ]


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    candidate_checks: Sequence[Mapping[str, Any]],
    candidate_scan: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summary = candidate_summary["candidate_artifact_summary"]["summary"]
    failed_candidate_checks = [check for check in candidate_checks if not check["passed"]]
    candidate_splits = set(str(split) for split in summary["splits"])
    return [
        _check(
            name="development_baseline_splits_are_train_dev_only",
            passed=set(split_samples) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=sorted(split_samples),
            expected=sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="candidate_artifact_splits_are_train_dev_only",
            passed=candidate_splits == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=sorted(candidate_splits),
            expected=sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="candidate_rows_have_no_test_split",
            passed=candidate_scan["rows_with_test_split"] == 0,
            observed=candidate_scan["rows_with_test_split"],
            expected=0,
        ),
        _check(
            name="candidate_artifact_checks_passed",
            passed=not failed_candidate_checks,
            observed=len(failed_candidate_checks),
            expected=0,
        ),
        _check(
            name="final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = all(bool(check["passed"]) for check in guard_checks)
    return {
        "status": (
            "primeqa_hybrid_train_dev_development_checks_ready"
            if passed
            else "primeqa_hybrid_train_dev_development_checks_blocked"
        ),
        "can_continue_train_dev_development": passed,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 71: run train/dev candidate-reranker policy development on "
            "primeqa_hybrid_stage68_v1, keeping test locked"
        ),
        "reason": (
            "Train/dev BM25 baselines and candidate artifact checks are ready. "
            "The frozen test split was not evaluated and remains locked."
        ),
    }


def _public_candidate_summary(candidate_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": candidate_summary["stage"],
        "split_name": candidate_summary["split_name"],
        "protocol_version": candidate_summary["protocol_version"],
        "summary": candidate_summary["candidate_artifact_summary"]["summary"],
        "guard_checks": candidate_summary["guard_checks"],
    }


def _bm25_hit_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for split, metrics in report["bm25_baseline"]["metrics_by_split"].items():
        for name, value in metrics["hit_at_k"].items():
            bars.append(BarDatum(f"{split} {name}", float(value), f"{value:.4f}"))
    return bars


def _bm25_mrr_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(split, float(metrics["mrr"]), f"{metrics['mrr']:.4f}")
        for split, metrics in report["bm25_baseline"]["metrics_by_split"].items()
    ]


def _candidate_rows_by_split_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    rows_by_split = report["candidate_artifact_checks"]["scan"]["rows_by_split"]
    return [
        BarDatum(split, float(count), str(count))
        for split, count in sorted(rows_by_split.items())
    ]


def _candidate_questions_by_split_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    question_counts = report["candidate_artifact_checks"]["scan"][
        "question_count_by_split"
    ]
    return [
        BarDatum(split, float(count), str(count))
        for split, count in sorted(question_counts.items())
    ]


def _validate_options(
    *,
    top_k_values: tuple[int, ...],
    bm25_k1: float,
    bm25_b: float,
) -> None:
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(top_k <= 0 for top_k in top_k_values):
        raise ValueError("top_k_values must be positive")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "expected": expected,
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
