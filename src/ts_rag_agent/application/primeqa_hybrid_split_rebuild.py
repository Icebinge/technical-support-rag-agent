from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.candidate_reranker_dataset import (
    CandidateRerankerDatasetBuild,
    build_candidate_reranker_dataset,
    candidate_reranker_dataset_build_to_dict,
    candidate_reranker_row_to_dict,
)
from ts_rag_agent.application.evidence_selection import create_sentence_evidence_selector
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
    write_primeqa_compatible_question_files,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 69"
_CREATED_AT = "2026-07-14"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_DEFAULT_CANDIDATE_SPLITS = ("train", "dev")
_FORBIDDEN_TUNING_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridRebuildBundle:
    """Stage69 report and local candidate dataset build."""

    report: dict[str, Any]
    split_samples: dict[str, list[PrimeQAHybridSplitSample]]
    candidate_build: CandidateRerankerDatasetBuild


@dataclass(frozen=True)
class PrimeQAHybridRebuildVisualization:
    """One generated Stage69 visualization."""

    name: str
    path: str


def rebuild_primeqa_hybrid_train_dev_artifacts(
    *,
    split_paths: Mapping[str, Path],
    documents_path: Path,
    candidate_splits: Sequence[str] = _DEFAULT_CANDIDATE_SPLITS,
    retrieval_top_k: int = 5,
    evidence_selector_name: str = "hybrid-routing",
    max_candidates_per_document: int = 3,
    candidate_limit: int = 25,
    min_candidate_score: float = 2.0,
) -> PrimeQAHybridRebuildBundle:
    """Load frozen splits and build train/dev candidate artifacts."""

    candidate_splits = tuple(split.strip().lower() for split in candidate_splits)
    _validate_options(
        split_paths=split_paths,
        candidate_splits=candidate_splits,
        retrieval_top_k=retrieval_top_k,
        max_candidates_per_document=max_candidates_per_document,
        candidate_limit=candidate_limit,
        min_candidate_score=min_candidate_score,
    )
    started_at = time.perf_counter()
    split_samples = {
        split: load_primeqa_hybrid_split_samples(path)
        for split, path in sorted(split_paths.items())
    }
    loaded_splits_at = time.perf_counter()
    split_questions = {
        split: [sample.to_primeqa_question() for sample in split_samples[split]]
        for split in candidate_splits
    }
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    loaded_documents_at = time.perf_counter()
    retriever = BM25Retriever()
    retriever.fit(documents)
    indexed_at = time.perf_counter()
    selector = create_sentence_evidence_selector(
        selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
    )
    candidate_build = build_candidate_reranker_dataset(
        split_questions=split_questions,
        search_fn=lambda question, top_k: retriever.search(
            question.full_question,
            top_k=top_k,
        ),
        evidence_selector=selector,
        retrieval_top_k=retrieval_top_k,
        candidate_limit=candidate_limit,
        min_candidate_score=min_candidate_score,
    )
    built_candidates_at = time.perf_counter()
    guard_checks = _guard_checks(
        split_samples=split_samples,
        candidate_splits=candidate_splits,
        candidate_build=candidate_build,
    )
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Rebuild loaders and train/dev derived candidate artifacts from "
            "the frozen PrimeQA hybrid split. This stage uses no test rows for "
            "training or tuning, does not run final metrics, and does not "
            "change the default runtime."
        ),
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "question_id_policy": (
                "PrimeQAQuestion.id is the frozen sample_id "
                "`source_split:QUESTION_ID` to avoid source duplicate collisions."
            ),
            "candidate_artifact_splits": list(candidate_splits),
            "forbidden_tuning_splits": sorted(_FORBIDDEN_TUNING_SPLITS),
        },
        "source_files": {
            "split_paths": {
                split: _fingerprint(path) for split, path in sorted(split_paths.items())
            },
            "documents": _fingerprint(documents_path),
        },
        "loaded_split_summary": summarize_primeqa_hybrid_split_samples(split_samples),
        "candidate_build_config": {
            "retrieval_top_k": retrieval_top_k,
            "evidence_selector": selector.name,
            "max_candidates_per_document": max_candidates_per_document,
            "candidate_limit": candidate_limit,
            "min_candidate_score": min_candidate_score,
        },
        "candidate_build_summary": candidate_reranker_dataset_build_to_dict(
            candidate_build
        ),
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_splits": round(loaded_splits_at - started_at, 3),
            "load_documents": round(loaded_documents_at - loaded_splits_at, 3),
            "bm25_index": round(indexed_at - loaded_documents_at, 3),
            "candidate_build": round(built_candidates_at - indexed_at, 3),
            "total": round(built_candidates_at - started_at, 3),
        },
    }
    return PrimeQAHybridRebuildBundle(
        report=report,
        split_samples=split_samples,
        candidate_build=candidate_build,
    )


def write_primeqa_hybrid_rebuild_question_artifacts(
    bundle: PrimeQAHybridRebuildBundle,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Write PrimeQA-compatible train/dev/test question JSON files."""

    artifacts = write_primeqa_compatible_question_files(
        split_samples=bundle.split_samples,
        output_dir=output_dir,
    )
    return [
        {
            **artifact,
            "sha256": _file_sha256(Path(artifact["path"])),
        }
        for artifact in artifacts
    ]


def write_primeqa_hybrid_rebuild_candidate_artifacts(
    bundle: PrimeQAHybridRebuildBundle,
    *,
    dataset_output: Path,
    summary_output: Path,
) -> dict[str, Any]:
    """Write train/dev candidate reranker JSONL and a public-safe summary."""

    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    with dataset_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in bundle.candidate_build.rows:
            handle.write(
                _json_dumps(candidate_reranker_row_to_dict(row))
            )
            handle.write("\n")
    summary = {
        "stage": _STAGE,
        "split_name": _SPLIT_NAME,
        "protocol_version": _PROTOCOL_VERSION,
        "candidate_artifact_summary": candidate_reranker_dataset_build_to_dict(
            bundle.candidate_build
        ),
        "guard_checks": bundle.report["guard_checks"],
        "contains_raw_candidate_text": False,
        "raw_text_location": "dataset_jsonl_metadata_fields_only",
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "dataset_path": str(dataset_output),
        "dataset_sha256": _file_sha256(dataset_output),
        "dataset_row_count": len(bundle.candidate_build.rows),
        "summary_path": str(summary_output),
        "summary_sha256": _file_sha256(summary_output),
        "contains_runtime_features": True,
        "contains_gold_labels": True,
        "contains_truncated_raw_metadata_text": True,
        "git_policy": "ignored_artifact",
    }


def write_primeqa_hybrid_rebuild_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRebuildVisualization]:
    """Write SVG charts for Stage69 loader and candidate rebuild results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage69_primeqa_loaded_split_rows.svg": render_horizontal_bar_chart_svg(
            title="Stage69 loaded split rows",
            bars=_loaded_split_bars(report, "row_count"),
            x_label="row count",
            margin_left=240,
        ),
        "stage69_primeqa_loaded_answerable_rows.svg": render_horizontal_bar_chart_svg(
            title="Stage69 loaded answerable rows",
            bars=_loaded_split_bars(report, "answerable_count"),
            x_label="answerable row count",
            margin_left=240,
        ),
        "stage69_primeqa_candidate_rows_by_split.svg": render_horizontal_bar_chart_svg(
            title="Stage69 candidate rows by split",
            bars=_candidate_rows_by_split_bars(report),
            x_label="candidate row count",
            margin_left=260,
        ),
        "stage69_primeqa_candidate_questions_by_split.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage69 candidate questions by split",
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
        artifacts.append(PrimeQAHybridRebuildVisualization(name=filename, path=str(path)))
    return artifacts


def _guard_checks(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    candidate_splits: Sequence[str],
    candidate_build: CandidateRerankerDatasetBuild,
) -> list[dict[str, Any]]:
    forbidden_candidate_splits = sorted(
        set(candidate_splits).intersection(_FORBIDDEN_TUNING_SPLITS)
    )
    candidate_summary_splits = set(candidate_build.summary.splits)
    expected_candidate_splits = {
        split
        for split in candidate_splits
        if any(sample.answerable for sample in split_samples[split])
    }
    return [
        _check(
            name="test_split_not_used_for_candidate_training_artifact",
            passed=not forbidden_candidate_splits,
            observed=forbidden_candidate_splits,
            expected=[],
        ),
        _check(
            name="candidate_build_splits_match_allowed_train_dev",
            passed=candidate_summary_splits == expected_candidate_splits,
            observed=sorted(candidate_summary_splits),
            expected=sorted(expected_candidate_splits),
        ),
        _check(
            name="all_loaded_splits_have_rows",
            passed=all(samples for samples in split_samples.values()),
            observed={
                split: len(samples)
                for split, samples in sorted(split_samples.items())
            },
            expected="nonempty",
        ),
        _check(
            name="candidate_rows_have_no_test_split",
            passed=all(row.split not in _FORBIDDEN_TUNING_SPLITS for row in candidate_build.rows),
            observed=sorted({row.split for row in candidate_build.rows}),
            expected=sorted(set(candidate_splits)),
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = all(bool(check["passed"]) for check in guard_checks)
    return {
        "status": (
            "primeqa_hybrid_train_dev_rebuild_ready"
            if passed
            else "primeqa_hybrid_train_dev_rebuild_blocked"
        ),
        "can_run_train_dev_metrics_next": passed,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 70: rerun PrimeQA train/dev baselines and candidate-reranker "
            "development checks on primeqa_hybrid_stage68_v1, keeping test "
            "locked"
        ),
        "reason": (
            "Frozen split loaders and train/dev candidate artifacts are ready "
            "for development reruns. The frozen test split remains locked for "
            "future final evaluation only."
        ),
    }


def _loaded_split_bars(report: Mapping[str, Any], metric_name: str) -> list[BarDatum]:
    return [
        BarDatum(split, float(summary[metric_name]), str(summary[metric_name]))
        for split, summary in report["loaded_split_summary"].items()
    ]


def _candidate_rows_by_split_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    rows_by_split = report["candidate_build_summary"]["summary"]["rows_by_split"]
    return [
        BarDatum(split, float(count), str(count))
        for split, count in sorted(rows_by_split.items())
    ]


def _candidate_questions_by_split_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    questions_by_split = report["candidate_build_summary"]["summary"][
        "questions_by_split"
    ]
    return [
        BarDatum(split, float(count), str(count))
        for split, count in sorted(questions_by_split.items())
    ]


def _validate_options(
    *,
    split_paths: Mapping[str, Path],
    candidate_splits: Sequence[str],
    retrieval_top_k: int,
    max_candidates_per_document: int,
    candidate_limit: int,
    min_candidate_score: float,
) -> None:
    required_splits = {"train", "dev", "test"}
    missing_splits = sorted(required_splits - set(split_paths))
    if missing_splits:
        raise ValueError(f"Missing split_paths for: {', '.join(missing_splits)}")
    if not candidate_splits:
        raise ValueError("candidate_splits must not be empty")
    invalid_candidate_splits = sorted(set(candidate_splits) - required_splits)
    if invalid_candidate_splits:
        raise ValueError(
            f"Unsupported candidate_splits: {', '.join(invalid_candidate_splits)}"
        )
    forbidden = sorted(set(candidate_splits).intersection(_FORBIDDEN_TUNING_SPLITS))
    if forbidden:
        raise ValueError(
            "Test split is locked and cannot be used for candidate training "
            f"artifacts: {', '.join(forbidden)}"
        )
    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")
    if max_candidates_per_document <= 0:
        raise ValueError("max_candidates_per_document must be positive")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    if min_candidate_score < 0:
        raise ValueError("min_candidate_score must be non-negative")


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dumps(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


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
