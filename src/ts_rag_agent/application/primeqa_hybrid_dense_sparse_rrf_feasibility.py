from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 80"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE_76 = "Stage 76"
_SOURCE_STAGE_79 = "Stage 79"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_CANDIDATE_ID = "dense_sparse_rrf_train_dev_probe"

_PACKAGE_CHECKS = (
    ("numpy", "numpy", True),
    ("sentence-transformers", "sentence_transformers", True),
    ("transformers", "transformers", True),
    ("torch", "torch", True),
    ("scikit-learn", "sklearn", True),
    ("scipy", "scipy", False),
    ("huggingface-hub", "huggingface_hub", False),
    ("faiss-cpu", "faiss", False),
)


@dataclass(frozen=True)
class PrimeQAHybridDenseSparseRRFFeasibilityVisualization:
    """One generated Stage80 dense+sparse RRF feasibility visualization."""

    name: str
    path: str


def check_primeqa_hybrid_dense_sparse_rrf_feasibility(
    *,
    documents_path: Path,
    pyproject_path: Path,
    stage76_report_path: Path,
    stage79_report_path: Path,
    dense_cache_dir: Path,
    huggingface_hub_dir: Path,
    legacy_metric_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Check local feasibility for dense+sparse RRF without downloading models."""

    started_at = time.perf_counter()
    stage76_report = _load_json_object(stage76_report_path)
    stage79_report = _load_json_object(stage79_report_path)
    pyproject_text = _read_text_if_exists(pyproject_path)
    loaded_reports_at = time.perf_counter()

    documents = load_primeqa_documents(documents_path)
    document_ids = tuple(documents)
    loaded_documents_at = time.perf_counter()

    package_checks = _package_checks()
    code_readiness = _code_readiness(pyproject_text=pyproject_text)
    legacy_metric_summaries = _legacy_metric_summaries(legacy_metric_paths)
    cache_candidates = _dense_cache_candidates(
        dense_cache_dir=dense_cache_dir,
        huggingface_hub_dir=huggingface_hub_dir,
        document_ids=document_ids,
        legacy_metric_summaries=legacy_metric_summaries,
    )
    inspected_local_at = time.perf_counter()
    guard_checks = _guard_checks(
        stage76_report=stage76_report,
        stage79_report=stage79_report,
        package_checks=package_checks,
        code_readiness=code_readiness,
        cache_candidates=cache_candidates,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Local feasibility check for dense_sparse_rrf_train_dev_probe. This "
            "stage inspects installed dependencies, existing project dense/hybrid "
            "code, local document-embedding caches, and local Hugging Face model "
            "cache directories. It does not load the frozen test split, does not "
            "run train/dev retrieval metrics, does not run final metrics, does "
            "not use source DOC_IDS as runtime retrieval evidence, does not "
            "download models, and does not choose a dense model silently."
        ),
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "documents": _fingerprint(documents_path),
            "pyproject": _fingerprint(pyproject_path),
            "stage76_report": _fingerprint(stage76_report_path),
            "stage79_report": _fingerprint(stage79_report_path),
        },
        "loaded_data_summary": {
            "document_count": len(document_ids),
            "dense_cache_dir": str(dense_cache_dir),
            "huggingface_hub_dir": str(huggingface_hub_dir),
            "legacy_metric_report_count": len(legacy_metric_summaries),
            "test_split_loaded": False,
            "train_dev_metrics_run": False,
        },
        "dependency_checks": package_checks,
        "code_readiness": code_readiness,
        "dense_cache_candidates": cache_candidates,
        "candidate_options": _candidate_options(cache_candidates),
        "legacy_metric_summaries": legacy_metric_summaries,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            cache_candidates=cache_candidates,
        ),
        "timing_seconds": {
            "load_reports": round(loaded_reports_at - started_at, 3),
            "load_documents": round(loaded_documents_at - loaded_reports_at, 3),
            "inspect_local_environment": round(
                inspected_local_at - loaded_documents_at,
                3,
            ),
            "guard_checks": round(checked_at - inspected_local_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_dense_sparse_rrf_feasibility_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridDenseSparseRRFFeasibilityVisualization]:
    """Write SVG charts for Stage80 dense+sparse RRF feasibility."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage80_dense_cache_readiness.svg": render_horizontal_bar_chart_svg(
            title="Stage80 dense cache readiness",
            bars=_cache_readiness_bars(report),
            x_label="readiness score",
            width=1120,
            margin_left=470,
        ),
        "stage80_dependency_availability.svg": render_horizontal_bar_chart_svg(
            title="Stage80 dependency availability",
            bars=_dependency_bars(report),
            x_label="available",
            width=1120,
            margin_left=300,
        ),
        "stage80_candidate_options.svg": render_horizontal_bar_chart_svg(
            title="Stage80 candidate option status",
            bars=_option_bars(report),
            x_label="eligible",
            width=1120,
            margin_left=470,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridDenseSparseRRFFeasibilityVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _package_checks() -> list[dict[str, Any]]:
    checks = []
    for distribution_name, module_name, required_for_cached_rrf in _PACKAGE_CHECKS:
        module_available = importlib.util.find_spec(module_name) is not None
        version = None
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        checks.append(
            {
                "distribution_name": distribution_name,
                "module_name": module_name,
                "required_for_cached_rrf": required_for_cached_rrf,
                "available": module_available,
                "version": version,
            }
        )
    return checks


def _code_readiness(*, pyproject_text: str) -> dict[str, Any]:
    files = {
        "dense_retriever": Path("src/ts_rag_agent/infrastructure/dense_retriever.py"),
        "dense_embedding_cache": Path(
            "src/ts_rag_agent/infrastructure/dense_embedding_cache.py"
        ),
        "hybrid_retriever": Path("src/ts_rag_agent/infrastructure/hybrid_retriever.py"),
        "evaluate_hybrid_script": Path("scripts/evaluate_hybrid.py"),
    }
    return {
        "files": {
            key: {"path": str(path), "exists": path.exists()}
            for key, path in files.items()
        },
        "pyproject_declares_sentence_transformers": "sentence-transformers" in pyproject_text,
        "pyproject_declares_scikit_learn": "scikit-learn" in pyproject_text,
        "pyproject_declares_rank_bm25": "rank-bm25" in pyproject_text,
        "rrf_implementation": "existing_hybrid_retriever",
        "vector_index_backend": "numpy_matrix_similarity",
        "faiss_required_for_existing_path": False,
    }


def _legacy_metric_summaries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        report = _load_json_object(path)
        dense_section = report.get("dense") or {}
        hybrid_section = report.get("hybrid") or {}
        paths_section = report.get("paths") or {}
        metrics = report.get("metrics") or {}
        summaries.append(
            {
                "path": str(path),
                "stage80_interpretation": (
                    "historical_pre_stage68_reference_only_not_current_split_evidence"
                ),
                "split": report.get("split"),
                "embedding_cache": paths_section.get("embedding_cache"),
                "method": "hybrid_rrf" if hybrid_section else "dense",
                "model_name": hybrid_section.get("dense_model_name")
                or dense_section.get("model_name"),
                "document_text_max_chars": hybrid_section.get("document_text_max_chars")
                or dense_section.get("document_text_max_chars"),
                "query_prefix": dense_section.get("query_prefix"),
                "document_prefix": dense_section.get("document_prefix"),
                "cache_status": hybrid_section.get("dense_cache_status")
                or dense_section.get("cache_status"),
                "hit_at_k": metrics.get("hit_at_k") or {},
                "mrr": metrics.get("mrr"),
            }
        )
    return summaries


def _dense_cache_candidates(
    *,
    dense_cache_dir: Path,
    huggingface_hub_dir: Path,
    document_ids: tuple[str, ...],
    legacy_metric_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not dense_cache_dir.exists():
        return []
    candidates = []
    for path in sorted(dense_cache_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            cached_document_ids = tuple(str(value) for value in data["document_ids"])
            embeddings = data["embeddings"]
            model_name = str(data["model_name"])
            document_text_max_chars = int(data["document_text_max_chars"])
            document_prefix = str(data["document_prefix"]) if "document_prefix" in data else ""
            model_cache = _huggingface_model_cache(model_name, huggingface_hub_dir)
            legacy_matches = _matching_legacy_metrics(path, legacy_metric_summaries)
            candidates.append(
                {
                    "cache_path": str(path),
                    "cache_bytes": path.stat().st_size,
                    "cache_sha256": _sha256_file(path),
                    "model_name": model_name,
                    "document_text_max_chars": document_text_max_chars,
                    "document_prefix": document_prefix,
                    "embedding_shape": list(embeddings.shape),
                    "embedding_dtype": str(embeddings.dtype),
                    "document_id_count": len(cached_document_ids),
                    "current_document_id_count": len(document_ids),
                    "document_ids_match_current_corpus": cached_document_ids
                    == document_ids,
                    "huggingface_model_cache": model_cache,
                    "legacy_metric_matches": legacy_matches,
                    "can_run_without_reencoding_documents": cached_document_ids
                    == document_ids,
                    "can_run_without_model_download": bool(model_cache["exists"])
                    and int(model_cache["snapshot_count"]) > 0,
                }
            )
    return candidates


def _huggingface_model_cache(
    model_name: str,
    huggingface_hub_dir: Path,
) -> dict[str, Any]:
    model_dir = huggingface_hub_dir / f"models--{model_name.replace('/', '--')}"
    refs_main_path = model_dir / "refs" / "main"
    snapshot_dir = model_dir / "snapshots"
    snapshots = (
        sorted(path.name for path in snapshot_dir.iterdir())
        if snapshot_dir.exists()
        else []
    )
    return {
        "model_cache_dir": str(model_dir),
        "exists": model_dir.exists(),
        "refs_main": refs_main_path.read_text(encoding="utf-8").strip()
        if refs_main_path.exists()
        else None,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }


def _matching_legacy_metrics(
    cache_path: Path,
    legacy_metric_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized_cache_path = str(cache_path).replace("/", "\\")
    matches = []
    for summary in legacy_metric_summaries:
        embedding_cache = str(summary.get("embedding_cache") or "").replace("/", "\\")
        if embedding_cache == normalized_cache_path:
            matches.append(dict(summary))
    return matches


def _candidate_options(
    cache_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [
        candidate
        for candidate in cache_candidates
        if candidate["document_ids_match_current_corpus"]
        and candidate["can_run_without_model_download"]
    ]
    options = [
        {
            "option_id": "compare_existing_cached_dense_models",
            "description": (
                "Run a fixed train/dev-only dense+sparse RRF comparison across all "
                "eligible existing local dense caches, then select by train and "
                "validate on dev."
            ),
            "recommended": len(eligible) >= 2,
            "eligible": len(eligible) >= 2,
            "requires_user_confirmation": True,
            "download_required": False,
            "candidate_count": len(eligible),
        }
    ]
    for candidate in eligible:
        options.append(
            {
                "option_id": f"single_cached_model::{candidate['model_name']}",
                "description": (
                    "Run the train/dev-only dense+sparse RRF probe using this one "
                    "existing local dense cache."
                ),
                "recommended": False,
                "eligible": True,
                "requires_user_confirmation": True,
                "download_required": False,
                "model_name": candidate["model_name"],
                "cache_path": candidate["cache_path"],
            }
        )
    return options


def _guard_checks(
    *,
    stage76_report: Mapping[str, Any],
    stage79_report: Mapping[str, Any],
    package_checks: Sequence[Mapping[str, Any]],
    code_readiness: Mapping[str, Any],
    cache_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    required_packages_available = all(
        check["available"]
        for check in package_checks
        if check["required_for_cached_rrf"]
    )
    code_files_available = all(row["exists"] for row in code_readiness["files"].values())
    compatible_cache_count = sum(
        bool(candidate["document_ids_match_current_corpus"])
        and bool(candidate["can_run_without_model_download"])
        for candidate in cache_candidates
    )
    stage76_candidates = stage76_report.get("candidate_designs") or []
    stage80_candidate_present = any(
        candidate.get("candidate_id") == _CANDIDATE_ID
        and candidate.get("status") == "recommended_for_train_dev_experiment"
        for candidate in stage76_candidates
    )
    stage79_decision = stage79_report.get("decision") or {}
    return [
        _check(
            name="stage76_source_report_is_stage76",
            passed=str(stage76_report.get("stage") or "") == _SOURCE_STAGE_76,
            observed=str(stage76_report.get("stage") or ""),
            expected=_SOURCE_STAGE_76,
        ),
        _check(
            name="stage76_dense_sparse_candidate_is_allowed",
            passed=stage80_candidate_present,
            observed=stage80_candidate_present,
            expected=True,
        ),
        _check(
            name="stage79_source_report_is_stage79",
            passed=str(stage79_report.get("stage") or "") == _SOURCE_STAGE_79,
            observed=str(stage79_report.get("stage") or ""),
            expected=_SOURCE_STAGE_79,
        ),
        _check(
            name="stage79_did_not_open_final_test_gate",
            passed=stage79_decision.get("can_open_final_test_gate_now") is False,
            observed=stage79_decision.get("can_open_final_test_gate_now"),
            expected=False,
        ),
        _check(
            name="required_cached_rrf_packages_available",
            passed=required_packages_available,
            observed=required_packages_available,
            expected=True,
        ),
        _check(
            name="existing_dense_hybrid_code_available",
            passed=code_files_available,
            observed=code_files_available,
            expected=True,
        ),
        _check(
            name="compatible_local_dense_cache_available",
            passed=compatible_cache_count > 0,
            observed=compatible_cache_count,
            expected="> 0",
        ),
        _check(
            name="no_model_download_attempted",
            passed=True,
            observed="not_attempted",
            expected="not_attempted",
        ),
        _check(
            name="train_dev_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="source_doc_ids_not_used_as_runtime_evidence",
            passed=True,
            observed="not_used",
            expected="not_used",
        ),
        _check(
            name="default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    cache_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failed_checks = [check["name"] for check in guard_checks if not check["passed"]]
    compatible_cache_count = sum(
        bool(candidate["document_ids_match_current_corpus"])
        and bool(candidate["can_run_without_model_download"])
        for candidate in cache_candidates
    )
    if failed_checks:
        return {
            "status": "primeqa_hybrid_dense_sparse_rrf_feasibility_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_run_dense_sparse_rrf_without_download": False,
            "requires_user_confirmation_before_train_dev_run": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_dense_sparse_rrf_feasibility_completed",
        "compatible_local_dense_cache_count": compatible_cache_count,
        "can_continue_train_dev_development": True,
        "can_run_dense_sparse_rrf_without_download": compatible_cache_count > 0,
        "requires_user_confirmation_before_train_dev_run": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 81: confirm the dense model/cache protocol, then run a "
            "train/dev-only dense+sparse RRF probe with existing local caches; "
            "keep test locked and do not download models silently."
        ),
    }


def _cache_readiness_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for candidate in report["dense_cache_candidates"]:
        readiness = 0.0
        readiness += 0.5 if candidate["document_ids_match_current_corpus"] else 0.0
        readiness += 0.5 if candidate["can_run_without_model_download"] else 0.0
        bars.append(
            BarDatum(
                label=candidate["model_name"],
                value=readiness,
                value_label=f"{readiness:.1f}",
            )
        )
    return sorted(bars, key=lambda bar: (-bar.value, bar.label))


def _dependency_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=check["distribution_name"],
            value=1.0 if check["available"] else 0.0,
            value_label="yes" if check["available"] else "no",
        )
        for check in report["dependency_checks"]
    ]


def _option_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=option["option_id"],
            value=1.0 if option["eligible"] else 0.0,
            value_label="eligible" if option["eligible"] else "blocked",
        )
        for option in report["candidate_options"]
    ]


def _load_json_object(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_text_if_exists(path: Path) -> str:
    _ensure_file(path)
    return path.read_text(encoding="utf-8")


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def _check(
    *,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }
