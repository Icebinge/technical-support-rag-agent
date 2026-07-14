from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    DEFAULT_MODEL_NAMES,
    CandidateRerankerCVResult,
    candidate_reranker_cv_result_to_dict,
    cross_validate_candidate_rerankers,
    write_cv_visualizations,
)
from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.candidate_score_guarded_policy_evaluation import (
    write_candidate_score_guarded_policy_visualizations,
)
from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    CandidateScoreGuardedPolicySplitValidationResult,
    candidate_score_guarded_policy_split_validation_to_dict,
    evaluate_candidate_score_guarded_policy_split_validation,
)
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 71"
_CREATED_AT = "2026-07-14"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridCandidateRerankerDevelopmentVisualization:
    """One generated Stage71 visualization."""

    group: str
    name: str
    path: str


@dataclass(frozen=True)
class PrimeQAHybridCandidateRerankerDevelopmentRun:
    """In-memory Stage71 run with typed results retained for visualization."""

    report: dict[str, Any]
    train_cv_result: CandidateRerankerCVResult
    split_validation_results: list[CandidateScoreGuardedPolicySplitValidationResult]


def run_primeqa_hybrid_candidate_reranker_development(
    *,
    candidate_dataset_path: Path,
    candidate_summary_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    fold_count: int = 5,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    policy_model_names: Sequence[str] | None = None,
    max_answer_candidates: int = 3,
) -> PrimeQAHybridCandidateRerankerDevelopmentRun:
    """Run train/dev-only candidate-reranker development on the frozen split."""

    normalized_model_names = tuple(model_names)
    normalized_policy_model_names = tuple(policy_model_names or normalized_model_names)
    _validate_options(
        fold_count=fold_count,
        model_names=normalized_model_names,
        policy_model_names=normalized_policy_model_names,
        max_answer_candidates=max_answer_candidates,
    )
    started_at = time.perf_counter()
    rows = load_candidate_reranker_rows(candidate_dataset_path)
    candidate_summary = _load_candidate_summary(candidate_summary_path)
    loaded_candidates_at = time.perf_counter()
    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    gold_answers = _gold_answers_by_question_key(split_samples)
    loaded_splits_at = time.perf_counter()
    rows_by_split = _rows_by_split(rows)
    train_cv_result = cross_validate_candidate_rerankers(
        rows=rows_by_split[_TRAIN_SPLIT],
        fold_count=fold_count,
        model_names=normalized_model_names,
    )
    train_cv_at = time.perf_counter()
    split_validation_results = [
        evaluate_candidate_score_guarded_policy_split_validation(
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name=policy_model_name,
            train_split=_TRAIN_SPLIT,
            evaluation_split=_DEV_SPLIT,
            train_fold_count=fold_count,
            max_answer_candidates=max_answer_candidates,
        )
        for policy_model_name in normalized_policy_model_names
    ]
    split_validation_at = time.perf_counter()
    candidate_checks = _candidate_checks(
        rows=rows,
        rows_by_split=rows_by_split,
        candidate_summary=candidate_summary,
        gold_answers_by_question_key=gold_answers,
    )
    guard_checks = _guard_checks(
        rows_by_split=rows_by_split,
        candidate_summary=candidate_summary,
        candidate_checks=candidate_checks,
        split_validation_results=split_validation_results,
        gold_answers_by_question_key=gold_answers,
    )
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only candidate-reranker policy development for "
            "primeqa_hybrid_stage68_v1. This stage runs train-only grouped CV, "
            "then train-to-dev split validation for fixed guarded policies. It "
            "keeps the frozen test split locked, does not run final metrics, and "
            "does not change the default runtime."
        ),
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "train_split": _TRAIN_SPLIT,
            "development_evaluation_split": _DEV_SPLIT,
            "allowed_development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "candidate_dataset": _fingerprint(candidate_dataset_path),
            "candidate_summary": _fingerprint(candidate_summary_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
        },
        "loaded_candidate_summary": _loaded_candidate_summary(
            rows=rows,
            rows_by_split=rows_by_split,
        ),
        "loaded_gold_answer_summary": _gold_answer_summary(gold_answers),
        "candidate_artifact_summary": _public_candidate_summary(candidate_summary),
        "train_only_model_cv": candidate_reranker_cv_result_to_dict(train_cv_result),
        "train_to_dev_policy_validations": [
            candidate_score_guarded_policy_split_validation_to_dict(
                split_validation_result
            )
            for split_validation_result in split_validation_results
        ],
        "candidate_checks": candidate_checks,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_candidates": round(loaded_candidates_at - started_at, 3),
            "load_train_dev_splits": round(loaded_splits_at - loaded_candidates_at, 3),
            "train_only_cv": round(train_cv_at - loaded_splits_at, 3),
            "train_to_dev_policy_validation": round(
                split_validation_at - train_cv_at,
                3,
            ),
            "candidate_checks": round(checked_at - split_validation_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return PrimeQAHybridCandidateRerankerDevelopmentRun(
        report=report,
        train_cv_result=train_cv_result,
        split_validation_results=split_validation_results,
    )


def write_primeqa_hybrid_candidate_reranker_development_visualizations(
    run: PrimeQAHybridCandidateRerankerDevelopmentRun,
    output_dir: Path,
) -> list[PrimeQAHybridCandidateRerankerDevelopmentVisualization]:
    """Write SVG charts for Stage71 development evidence."""

    output_dir.mkdir(parents=True, exist_ok=True)
    visualizations: list[PrimeQAHybridCandidateRerankerDevelopmentVisualization] = []
    for artifact in write_cv_visualizations(
        result=run.train_cv_result,
        output_dir=output_dir,
    ):
        visualizations.append(
            PrimeQAHybridCandidateRerankerDevelopmentVisualization(
                group="train_only_model_cv",
                name=artifact.name,
                path=artifact.path,
            )
        )
    for split_validation_result in run.split_validation_results:
        model_token = _model_token(split_validation_result.model_name)
        for artifact in write_candidate_score_guarded_policy_visualizations(
            result=split_validation_result.train_cv_evaluation,
            output_dir=output_dir,
            artifact_prefix=f"stage71_{model_token}_train_cv",
            title_prefix=f"Stage 71 {model_token} train-only CV",
        ):
            visualizations.append(
                PrimeQAHybridCandidateRerankerDevelopmentVisualization(
                    group=f"{split_validation_result.model_name}_train_cv_policy",
                    name=artifact.name,
                    path=artifact.path,
                )
            )
        for artifact in write_candidate_score_guarded_policy_visualizations(
            result=split_validation_result.holdout_evaluation,
            output_dir=output_dir,
            artifact_prefix=f"stage71_{model_token}_dev_holdout",
            title_prefix=f"Stage 71 {model_token} train-to-dev holdout",
        ):
            visualizations.append(
                PrimeQAHybridCandidateRerankerDevelopmentVisualization(
                    group=f"{split_validation_result.model_name}_train_to_dev_policy",
                    name=artifact.name,
                    path=artifact.path,
                )
            )
    return visualizations


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


def _gold_answers_by_question_key(
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
) -> dict[str, str]:
    answers = {}
    for split, samples in split_samples.items():
        for sample in samples:
            if sample.answerable:
                answers[f"{split}::{sample.sample_id}"] = sample.answer
    return dict(sorted(answers.items()))


def _rows_by_split(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    rows_by_split: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split") or "")
        rows_by_split.setdefault(split, []).append(row)
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        rows_by_split.setdefault(split, [])
    return rows_by_split


def _loaded_candidate_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    question_ids_by_split = {
        split: sorted({str(row["question_id"]) for row in split_rows})
        for split, split_rows in rows_by_split.items()
    }
    return {
        "row_count": len(rows),
        "rows_by_split": {
            split: len(split_rows)
            for split, split_rows in sorted(rows_by_split.items())
            if split_rows
        },
        "question_count_by_split": {
            split: len(question_ids)
            for split, question_ids in sorted(question_ids_by_split.items())
            if question_ids
        },
        "candidate_splits": sorted({str(row.get("split") or "") for row in rows}),
    }


def _gold_answer_summary(gold_answers_by_question_key: Mapping[str, str]) -> dict[str, Any]:
    counts = Counter(key.split("::", maxsplit=1)[0] for key in gold_answers_by_question_key)
    return {
        "answer_count": len(gold_answers_by_question_key),
        "answer_count_by_split": dict(sorted(counts.items())),
    }


def _public_candidate_summary(candidate_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": candidate_summary["stage"],
        "split_name": candidate_summary["split_name"],
        "protocol_version": candidate_summary["protocol_version"],
        "summary": candidate_summary["candidate_artifact_summary"]["summary"],
        "guard_checks": candidate_summary["guard_checks"],
    }


def _candidate_checks(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_summary: Mapping[str, Any],
    gold_answers_by_question_key: Mapping[str, str],
) -> list[dict[str, Any]]:
    summary = candidate_summary["candidate_artifact_summary"]["summary"]
    observed_rows_by_split = {
        split: len(split_rows)
        for split, split_rows in sorted(rows_by_split.items())
        if split_rows
    }
    observed_questions_by_split = {
        split: len({str(row["question_id"]) for row in split_rows})
        for split, split_rows in sorted(rows_by_split.items())
        if split_rows
    }
    candidate_question_keys = _candidate_question_keys(rows)
    missing_gold_answers = sorted(candidate_question_keys - set(gold_answers_by_question_key))
    return [
        _check(
            name="candidate_dataset_row_count_matches_summary",
            passed=len(rows) == int(summary["total_rows"]),
            observed=len(rows),
            expected=int(summary["total_rows"]),
        ),
        _check(
            name="candidate_rows_by_split_match_summary",
            passed=observed_rows_by_split == dict(summary["rows_by_split"]),
            observed=observed_rows_by_split,
            expected=dict(summary["rows_by_split"]),
        ),
        _check(
            name="candidate_questions_by_split_match_summary",
            passed=observed_questions_by_split == dict(summary["questions_by_split"]),
            observed=observed_questions_by_split,
            expected=dict(summary["questions_by_split"]),
        ),
        _check(
            name="gold_answers_cover_candidate_questions",
            passed=not missing_gold_answers,
            observed={
                "candidate_question_count": len(candidate_question_keys),
                "missing_gold_answer_count": len(missing_gold_answers),
                "missing_gold_answers": missing_gold_answers[:20],
            },
            expected={"missing_gold_answer_count": 0},
        ),
    ]


def _guard_checks(
    *,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_summary: Mapping[str, Any],
    candidate_checks: Sequence[Mapping[str, Any]],
    split_validation_results: Sequence[CandidateScoreGuardedPolicySplitValidationResult],
    gold_answers_by_question_key: Mapping[str, str],
) -> list[dict[str, Any]]:
    summary = candidate_summary["candidate_artifact_summary"]["summary"]
    candidate_splits = sorted(split for split, split_rows in rows_by_split.items() if split_rows)
    failed_candidate_checks = [check for check in candidate_checks if not check["passed"]]
    gold_answer_splits = sorted(
        {key.split("::", maxsplit=1)[0] for key in gold_answers_by_question_key}
    )
    return [
        _check(
            name="candidate_artifact_splits_are_train_dev_only",
            passed=set(candidate_splits) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=candidate_splits,
            expected=sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="candidate_summary_splits_are_train_dev_only",
            passed=set(summary["splits"]) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=sorted(summary["splits"]),
            expected=sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="candidate_rows_have_no_test_split",
            passed=not any(split in _FORBIDDEN_FINAL_SPLITS for split in candidate_splits),
            observed=sum(
                len(rows_by_split.get(split, [])) for split in _FORBIDDEN_FINAL_SPLITS
            ),
            expected=0,
        ),
        _check(
            name="gold_answer_splits_are_train_dev_only",
            passed=set(gold_answer_splits) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=gold_answer_splits,
            expected=sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="train_cv_uses_train_only",
            passed=all(
                result.train_cv_evaluation.evaluation_split == _TRAIN_SPLIT
                for result in split_validation_results
            ),
            observed={
                result.model_name: result.train_cv_evaluation.evaluation_split
                for result in split_validation_results
            },
            expected={result.model_name: _TRAIN_SPLIT for result in split_validation_results},
        ),
        _check(
            name="split_validations_are_train_to_dev",
            passed=all(
                result.train_split == _TRAIN_SPLIT and result.evaluation_split == _DEV_SPLIT
                for result in split_validation_results
            ),
            observed={
                result.model_name: {
                    "train_split": result.train_split,
                    "evaluation_split": result.evaluation_split,
                }
                for result in split_validation_results
            },
            expected={
                result.model_name: {
                    "train_split": _TRAIN_SPLIT,
                    "evaluation_split": _DEV_SPLIT,
                }
                for result in split_validation_results
            },
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
        _check(
            name="default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _candidate_question_keys(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {f"{row['split']}::{row['question_id']}" for row in rows}


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = all(bool(check["passed"]) for check in guard_checks)
    return {
        "status": (
            "primeqa_hybrid_candidate_reranker_development_ready"
            if passed
            else "primeqa_hybrid_candidate_reranker_development_blocked"
        ),
        "can_continue_train_dev_development": passed,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 72: review Stage71 train/dev candidate-reranker policy "
            "changed cases before considering any final-test evaluation gate"
        ),
        "reason": (
            "Stage71 completed train-only CV and train-to-dev guarded policy "
            "validation without using the frozen test split."
            if passed
            else "One or more Stage71 train/dev boundary checks failed."
        ),
    }


def _validate_options(
    *,
    fold_count: int,
    model_names: Sequence[str],
    policy_model_names: Sequence[str],
    max_answer_candidates: int,
) -> None:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if not tuple(model_names):
        raise ValueError("model_names must not be empty")
    if not tuple(policy_model_names):
        raise ValueError("policy_model_names must not be empty")
    if any(not policy_model_name.strip() for policy_model_name in policy_model_names):
        raise ValueError("policy_model_names must not contain empty values")
    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")


def _model_token(model_name: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in model_name
    ).strip("_")


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
