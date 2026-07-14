from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 59"
_CREATED_AT = "2026-07-14"
_PRIMARY_MODE = "answer_only"
_DIAGNOSTIC_MODE = "question_answer_page_text"
_PASS = "pass"
_BLOCKED = "blocked"
_INFO = "info"
_BLOCKER = "blocker"
_OBSERVATION = "observation"


@dataclass(frozen=True)
class CompatibilityGateCheck:
    """One Stage 51 to MSQA compatibility gate check."""

    check_id: str
    status: str
    severity: str
    evidence: str
    decision_effect: str


@dataclass(frozen=True)
class MsqaStage51CompatibilityVisualization:
    """One generated Stage 59 compatibility visualization."""

    name: str
    path: str


def review_msqa_stage51_compatibility(stage58_report_path: Path) -> dict[str, Any]:
    """Review whether Stage 51 can be fairly compared on the Stage 58 MSQA task."""

    _ensure_file(stage58_report_path)
    stage58_report = json.loads(stage58_report_path.read_text(encoding="utf-8"))
    _validate_stage58_report(stage58_report)

    primary = _variant_by_mode(stage58_report, _PRIMARY_MODE)
    diagnostic = _variant_by_mode(stage58_report, _DIAGNOSTIC_MODE)
    samples = int(stage58_report["data"]["frozen_split_samples"])
    max_k = _max_hit_k(primary)
    primary_failure_counts = {
        str(key): int(value)
        for key, value in primary["failure_mode_counts"].items()
    }
    primary_failure_rates = {
        key: round(value / samples, 4)
        for key, value in primary_failure_counts.items()
    }
    metric_gap = _primary_vs_diagnostic_gap(primary=primary, diagnostic=diagnostic)
    checks = _gate_checks(
        stage58_report=stage58_report,
        primary=primary,
        diagnostic=diagnostic,
        max_k=max_k,
    )
    gate_summary = _gate_summary(checks)

    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "MSQA Stage 58 failure-mode review and Stage 51 compatibility gate. "
            "This report does not run Stage 51, does not tune policies, does not "
            "run PrimeQA verified RAG, and does not change the default runtime."
        ),
        "source_files": {
            "stage58_report": _fingerprint(stage58_report_path),
            "stage51_runtime_policy": _fingerprint(
                Path(
                    "src/ts_rag_agent/application/"
                    "candidate_score_guarded_composition_policy.py"
                )
            ),
            "stage51_candidate_dataset_contract": _fingerprint(
                Path("src/ts_rag_agent/application/candidate_reranker_dataset.py")
            ),
        },
        "stage58_baseline_summary": {
            "split_name": stage58_report["input_contract"]["split_name"],
            "adapter_contract_version": stage58_report["input_contract"][
                "adapter_contract_version"
            ],
            "corpus_scope": stage58_report["baseline_definition"]["corpus_scope"],
            "evaluated_questions": samples,
            "primary_variant": _PRIMARY_MODE,
            "diagnostic_variant": _DIAGNOSTIC_MODE,
            "max_k": max_k,
        },
        "stage51_contract_summary": {
            "policy_name": (
                "candidate_score_gte_60_rank_contained_"
                "preserve_baseline_out_of_rank_guarded_reranker"
            ),
            "runtime_question_type": "PrimeQAQuestion",
            "runtime_candidate_type": "SentenceEvidenceCandidate",
            "requires_candidate_score": True,
            "requires_candidate_sentence": True,
            "requires_document_id_citation_identity": True,
            "requires_retrieval_rank_guard": True,
            "training_dataset": (
                "artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl"
            ),
            "training_boundary": "PrimeQA dev/train candidate-reranker rows",
        },
        "failure_mode_review": {
            "primary_variant": _PRIMARY_MODE,
            "primary_retrieval_metrics": primary["retrieval_metrics"],
            "primary_answer_metrics": primary["answer_metrics"],
            "primary_failure_counts": primary_failure_counts,
            "primary_failure_rates": primary_failure_rates,
            "diagnostic_retrieval_metrics": diagnostic["retrieval_metrics"],
            "diagnostic_answer_metrics": diagnostic["answer_metrics"],
            "primary_vs_diagnostic_gap": metric_gap,
            "interpretation": (
                "The primary answer-only task still has substantial source-row "
                "retrieval misses and wrong top1 sources. The diagnostic variant "
                "is trivialized by indexing the question text and is not a fair "
                "Stage 51 comparison target."
            ),
        },
        "compatibility_gate": {
            "checks": [asdict(check) for check in checks],
            "summary": gate_summary,
        },
        "decision": _decision(gate_summary),
    }


def write_msqa_stage51_compatibility_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[MsqaStage51CompatibilityVisualization]:
    """Write SVG charts for the Stage 59 compatibility report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage59_msqa_stage51_gate_checks.svg": render_horizontal_bar_chart_svg(
            title="Stage 59 MSQA Stage 51 gate checks",
            bars=_gate_bars(report),
            x_label="check count",
            margin_left=180,
        ),
        "stage59_msqa_answer_only_failure_modes.svg": render_horizontal_bar_chart_svg(
            title="Stage 59 MSQA answer-only failure modes",
            bars=_failure_mode_bars(report),
            x_label="sample count",
            margin_left=280,
        ),
        "stage59_msqa_variant_metric_comparison.svg": render_horizontal_bar_chart_svg(
            title="Stage 59 MSQA primary vs diagnostic metrics",
            bars=_metric_bars(report),
            x_label="metric value",
            margin_left=330,
        ),
    }

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            MsqaStage51CompatibilityVisualization(name=filename, path=str(path))
        )
    return artifacts


def _validate_stage58_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 58":
        raise ValueError(f"Expected a Stage 58 report, got: {report.get('stage')!r}")
    required_top_level = {
        "input_contract",
        "baseline_definition",
        "data",
        "variants",
        "decision",
    }
    missing = sorted(required_top_level.difference(report))
    if missing:
        raise ValueError(f"Stage 58 report missing required keys: {missing}")
    if report["baseline_definition"].get("primary_variant") != _PRIMARY_MODE:
        raise ValueError("Stage 58 report primary_variant must be answer_only")
    if report["baseline_definition"].get("diagnostic_variant") != _DIAGNOSTIC_MODE:
        raise ValueError(
            "Stage 58 report diagnostic_variant must be question_answer_page_text"
        )
    if int(report["data"].get("frozen_split_samples", 0)) <= 0:
        raise ValueError("Stage 58 report must contain frozen_split_samples")


def _variant_by_mode(report: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    matches = [
        variant
        for variant in report["variants"]
        if str(variant.get("corpus_mode")) == mode
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Stage 58 variant for mode: {mode}")
    return matches[0]


def _max_hit_k(variant: Mapping[str, Any]) -> int:
    hit_at_k = variant["retrieval_metrics"]["hit_at_k"]
    values = sorted(_parse_hit_key(key) for key in hit_at_k)
    if not values:
        raise ValueError("Stage 58 variant has no hit@k metrics")
    return values[-1]


def _parse_hit_key(key: str) -> int:
    prefix, _, raw_k = key.partition("@")
    if prefix != "hit" or not raw_k:
        raise ValueError(f"Unsupported hit@k key: {key}")
    return int(raw_k)


def _primary_vs_diagnostic_gap(
    *,
    primary: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> dict[str, float]:
    primary_hit = primary["retrieval_metrics"]["hit_at_k"]
    diagnostic_hit = diagnostic["retrieval_metrics"]["hit_at_k"]
    max_k = _max_hit_k(primary)
    max_key = f"hit@{max_k}"
    return {
        "hit@1": round(float(diagnostic_hit["hit@1"]) - float(primary_hit["hit@1"]), 4),
        max_key: round(float(diagnostic_hit[max_key]) - float(primary_hit[max_key]), 4),
        "mrr": round(
            float(diagnostic["retrieval_metrics"]["mrr"])
            - float(primary["retrieval_metrics"]["mrr"]),
            4,
        ),
        "average_top1_token_f1": round(
            float(diagnostic["answer_metrics"]["average_top1_token_f1"])
            - float(primary["answer_metrics"]["average_top1_token_f1"]),
            4,
        ),
    }


def _gate_checks(
    *,
    stage58_report: Mapping[str, Any],
    primary: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    max_k: int,
) -> tuple[CompatibilityGateCheck, ...]:
    primary_hit_at_max = primary["retrieval_metrics"]["hit_at_k"][f"hit@{max_k}"]
    diagnostic_hit_at_1 = diagnostic["retrieval_metrics"]["hit_at_k"]["hit@1"]
    gold_missing_key = f"gold_source_missing_at_{max_k}"
    gold_missing = int(primary["failure_mode_counts"][gold_missing_key])
    top1_wrong = int(primary["failure_mode_counts"]["top1_wrong_source"])
    low_f1 = int(primary["failure_mode_counts"]["top1_token_f1_below_0_3"])
    samples = int(stage58_report["data"]["frozen_split_samples"])
    return (
        CompatibilityGateCheck(
            check_id="frozen_msqa_split_available",
            status=_PASS,
            severity=_INFO,
            evidence=(
                f"{samples} rows are frozen under "
                f"{stage58_report['input_contract']['split_name']}."
            ),
            decision_effect="Allows a Stage 59 compatibility review.",
        ),
        CompatibilityGateCheck(
            check_id="stage58_primary_baseline_recorded",
            status=_PASS,
            severity=_INFO,
            evidence=(
                f"Primary answer_only baseline recorded hit@{max_k}="
                f"{primary_hit_at_max}."
            ),
            decision_effect="Provides the current MSQA baseline reference.",
        ),
        CompatibilityGateCheck(
            check_id="stage51_task_semantics_match_msqa",
            status=_BLOCKED,
            severity=_BLOCKER,
            evidence=(
                "Stage 51 selects PrimeQA document-grounded evidence sentences; "
                "Stage 58 evaluates MSQA answer-source row retrieval."
            ),
            decision_effect="Blocks direct Stage 51 comparison on this task.",
        ),
        CompatibilityGateCheck(
            check_id="citation_identity_contract_match",
            status=_BLOCKED,
            severity=_BLOCKER,
            evidence=(
                "Stage 51 guards document IDs and citation ranks. The frozen MSQA "
                "task uses Q&A row IDs and source URLs, not separate document-span "
                "citation identities."
            ),
            decision_effect="Requires an MSQA source/citation contract before comparison.",
        ),
        CompatibilityGateCheck(
            check_id="candidate_feature_contract_available",
            status=_BLOCKED,
            severity=_BLOCKER,
            evidence=(
                "Stage 51 expects SentenceEvidenceCandidate fields including "
                "candidate score, sentence text, retrieval rank, and document ID. "
                "Stage 58 report contains only source-row BM25 results."
            ),
            decision_effect="Requires an MSQA-compatible candidate construction step.",
        ),
        CompatibilityGateCheck(
            check_id="diagnostic_variant_usable_for_comparison",
            status=_BLOCKED,
            severity=_BLOCKER,
            evidence=(
                f"The diagnostic question_answer_page_text variant has hit@1="
                f"{diagnostic_hit_at_1}, because it indexes the question text."
            ),
            decision_effect="Rejects diagnostic metrics as a Stage 51 comparison target.",
        ),
        CompatibilityGateCheck(
            check_id="failure_modes_are_policy_test_ready",
            status=_BLOCKED,
            severity=_BLOCKER,
            evidence=(
                f"answer_only has {gold_missing} gold-source misses at {max_k}, "
                f"{top1_wrong} wrong top1 sources, and {low_f1} low-F1 top1 answers."
            ),
            decision_effect=(
                "Requires MSQA retrieval/candidate protocol review before a "
                "composition-policy experiment."
            ),
        ),
    )


def _gate_summary(
    checks: Sequence[CompatibilityGateCheck],
) -> dict[str, Any]:
    status_counts = Counter(check.status for check in checks)
    blocker_checks = [
        check.check_id
        for check in checks
        if check.status == _BLOCKED and check.severity == _BLOCKER
    ]
    return {
        "total_checks": len(checks),
        "status_counts": dict(sorted(status_counts.items())),
        "blocker_count": len(blocker_checks),
        "blocker_checks": blocker_checks,
    }


def _decision(gate_summary: Mapping[str, Any]) -> dict[str, Any]:
    blocked = int(gate_summary["blocker_count"]) > 0
    return {
        "status": "stage51_msqa_compatibility_blocked"
        if blocked
        else "stage51_msqa_compatibility_passed",
        "can_run_stage51_candidate_now": not blocked,
        "can_defaultize_runtime_now": False,
        "default_runtime_policy": "unchanged",
        "rejected_comparison_variant": _DIAGNOSTIC_MODE,
        "required_before_stage51_comparison": [
            "Define an MSQA-compatible source/citation identity contract.",
            (
                "Construct MSQA evidence candidates with sentence, score, "
                "retrieval rank, and source identity."
            ),
            "Freeze a comparison protocol that does not use the diagnostic question-text index.",
            "Rerun the baseline and candidate under the same MSQA-compatible contract.",
        ],
        "recommended_next_stage": (
            "Stage 60: design the MSQA source/citation adapter and comparison "
            "protocol before any Stage 51 candidate run"
        ),
        "reason": (
            "Direct Stage 51 comparison is blocked because the current MSQA "
            "baseline is an answer-source row retrieval task, while Stage 51 is "
            "a PrimeQA document-grounded evidence composition policy."
        ),
    }


def _gate_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = report["compatibility_gate"]["summary"]["status_counts"]
    ordered_statuses = (_PASS, _BLOCKED)
    return [
        BarDatum(
            label=status,
            value=float(counts.get(status, 0)),
            value_label=str(counts.get(status, 0)),
        )
        for status in ordered_statuses
    ]


def _failure_mode_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = report["failure_mode_review"]["primary_failure_counts"]
    rates = report["failure_mode_review"]["primary_failure_rates"]
    return [
        BarDatum(
            label=key,
            value=float(value),
            value_label=f"{value} ({rates[key]:.1%})",
        )
        for key, value in counts.items()
    ]


def _metric_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    review = report["failure_mode_review"]
    primary_retrieval = review["primary_retrieval_metrics"]
    diagnostic_retrieval = review["diagnostic_retrieval_metrics"]
    primary_answer = review["primary_answer_metrics"]
    diagnostic_answer = review["diagnostic_answer_metrics"]
    max_k = report["stage58_baseline_summary"]["max_k"]
    metric_rows = (
        ("answer_only hit@1", primary_retrieval["hit_at_k"]["hit@1"]),
        (f"answer_only hit@{max_k}", primary_retrieval["hit_at_k"][f"hit@{max_k}"]),
        ("answer_only MRR", primary_retrieval["mrr"]),
        ("answer_only top1 F1", primary_answer["average_top1_token_f1"]),
        ("diagnostic hit@1", diagnostic_retrieval["hit_at_k"]["hit@1"]),
        (f"diagnostic hit@{max_k}", diagnostic_retrieval["hit_at_k"][f"hit@{max_k}"]),
        ("diagnostic MRR", diagnostic_retrieval["mrr"]),
        ("diagnostic top1 F1", diagnostic_answer["average_top1_token_f1"]),
    )
    return [
        BarDatum(label=label, value=float(value), value_label=str(value))
        for label, value in metric_rows
    ]


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
