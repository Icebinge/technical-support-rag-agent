from __future__ import annotations

import hashlib
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol as stage160,
)
from ts_rag_agent.application.evidence_selection import tokenize_text
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    _fingerprint,
    _public_safe_contract,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    StructuredRouterPromptPolicy,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 164"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_gold_visible_agent_refusal_diagnostics_v1"
_PROTOCOL_ID = "primeqa_hybrid_stage164_gold_visible_refusal_protocol_v1"
_EXPECTED_DEV_ROWS = 121
_EXPECTED_GOLD_VISIBLE_ROWS = 36
_EXPECTED_GOLD_VISIBLE_REFUSALS = 19
_EXPECTED_GOLD_VISIBLE_ANSWERS = 17
_EXPECTED_FOLD_COUNT = 5
_EXPECTED_SOURCE_HASHES = {
    "stage163_correction": "f31efd39fc87f3c9289d2cc2521d0928e283a2535418565cf6d1d668565da15b",
    "stage160_public": "e17e5fe5bbc5fef4e25e41234e47b89daf19ea4ef18f3c7270601f0fee7d9377",
    "stage160_hashed": "3f10cffe245a4405dfc56044f2a3c0d364fdd0f8723e6cc3ae401260199652db",
    "dev": "071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
    "router_source": "d9eeaff5fbb9c97a689efdee72d17f699cce47d1c94361047a74c90906442195",
}
_STAGE160_PRIVATE_CANONICAL_SHA256 = (
    "1c8aa4260be5427e13322cb3304e518dd3609c2e38f839cda4f10ce01c911a0d"
)
_STAGE163_CORRECTION_STATUS = "primeqa_hybrid_stage163_contract_correction_completed"
_STAGE163_RESULT_STATUS = "primeqa_hybrid_untouched_rrf_not_dev_safe"
_STAGE160_STATUS = "primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_completed"

ProgressSink = Callable[[Mapping[str, Any]], None]
RiskDirection = Literal["higher", "lower"]


@dataclass(frozen=True)
class GoldVisibleRefusalCaseProfile:
    """Hashed, content-free features for one Stage160 gold-document-visible case."""

    private_identity_sha256: str
    diagnostic_group_sha256: str
    fold_id: int
    refused: bool
    question_route: str
    split_subtype: str
    gold_generation_rank: int
    gold_candidate_rank: int
    gold_verification_rank: int | None
    top_candidate_score: float
    gold_candidate_score: float
    gold_score_ratio_to_top: float
    router_input_token_count: int
    retained_state_bytes: int
    turn_position: int
    completed_turn_count: int
    router_generation_latency_ms: float
    answer_token_count: int
    gold_document_length_chars: int
    gold_excerpt_truncated: bool
    answer_found_in_full_document: bool
    answer_exact_span_visible: bool
    answer_all_tokens_visible: bool
    answer_token_recall_visible: float
    question_token_recall_in_gold_prompt: float
    answer_character_start: int | None
    answer_visibility_class: str


@dataclass(frozen=True)
class PrimeQAHybridGoldVisibleRefusalDiagnosticsRun:
    public_report: dict[str, Any]
    private_report: dict[str, Any]


@dataclass(frozen=True)
class PrimeQAHybridGoldVisibleRefusalVisualization:
    name: str
    path: str


class GoldVisibleRefusalAnalyzer:
    """Join frozen Stage160 observations to the exact router prompt visibility contract."""

    def __init__(self, *, prompt_policy: StructuredRouterPromptPolicy) -> None:
        self._prompt_policy = prompt_policy

    def build_profiles(
        self,
        *,
        diagnostic_samples: Sequence[stage160.Stage160DiagnosticSample],
        stage160_rows: Sequence[Mapping[str, Any]],
        documents_by_id: Mapping[str, PrimeQADocument],
    ) -> tuple[GoldVisibleRefusalCaseProfile, ...]:
        rows_by_identity = {str(row["private_identity_sha256"]): row for row in stage160_rows}
        if len(rows_by_identity) != len(stage160_rows):
            raise ValueError("Stage164 rejects duplicate Stage160 hashed identities")
        sample_identities = {sample.private_identity_sha256 for sample in diagnostic_samples}
        if sample_identities != set(rows_by_identity):
            raise ValueError("Stage164 Stage160/dev hashed identity join is not exact")

        profiles = []
        for sample in diagnostic_samples:
            row = rows_by_identity[sample.private_identity_sha256]
            if not sample.answerable or row.get("gold_generation_rank") is None:
                continue
            if not sample.gold_document_id:
                raise ValueError("Stage164 gold-visible sample has no gold document")
            document = documents_by_id.get(sample.gold_document_id)
            if document is None:
                raise ValueError("Stage164 gold-visible document is absent from the corpus")
            profiles.append(self._profile(sample=sample, row=row, document=document))
        return tuple(sorted(profiles, key=lambda item: item.private_identity_sha256))

    def _profile(
        self,
        *,
        sample: stage160.Stage160DiagnosticSample,
        row: Mapping[str, Any],
        document: PrimeQADocument,
    ) -> GoldVisibleRefusalCaseProfile:
        excerpt = document.text[: self._prompt_policy.max_evidence_chars_per_result]
        prompt_evidence = f"{document.title}\n{excerpt}"
        answer_tokens = tokenize_text(sample.gold_answer)
        prompt_tokens = tokenize_text(prompt_evidence)
        question_tokens = tokenize_text(sample.runtime_query.full_question)
        answer_recall = _multiset_recall(answer_tokens, prompt_tokens)
        question_recall = _multiset_recall(question_tokens, prompt_tokens)
        normalized_answer = _normalized_text(sample.gold_answer)
        normalized_prompt = _normalized_text(prompt_evidence)
        normalized_document = _normalized_text(document.text)
        exact_visible = bool(normalized_answer and normalized_answer in normalized_prompt)
        answer_found = bool(normalized_answer and normalized_answer in normalized_document)
        answer_start = normalized_document.find(normalized_answer) if answer_found else None
        all_tokens_visible = bool(answer_tokens) and answer_recall >= 1.0 - 1e-12
        top_score = float(row["top_candidate_score"])
        gold_score = float(row["gold_candidate_score"])
        return GoldVisibleRefusalCaseProfile(
            private_identity_sha256=sample.private_identity_sha256,
            diagnostic_group_sha256=sample.diagnostic_group_sha256,
            fold_id=int(row["fold_id"]),
            refused=bool(row["refused"]),
            question_route=str(row["question_route"]),
            split_subtype=str(row["split_subtype"]),
            gold_generation_rank=int(row["gold_generation_rank"]),
            gold_candidate_rank=int(row["gold_candidate_rank"]),
            gold_verification_rank=(
                int(row["gold_verification_rank"])
                if row.get("gold_verification_rank") is not None
                else None
            ),
            top_candidate_score=top_score,
            gold_candidate_score=gold_score,
            gold_score_ratio_to_top=round(gold_score / top_score if top_score else 0.0, 6),
            router_input_token_count=int(row["router_input_token_count"]),
            retained_state_bytes=int(row["retained_state_bytes"]),
            turn_position=int(row["turn_position"]),
            completed_turn_count=int(row["completed_turn_count"]),
            router_generation_latency_ms=float(row["router_generation_latency_ms"]),
            answer_token_count=len(answer_tokens),
            gold_document_length_chars=len(document.text),
            gold_excerpt_truncated=len(document.text)
            > self._prompt_policy.max_evidence_chars_per_result,
            answer_found_in_full_document=answer_found,
            answer_exact_span_visible=exact_visible,
            answer_all_tokens_visible=all_tokens_visible,
            answer_token_recall_visible=answer_recall,
            question_token_recall_in_gold_prompt=question_recall,
            answer_character_start=answer_start,
            answer_visibility_class=_visibility_class(
                exact_visible=exact_visible,
                all_tokens_visible=all_tokens_visible,
                answer_recall=answer_recall,
            ),
        )


def run_primeqa_hybrid_gold_visible_refusal_diagnostics(
    *,
    stage163_correction_path: Path,
    stage160_report_path: Path,
    stage160_hashed_report_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    router_source_path: Path,
    user_confirmed_diagnostics: bool,
    confirmation_note: str,
    progress_sink: ProgressSink | None = None,
) -> PrimeQAHybridGoldVisibleRefusalDiagnosticsRun:
    """Analyze existing Stage160 outcomes without rerunning Agent or loading test."""

    source_authorization = _authorize_sources(
        stage163_correction_path=stage163_correction_path,
        stage160_report_path=stage160_report_path,
        stage160_hashed_report_path=stage160_hashed_report_path,
        dev_split_path=dev_split_path,
        documents_path=documents_path,
        router_source_path=router_source_path,
    )
    protocol = _frozen_protocol()
    protocol_sha256 = stage160.canonical_json_sha256(protocol)
    _emit(progress_sink, phase="sources_authorized")

    diagnostic_set = stage160.load_stage160_dev_diagnostic_samples(dev_split_path)
    stage160_hashed = _load_json_object(stage160_hashed_report_path)
    documents_by_id = load_primeqa_documents(documents_path)
    prompt_policy = StructuredRouterPromptPolicy()
    _emit(
        progress_sink,
        phase="existing_diagnostics_and_documents_loaded",
        dev_rows=len(diagnostic_set.samples),
    )

    profiles = GoldVisibleRefusalAnalyzer(prompt_policy=prompt_policy).build_profiles(
        diagnostic_samples=diagnostic_set.samples,
        stage160_rows=stage160_hashed["rows"],
        documents_by_id=documents_by_id,
    )
    private_report = _private_report(profiles)
    analysis = _analyze_profiles(profiles)
    _emit(progress_sink, phase="gold_visible_profiles_analyzed", cohort_rows=len(profiles))

    public_report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Diagnostic-only analysis of the 36 existing Stage160 answerable cases whose "
            "gold document reached generation Top10. It separates document visibility from "
            "answer-evidence visibility under the exact 10-document, 600-character router "
            "prompt contract. No Agent inference, retrieval, fitting, tuning, policy selection, "
            "test evaluation, runtime registration, or fallback is performed."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_diagnostics),
            "confirmation_note": confirmation_note,
        },
        "source_authorization": source_authorization,
        "frozen_protocol": protocol,
        "frozen_protocol_sha256": protocol_sha256,
        "split_contract": {
            "loaded_split": "dev_diagnostic_join_only",
            "train_rows_loaded": 0,
            "dev_rows_loaded": len(diagnostic_set.samples),
            "test_rows_loaded": 0,
            "existing_agent_outcomes_reused": True,
            "agent_inference_rerun": False,
            "retrieval_rerun": False,
            "dev_used_for_fit_selection_or_tuning": False,
            "test_metrics_run": False,
        },
        "router_prompt_contract": {
            "max_evidence_results": prompt_policy.max_evidence_results,
            "max_evidence_chars_per_result": prompt_policy.max_evidence_chars_per_result,
            "max_input_tokens": prompt_policy.max_input_tokens,
            "max_new_tokens": prompt_policy.max_new_tokens,
            "evidence_payload_fields": ["rank", "retrieval_score", "title", "excerpt"],
            "excerpt_policy": "document_text_prefix_only",
            "gold_document_in_top10_does_not_prove_answer_span_in_prompt": True,
        },
        "cohort_summary": analysis["cohort_summary"],
        "answer_visibility_summary": analysis["answer_visibility_summary"],
        "fixed_binary_associations": analysis["fixed_binary_associations"],
        "fixed_numeric_associations": analysis["fixed_numeric_associations"],
        "question_route_summary": analysis["question_route_summary"],
        "fold_stability": analysis["fold_stability"],
        "exploratory_feature_ranking": analysis["exploratory_feature_ranking"],
        "primary_hypothesis_assessment": analysis["primary_hypothesis_assessment"],
        "private_feature_artifact_contract": {
            "canonical_content_sha256": stage160.canonical_json_sha256(private_report),
            "row_count": len(profiles),
            "contains_hashed_sample_identity": True,
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "public_report_contains_case_rows": False,
            "git_policy": "ignored_local_artifact",
        },
        "closed_boundaries": {
            "agent_model_loaded": False,
            "agent_inference_run": False,
            "retrieval_run": False,
            "policy_fit": False,
            "threshold_tuned": False,
            "policy_selected": False,
            "test_loaded": False,
            "runtime_registered_as_default": False,
            "fallback_strategies_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
        },
    }
    public_report["guard_checks"] = _guard_checks(
        report=public_report,
        profiles=profiles,
        stage160_hashed=stage160_hashed,
    )
    public_report["public_safe_contract"] = _public_safe_contract(public_report)
    all_guards_passed = all(check["passed"] for check in public_report["guard_checks"])
    public_report["decision"] = _decision(
        report=public_report,
        all_guards_passed=all_guards_passed,
    )
    return PrimeQAHybridGoldVisibleRefusalDiagnosticsRun(
        public_report=public_report,
        private_report=private_report,
    )


def write_stage164_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridGoldVisibleRefusalVisualization]:
    """Write ten aggregate, public-safe Stage164 SVG diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = report["cohort_summary"]
    visibility = report["answer_visibility_summary"]
    binary = report["fixed_binary_associations"]
    numeric = report["fixed_numeric_associations"]
    fold = report["fold_stability"]
    chart_specs = {
        "stage164_gold_visible_outcomes.svg": (
            "Stage164 gold-document-visible Agent outcomes",
            [_bar("refused", cohort["refused_count"]), _bar("answered", cohort["answered_count"])],
        ),
        "stage164_visibility_classes.svg": (
            "Stage164 answer-evidence visibility classes",
            [
                _bar(label, values["case_count"])
                for label, values in visibility["by_visibility_class"].items()
            ],
        ),
        "stage164_visibility_refusal_rates.svg": (
            "Stage164 refusal rate by answer-evidence visibility",
            [
                _bar(label, values["refusal_rate"])
                for label, values in visibility["by_visibility_class"].items()
            ],
        ),
        "stage164_binary_risk_differences.svg": (
            "Stage164 fixed binary refusal-risk differences",
            [
                _bar(name, values["refusal_rate_difference_risk_minus_reference"])
                for name, values in binary.items()
            ],
        ),
        "stage164_numeric_risk_auc.svg": (
            "Stage164 fixed numeric risk-aligned AUC",
            [_bar(name, values["risk_aligned_auc"]) for name, values in numeric.items()],
        ),
        "stage164_answer_visibility_medians.svg": (
            "Stage164 median answer-token visibility",
            _group_median_bars(numeric, "answer_token_recall_visible"),
        ),
        "stage164_question_alignment_medians.svg": (
            "Stage164 median question-to-gold-prompt alignment",
            _group_median_bars(numeric, "question_token_recall_in_gold_prompt"),
        ),
        "stage164_gold_rank_medians.svg": (
            "Stage164 median gold generation rank",
            _group_median_bars(numeric, "gold_generation_rank"),
        ),
        "stage164_fold_visibility_risk.svg": (
            "Stage164 fold refusal-risk difference when exact answer span is absent",
            [
                _bar(fold_id, values["refusal_rate_difference_risk_minus_reference"])
                for fold_id, values in fold["answer_exact_span_visible"]["folds"].items()
                if values["comparable"]
            ],
        ),
        "stage164_guard_status.svg": (
            "Stage164 process guard status",
            [_bar(check["name"], check["passed"]) for check in report["guard_checks"]],
        ),
    }
    artifacts = []
    for filename, (title, bars) in chart_specs.items():
        path = output_dir / filename
        path.write_text(
            render_horizontal_bar_chart_svg(title=title, bars=bars, x_label="value"),
            encoding="utf-8",
        )
        artifacts.append(
            PrimeQAHybridGoldVisibleRefusalVisualization(name=filename, path=str(path))
        )
    return artifacts


def _analyze_profiles(
    profiles: Sequence[GoldVisibleRefusalCaseProfile],
) -> dict[str, Any]:
    refused = [profile for profile in profiles if profile.refused]
    answered = [profile for profile in profiles if not profile.refused]
    binary_specs = {
        "answer_exact_span_visible": ("answer_exact_span_visible", False),
        "answer_all_tokens_visible": ("answer_all_tokens_visible", False),
        "answer_found_in_full_document": ("answer_found_in_full_document", False),
        "gold_excerpt_truncated": ("gold_excerpt_truncated", True),
        "turn_position_after_first": ("turn_position_after_first", True),
    }
    numeric_specs: dict[str, tuple[str, RiskDirection]] = {
        "answer_token_recall_visible": ("answer_token_recall_visible", "lower"),
        "question_token_recall_in_gold_prompt": (
            "question_token_recall_in_gold_prompt",
            "lower",
        ),
        "gold_generation_rank": ("gold_generation_rank", "higher"),
        "gold_candidate_rank": ("gold_candidate_rank", "higher"),
        "gold_score_ratio_to_top": ("gold_score_ratio_to_top", "lower"),
        "router_input_token_count": ("router_input_token_count", "higher"),
        "retained_state_bytes": ("retained_state_bytes", "higher"),
        "turn_position": ("turn_position", "higher"),
        "answer_token_count": ("answer_token_count", "higher"),
        "gold_document_length_chars": ("gold_document_length_chars", "higher"),
    }
    binary = {
        name: _binary_association(profiles, attribute=attribute, risk_value=risk_value)
        for name, (attribute, risk_value) in binary_specs.items()
    }
    numeric = {
        name: _numeric_association(
            refused=refused,
            answered=answered,
            attribute=attribute,
            risk_direction=risk_direction,
        )
        for name, (attribute, risk_direction) in numeric_specs.items()
    }
    fold_stability = {
        name: _binary_fold_stability(
            profiles,
            attribute=attribute,
            risk_value=risk_value,
        )
        for name, (attribute, risk_value) in binary_specs.items()
    }
    return {
        "cohort_summary": {
            "gold_document_visible_count": len(profiles),
            "refused_count": len(refused),
            "answered_count": len(answered),
            "refusal_rate": _ratio(len(refused), len(profiles)),
            "fold_count": len({profile.fold_id for profile in profiles}),
            "diagnostic_group_count": len(
                {profile.diagnostic_group_sha256 for profile in profiles}
            ),
            "case_rows_written_in_public_report": False,
        },
        "answer_visibility_summary": {
            "prompt_excerpt_chars_per_document": 600,
            "by_visibility_class": _group_outcomes(
                profiles,
                key=lambda item: item.answer_visibility_class,
            ),
            "exact_span_visible_count": sum(
                profile.answer_exact_span_visible for profile in profiles
            ),
            "all_answer_tokens_visible_count": sum(
                profile.answer_all_tokens_visible for profile in profiles
            ),
            "answer_found_in_full_document_count": sum(
                profile.answer_found_in_full_document for profile in profiles
            ),
            "document_truncated_count": sum(profile.gold_excerpt_truncated for profile in profiles),
        },
        "fixed_binary_associations": binary,
        "fixed_numeric_associations": numeric,
        "question_route_summary": _group_outcomes(
            profiles,
            key=lambda item: item.question_route,
        ),
        "fold_stability": fold_stability,
        "exploratory_feature_ranking": [
            {
                "feature": name,
                "risk_aligned_auc": values["risk_aligned_auc"],
                "distance_from_random": round(abs(values["risk_aligned_auc"] - 0.5), 6),
            }
            for name, values in sorted(
                numeric.items(),
                key=lambda item: (-abs(item[1]["risk_aligned_auc"] - 0.5), item[0]),
            )
        ],
        "primary_hypothesis_assessment": _primary_hypothesis_assessment(
            binary_associations=binary,
            fold_stability=fold_stability,
        ),
    }


def _primary_hypothesis_assessment(
    *,
    binary_associations: Mapping[str, Mapping[str, Any]],
    fold_stability: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    primary_exact = binary_associations["answer_exact_span_visible"]
    primary_tokens = binary_associations["answer_all_tokens_visible"]
    exact_fold = fold_stability["answer_exact_span_visible"]
    aggregate_observed = (
        float(primary_exact["refusal_rate_difference_risk_minus_reference"]) > 0.0
        or float(primary_tokens["refusal_rate_difference_risk_minus_reference"]) > 0.0
    )
    fold_stable = bool(
        aggregate_observed
        and int(exact_fold["risk_direction_fold_count"])
        > int(exact_fold["opposite_direction_fold_count"])
    )
    return {
        "hypothesis": (
            "gold_document_in_generation_top10_can_still_lack_gold_answer_evidence_in_"
            "the_fixed_600_character_prompt_excerpt"
        ),
        "aggregate_visibility_gap_observed": aggregate_observed,
        "fold_stable_visibility_gap_observed": fold_stable,
        "exact_span_absence_refusal_rate_difference": primary_exact[
            "refusal_rate_difference_risk_minus_reference"
        ],
        "all_token_absence_refusal_rate_difference": primary_tokens[
            "refusal_rate_difference_risk_minus_reference"
        ],
        "exact_span_fold_direction_count": exact_fold["risk_direction_fold_count"],
        "exact_span_fold_opposite_direction_count": exact_fold["opposite_direction_fold_count"],
        "exact_span_comparable_fold_count": exact_fold["comparable_fold_count"],
        "causal_claim": False,
        "policy_selected": False,
    }


def _binary_association(
    profiles: Sequence[GoldVisibleRefusalCaseProfile],
    *,
    attribute: str,
    risk_value: bool,
) -> dict[str, Any]:
    risk = [profile for profile in profiles if _binary_value(profile, attribute) is risk_value]
    reference = [
        profile for profile in profiles if _binary_value(profile, attribute) is not risk_value
    ]
    risk_refused = sum(profile.refused for profile in risk)
    reference_refused = sum(profile.refused for profile in reference)
    risk_rate = _ratio(risk_refused, len(risk))
    reference_rate = _ratio(reference_refused, len(reference))
    return {
        "risk_value": risk_value,
        "risk_count": len(risk),
        "risk_refused_count": risk_refused,
        "risk_refusal_rate": risk_rate,
        "reference_count": len(reference),
        "reference_refused_count": reference_refused,
        "reference_refusal_rate": reference_rate,
        "refusal_rate_difference_risk_minus_reference": round(risk_rate - reference_rate, 6),
        "haldane_corrected_odds_ratio": _haldane_odds_ratio(
            risk_refused=risk_refused,
            risk_answered=len(risk) - risk_refused,
            reference_refused=reference_refused,
            reference_answered=len(reference) - reference_refused,
        ),
    }


def _numeric_association(
    *,
    refused: Sequence[GoldVisibleRefusalCaseProfile],
    answered: Sequence[GoldVisibleRefusalCaseProfile],
    attribute: str,
    risk_direction: RiskDirection,
) -> dict[str, Any]:
    refused_values = [float(getattr(profile, attribute)) for profile in refused]
    answered_values = [float(getattr(profile, attribute)) for profile in answered]
    return {
        "risk_direction": risk_direction,
        "refused": _distribution(refused_values),
        "answered": _distribution(answered_values),
        "median_difference_refused_minus_answered": round(
            statistics.median(refused_values) - statistics.median(answered_values),
            6,
        ),
        "risk_aligned_auc": _risk_aligned_auc(
            refused_values,
            answered_values,
            risk_direction=risk_direction,
        ),
    }


def _binary_fold_stability(
    profiles: Sequence[GoldVisibleRefusalCaseProfile],
    *,
    attribute: str,
    risk_value: bool,
) -> dict[str, Any]:
    folds = {}
    for fold_id in sorted({profile.fold_id for profile in profiles}):
        fold_profiles = [profile for profile in profiles if profile.fold_id == fold_id]
        risk = [
            profile for profile in fold_profiles if _binary_value(profile, attribute) is risk_value
        ]
        reference = [
            profile
            for profile in fold_profiles
            if _binary_value(profile, attribute) is not risk_value
        ]
        comparable = bool(risk and reference)
        risk_rate = _ratio(sum(profile.refused for profile in risk), len(risk))
        reference_rate = _ratio(sum(profile.refused for profile in reference), len(reference))
        folds[f"fold_{fold_id + 1}"] = {
            "case_count": len(fold_profiles),
            "risk_count": len(risk),
            "reference_count": len(reference),
            "comparable": comparable,
            "refusal_rate_difference_risk_minus_reference": (
                round(risk_rate - reference_rate, 6) if comparable else 0.0
            ),
        }
    comparable_deltas = [
        values["refusal_rate_difference_risk_minus_reference"]
        for values in folds.values()
        if values["comparable"]
    ]
    return {
        "folds": folds,
        "comparable_fold_count": len(comparable_deltas),
        "risk_direction_fold_count": sum(delta > 0.0 for delta in comparable_deltas),
        "tie_fold_count": sum(delta == 0.0 for delta in comparable_deltas),
        "opposite_direction_fold_count": sum(delta < 0.0 for delta in comparable_deltas),
        "minimum_comparable_fold_delta": (min(comparable_deltas) if comparable_deltas else 0.0),
        "maximum_comparable_fold_delta": (max(comparable_deltas) if comparable_deltas else 0.0),
    }


def _group_outcomes(
    profiles: Sequence[GoldVisibleRefusalCaseProfile],
    *,
    key: Callable[[GoldVisibleRefusalCaseProfile], str],
) -> dict[str, Any]:
    groups: dict[str, list[GoldVisibleRefusalCaseProfile]] = defaultdict(list)
    for profile in profiles:
        groups[key(profile)].append(profile)
    return {
        label: {
            "case_count": len(rows),
            "refused_count": sum(row.refused for row in rows),
            "answered_count": sum(not row.refused for row in rows),
            "refusal_rate": _ratio(sum(row.refused for row in rows), len(rows)),
        }
        for label, rows in sorted(groups.items())
    }


def _private_report(
    profiles: Sequence[GoldVisibleRefusalCaseProfile],
) -> dict[str, Any]:
    return {
        "stage": _STAGE,
        "privacy_class": "ignored_hashed_numeric_features",
        "contains_raw_question": False,
        "contains_raw_answer": False,
        "contains_raw_document_id": False,
        "contains_raw_document_text": False,
        "contains_hashed_sample_identity": True,
        "row_count": len(profiles),
        "rows": [asdict(profile) for profile in profiles],
    }


def _authorize_sources(
    *,
    stage163_correction_path: Path,
    stage160_report_path: Path,
    stage160_hashed_report_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    router_source_path: Path,
) -> dict[str, Any]:
    paths = {
        "stage163_correction": stage163_correction_path,
        "stage160_public": stage160_report_path,
        "stage160_hashed": stage160_hashed_report_path,
        "dev": dev_split_path,
        "documents": documents_path,
        "router_source": router_source_path,
    }
    fingerprints = {name: _fingerprint(path) for name, path in paths.items()}
    mismatches = {
        name: fingerprints[name]["sha256"]
        for name, expected in _EXPECTED_SOURCE_HASHES.items()
        if fingerprints[name]["sha256"] != expected
    }
    if mismatches:
        raise ValueError(f"Stage164 source fingerprint mismatch: {mismatches}")
    stage163 = _load_json_object(stage163_correction_path)
    stage160_report = _load_json_object(stage160_report_path)
    stage160_hashed = _load_json_object(stage160_hashed_report_path)
    if stage163.get("decision", {}).get("status") != _STAGE163_CORRECTION_STATUS:
        raise ValueError("Stage164 requires the completed Stage163 correction")
    if stage163.get("decision", {}).get("corrected_stage163_status") != (_STAGE163_RESULT_STATUS):
        raise ValueError("Stage164 requires the Stage163 rejected context policy")
    if stage163.get("decision", {}).get("next_direction") != (
        "stop_context_policy_changes_and_analyze_gold_visible_refusals"
    ):
        raise ValueError("Stage164 requires the frozen gold-visible-refusal direction")
    if stage160_report.get("decision", {}).get("status") != _STAGE160_STATUS:
        raise ValueError("Stage164 requires completed Stage160 diagnostics")
    declared_private_sha = stage160_report.get("private_diagnostic_artifact_contract", {}).get(
        "canonical_content_sha256"
    )
    actual_private_sha = stage160.canonical_json_sha256(stage160_hashed)
    if (
        declared_private_sha != _STAGE160_PRIVATE_CANONICAL_SHA256
        or actual_private_sha != declared_private_sha
    ):
        raise ValueError("Stage164 Stage160 hashed diagnostic content mismatch")
    return {
        "fingerprints": fingerprints,
        "stage163_correction_status": _STAGE163_CORRECTION_STATUS,
        "stage163_corrected_result": _STAGE163_RESULT_STATUS,
        "stage160_status": _STAGE160_STATUS,
        "stage160_hashed_canonical_sha256": actual_private_sha,
        "all_sources_authorized_before_dev_join": True,
    }


def _frozen_protocol() -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "cohort": ("stage160_answerable_rows_with_gold_generation_rank_1_through_10"),
        "outcome": "stage160_existing_router_refused_boolean",
        "cohort_expected_count": _EXPECTED_GOLD_VISIBLE_ROWS,
        "refused_expected_count": _EXPECTED_GOLD_VISIBLE_REFUSALS,
        "answered_expected_count": _EXPECTED_GOLD_VISIBLE_ANSWERS,
        "primary_hypothesis": (
            "gold_document_visibility_does_not_imply_answer_evidence_visibility_under_"
            "the_router_600_character_excerpt"
        ),
        "fixed_feature_families": [
            "answer_evidence_visibility",
            "gold_retrieval_position_and_score",
            "question_to_gold_prompt_lexical_alignment",
            "prompt_history_load",
        ],
        "fixed_binary_features": [
            "answer_exact_span_visible",
            "answer_all_tokens_visible",
            "answer_found_in_full_document",
            "gold_excerpt_truncated",
            "turn_position_after_first",
        ],
        "fixed_numeric_features": [
            "answer_token_recall_visible",
            "question_token_recall_in_gold_prompt",
            "gold_generation_rank",
            "gold_candidate_rank",
            "gold_score_ratio_to_top",
            "router_input_token_count",
            "retained_state_bytes",
            "turn_position",
            "answer_token_count",
            "gold_document_length_chars",
        ],
        "grouped_fold_role": "direction_stability_only_no_fit_or_selection",
        "causal_claim_allowed": False,
        "blocked": {
            "agent_rerun": True,
            "retrieval_rerun": True,
            "model_fit": True,
            "threshold_tuning": True,
            "policy_selection": True,
            "test_load": True,
            "runtime_defaultization": True,
            "fallback": True,
            "query_rewrite": True,
            "second_retrieval": True,
        },
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    profiles: Sequence[GoldVisibleRefusalCaseProfile],
    stage160_hashed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = report["source_authorization"]
    split = report["split_contract"]
    prompt = report["router_prompt_contract"]
    cohort = report["cohort_summary"]
    private = report["private_feature_artifact_contract"]
    boundaries = report["closed_boundaries"]
    group_folds: dict[str, set[int]] = defaultdict(set)
    for profile in profiles:
        group_folds[profile.diagnostic_group_sha256].add(profile.fold_id)
    return [
        _check("user_confirmed_stage164", report["user_confirmation"]["confirmed"] is True),
        _check(
            "frozen_protocol_identity_exact",
            report["frozen_protocol_sha256"]
            == stage160.canonical_json_sha256(report["frozen_protocol"]),
        ),
        _check(
            "stage163_direction_exact",
            source["stage163_corrected_result"] == _STAGE163_RESULT_STATUS,
        ),
        _check(
            "stage160_sources_exact",
            source["fingerprints"]["stage160_public"]["sha256"]
            == _EXPECTED_SOURCE_HASHES["stage160_public"]
            and source["fingerprints"]["stage160_hashed"]["sha256"]
            == _EXPECTED_SOURCE_HASHES["stage160_hashed"]
            and source["stage160_hashed_canonical_sha256"] == _STAGE160_PRIVATE_CANONICAL_SHA256,
        ),
        _check(
            "dev_and_documents_exact",
            source["fingerprints"]["dev"]["sha256"] == _EXPECTED_SOURCE_HASHES["dev"]
            and source["fingerprints"]["documents"]["sha256"]
            == _EXPECTED_SOURCE_HASHES["documents"],
        ),
        _check(
            "router_source_and_prompt_contract_exact",
            source["fingerprints"]["router_source"]["sha256"]
            == _EXPECTED_SOURCE_HASHES["router_source"]
            and prompt["max_evidence_results"] == 10
            and prompt["max_evidence_chars_per_result"] == 600,
        ),
        _check(
            "only_diagnostic_dev_join_loaded",
            split["train_rows_loaded"] == 0
            and split["dev_rows_loaded"] == _EXPECTED_DEV_ROWS
            and split["test_rows_loaded"] == 0,
        ),
        _check(
            "existing_stage160_outcomes_only",
            split["existing_agent_outcomes_reused"] is True
            and split["agent_inference_rerun"] is False
            and split["retrieval_rerun"] is False,
        ),
        _check(
            "stage160_hashed_rows_exact",
            stage160_hashed.get("row_count") == _EXPECTED_DEV_ROWS
            and len(stage160_hashed.get("rows", [])) == _EXPECTED_DEV_ROWS,
        ),
        _check(
            "gold_visible_cohort_exact",
            cohort["gold_document_visible_count"] == _EXPECTED_GOLD_VISIBLE_ROWS
            and cohort["refused_count"] == _EXPECTED_GOLD_VISIBLE_REFUSALS
            and cohort["answered_count"] == _EXPECTED_GOLD_VISIBLE_ANSWERS,
        ),
        _check(
            "gold_generation_context_membership_exact",
            len(
                [
                    row
                    for row in stage160_hashed["rows"]
                    if row["answerable"] and row["gold_generation_rank"] is not None
                ]
            )
            == _EXPECTED_GOLD_VISIBLE_ROWS
            and all(profile.gold_generation_rank >= 1 for profile in profiles)
            and all(
                row["generation_context_count"] == 10
                for row in stage160_hashed["rows"]
                if row["answerable"] and row["gold_generation_rank"] is not None
            ),
        ),
        _check(
            "grouped_five_fold_isolation",
            cohort["fold_count"] == _EXPECTED_FOLD_COUNT
            and all(len(folds) == 1 for folds in group_folds.values()),
        ),
        _check(
            "private_feature_rows_public_safe",
            private["row_count"] == _EXPECTED_GOLD_VISIBLE_ROWS
            and private["contains_raw_question"] is False
            and private["contains_raw_answer"] is False
            and private["contains_raw_document_id"] is False
            and private["contains_raw_document_text"] is False
            and private["public_report_contains_case_rows"] is False,
        ),
        _check(
            "no_fit_tuning_or_policy_selection",
            split["dev_used_for_fit_selection_or_tuning"] is False
            and boundaries["policy_fit"] is False
            and boundaries["threshold_tuned"] is False
            and boundaries["policy_selected"] is False,
        ),
        _check(
            "agent_retrieval_and_test_closed",
            boundaries["agent_model_loaded"] is False
            and boundaries["agent_inference_run"] is False
            and boundaries["retrieval_run"] is False
            and boundaries["test_loaded"] is False,
        ),
        _check(
            "runtime_fallback_rewrite_second_retrieval_closed",
            boundaries["runtime_registered_as_default"] is False
            and boundaries["fallback_strategies_enabled"] is False
            and boundaries["query_rewrite_enabled"] is False
            and boundaries["second_retrieval_enabled"] is False,
        ),
    ]


def _decision(*, report: Mapping[str, Any], all_guards_passed: bool) -> dict[str, Any]:
    assessment = report["primary_hypothesis_assessment"]
    aggregate_observed = assessment["aggregate_visibility_gap_observed"]
    fold_stable = assessment["fold_stable_visibility_gap_observed"]
    if not all_guards_passed:
        status = "primeqa_hybrid_gold_visible_refusal_diagnostics_invalid"
        next_direction = "repair_stage164_process_guards_without_rerunning_agent"
    elif fold_stable:
        status = "primeqa_hybrid_gold_visible_refusal_diagnostics_completed"
        next_direction = "design_train_only_prompt_evidence_visibility_intervention"
    else:
        status = "primeqa_hybrid_gold_visible_refusal_diagnostics_completed"
        next_direction = "design_train_only_router_history_and_question_alignment_diagnostics"
    return {
        "status": status,
        "all_process_guards_passed": all_guards_passed,
        "failed_process_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "aggregate_visibility_gap_observed": aggregate_observed,
        "fold_stable_visibility_gap_observed": fold_stable,
        "diagnostic_only": True,
        "causal_claim": False,
        "policy_selected": False,
        "agent_rerun": False,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": next_direction,
    }


def _binary_value(profile: GoldVisibleRefusalCaseProfile, attribute: str) -> bool:
    if attribute == "turn_position_after_first":
        return profile.turn_position > 1
    return bool(getattr(profile, attribute))


def _visibility_class(
    *,
    exact_visible: bool,
    all_tokens_visible: bool,
    answer_recall: float,
) -> str:
    if exact_visible:
        return "exact_span_visible"
    if all_tokens_visible:
        return "all_tokens_noncontiguous"
    if answer_recall > 0.0:
        return "partial_answer_tokens"
    return "no_answer_tokens"


def _normalized_text(value: str) -> str:
    return " ".join(tokenize_text(value))


def _multiset_recall(required: Sequence[str], available: Sequence[str]) -> float:
    if not required:
        return 0.0
    required_counts = Counter(required)
    available_counts = Counter(available)
    overlap = sum(
        min(count, available_counts.get(token, 0)) for token, count in required_counts.items()
    )
    return round(overlap / len(required), 6)


def _risk_aligned_auc(
    refused: Sequence[float],
    answered: Sequence[float],
    *,
    risk_direction: RiskDirection,
) -> float:
    if not refused or not answered:
        return 0.5
    aligned = 0.0
    for refused_value in refused:
        for answered_value in answered:
            if refused_value == answered_value:
                aligned += 0.5
            elif risk_direction == "higher" and refused_value > answered_value:
                aligned += 1.0
            elif risk_direction == "lower" and refused_value < answered_value:
                aligned += 1.0
    return round(aligned / (len(refused) * len(answered)), 6)


def _haldane_odds_ratio(
    *,
    risk_refused: int,
    risk_answered: int,
    reference_refused: int,
    reference_answered: int,
) -> float:
    return round(
        ((risk_refused + 0.5) * (reference_answered + 0.5))
        / ((risk_answered + 0.5) * (reference_refused + 0.5)),
        6,
    )


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": 0.0, "median": 0.0, "maximum": 0.0, "average": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": round(ordered[0], 6),
        "median": round(float(statistics.median(ordered)), 6),
        "maximum": round(ordered[-1], 6),
        "average": round(float(statistics.fmean(ordered)), 6),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: ProgressSink | None, *, phase: str, **values: Any) -> None:
    if progress_sink is not None:
        progress_sink({"stage": _STAGE, "phase": phase, **values})


def _group_median_bars(
    numeric: Mapping[str, Any],
    feature: str,
) -> list[BarDatum]:
    values = numeric[feature]
    return [
        _bar("refused", values["refused"]["median"]),
        _bar("answered", values["answered"]["median"]),
    ]


def _bar(label: str, value: int | float | bool) -> BarDatum:
    numeric = float(value)
    if isinstance(value, bool):
        value_label = "pass" if value else "fail"
    elif isinstance(value, int):
        value_label = str(value)
    else:
        value_label = f"{numeric:.6f}"
    return BarDatum(label=str(label), value=numeric, value_label=value_label)


def private_report_byte_sha256(path: Path) -> str:
    """Return the byte SHA-256 of a written private Stage164 report."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_report_canonical_sha256(report: Mapping[str, Any]) -> str:
    """Return the canonical JSON SHA-256 of a private Stage164 report."""

    return stage160.canonical_json_sha256(report)
