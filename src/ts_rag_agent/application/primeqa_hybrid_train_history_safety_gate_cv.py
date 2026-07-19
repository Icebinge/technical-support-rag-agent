from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_train_history_isolation_validation as stage165
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 166"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_train_history_safety_gate_outer_cv_v1"
_CORRECTION_SHA256 = "589b65959069d4f12aacc0ff95c2a5f65df173ea0e8c67ca426c15026c01e29d"
_PRIVATE_SHA256 = "ce4b5b281093319696a51251d475a3fc5fa6b7dac2e7f9659464fe1d8e55ad1b"
_CORRECTION_STATUS = "primeqa_hybrid_stage165_transition_correction_completed"
_ROUTES = (
    "error_or_log",
    "how_to_or_lookup",
    "install_upgrade_config",
    "limitation_or_restriction",
    "other",
    "security_bulletin_post_fix_behavior",
    "security_bulletin_remediation",
    "security_bulletin_vulnerability_detail",
)
_POSITIONS = (2, 3, 4)
_FOLDS = (0, 1, 2, 3, 4)
_EXPECTED_CASE_COUNT = 421
_EXPECTED_ANSWERABLE_COUNT = 271
_EXPECTED_UNANSWERABLE_COUNT = 150
_EXPECTED_SPEC_COUNT = ((2 ** len(_ROUTES)) - 1) * ((2 ** len(_POSITIONS)) - 1)
_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Stage166GateSpec:
    routes: tuple[str, ...]
    positions: tuple[int, ...]

    @property
    def spec_id(self) -> str:
        material = json.dumps(
            asdict(self), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def selects(self, case: Stage166PairCase) -> bool:
        return case.question_route in self.routes and case.turn_position in self.positions


@dataclass(frozen=True)
class Stage166PairCase:
    private_identity_sha256: str
    diagnostic_group_sha256: str
    fold_id: int
    question_route: str
    turn_position: int
    answerable: bool
    top_candidate_score: float
    isolated_refused: bool
    synthetic_refused: bool
    isolated_f1: float
    synthetic_f1: float
    isolated_gold_cited: bool
    synthetic_gold_cited: bool


@dataclass(frozen=True)
class Stage166PolicyMetrics:
    case_count: int
    isolated_selection_count: int
    answerable_count: int
    unanswerable_count: int
    answerable_refusal_count: int
    answerable_f1_sum: float
    answerable_average_f1: float
    answerable_gold_citation_count: int
    unanswerable_false_answer_count: int

    def public_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage166Visualization:
    name: str
    path: str


def run_stage166_safety_gate_outer_cv(
    *,
    correction_report_path: Path,
    private_report_path: Path,
    user_confirmed_stage166: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Run strict train-only outer CV over exhaustive route-by-position gates."""

    fingerprints = {
        "stage165_correction": _fingerprint(correction_report_path),
        "stage165_private": _fingerprint(private_report_path),
    }
    _authorize_fingerprints(fingerprints)
    correction = _load_json_object(correction_report_path)
    private = _load_json_object(private_report_path)
    _authorize_reports(correction=correction, private=private)
    cases = _build_cases(private["rows"])
    specs = build_stage166_gate_specs()
    family_sha256 = _spec_family_sha256(specs)

    outer_folds = []
    oof_choices: dict[str, bool] = {}
    for heldout_fold in _FOLDS:
        train_cases = tuple(case for case in cases if case.fold_id != heldout_fold)
        heldout_cases = tuple(case for case in cases if case.fold_id == heldout_fold)
        eligible = tuple(spec for spec in specs if _strict_train_eligible(train_cases, spec))
        selected = _select_spec(train_cases, eligible)
        heldout_candidate = evaluate_stage166_policy(heldout_cases, selected)
        heldout_baseline = evaluate_stage166_policy(heldout_cases, None)
        for case in heldout_cases:
            oof_choices[case.private_identity_sha256] = selected.selects(case)
        outer_folds.append(
            {
                "heldout_fold": heldout_fold,
                "train_case_count": len(train_cases),
                "heldout_case_count": len(heldout_cases),
                "candidate_spec_count": len(specs),
                "strict_train_eligible_spec_count": len(eligible),
                "selected_spec": _spec_dict(selected),
                "heldout_metrics": heldout_candidate.public_dict(),
                "heldout_baseline": heldout_baseline.public_dict(),
                "heldout_delta": _metric_delta(heldout_candidate, heldout_baseline),
                "heldout_strict_nonregression": _strict_nonregression(
                    heldout_candidate,
                    heldout_baseline,
                ),
            }
        )

    oof = _evaluate_choices(cases, oof_choices)
    baseline = evaluate_stage166_policy(cases, None)
    oof_by_fold = {
        str(fold): {
            "candidate": _evaluate_choices(
                tuple(case for case in cases if case.fold_id == fold),
                oof_choices,
            ).public_dict(),
            "baseline": evaluate_stage166_policy(
                tuple(case for case in cases if case.fold_id == fold),
                None,
            ).public_dict(),
        }
        for fold in _FOLDS
    }
    for values in oof_by_fold.values():
        candidate = Stage166PolicyMetrics(**values["candidate"])
        base = Stage166PolicyMetrics(**values["baseline"])
        values["delta"] = _metric_delta(candidate, base)
        values["strict_nonregression"] = _strict_nonregression(candidate, base)

    controls = {
        "always_synthetic_history": baseline.public_dict(),
        "always_isolated": evaluate_stage166_policy(cases, _all_spec()).public_dict(),
        "turn_position_4_only": evaluate_stage166_policy(
            cases,
            Stage166GateSpec(routes=_ROUTES, positions=(4,)),
        ).public_dict(),
    }
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only strict outer five-fold diagnostic of pre-generation runtime "
            "features for selective history isolation. Every non-empty question-route "
            "subset is crossed with every non-empty turn-position subset. Rules are "
            "selected on four folds and evaluated once on the held-out fold."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_stage166),
            "confirmation_note": confirmation_note,
        },
        "source_authorization": fingerprints,
        "corrected_stage165_contract": {
            "correction_status": correction["decision"]["status"],
            "corrected_worsened_count": correction["correction"][
                "corrected_post_first_unanswerable"
            ]["synthetic_refusal_to_isolated_false_answer_count"],
            "corrected_improved_count": correction["correction"][
                "corrected_post_first_unanswerable"
            ]["synthetic_false_answer_to_isolated_refusal_count"],
            "stage165_decision_unchanged": correction["decision"][
                "stage165_candidate_status_unchanged"
            ],
        },
        "feature_contract": {
            "runtime_features": [
                "question_route",
                "synthetic_turn_position",
                "top_candidate_score",
            ],
            "decision_time": "after retrieval_before_router_generation",
            "arm_invariant_feature_check": True,
            "top_candidate_score_unique_count": len({case.top_candidate_score for case in cases}),
            "top_candidate_score_minimum": min(case.top_candidate_score for case in cases),
            "top_candidate_score_maximum": max(case.top_candidate_score for case in cases),
            "top_candidate_score_excluded_from_rules": True,
            "top_candidate_score_exclusion_reason": "constant_1.0_on_all_post_first_cases",
            "forbidden_runtime_features": [
                "answerable_label",
                "gold_document",
                "gold_rank",
                "gold_prompt_overlap",
                "selected_action",
                "refused_outcome",
                "answer_f1",
                "citation_outcome",
            ],
        },
        "case_summary": {
            "case_count": len(cases),
            "answerable_count": sum(case.answerable for case in cases),
            "unanswerable_count": sum(not case.answerable for case in cases),
            "fold_case_counts": {
                str(fold): sum(case.fold_id == fold for case in cases) for fold in _FOLDS
            },
            "route_counts": {
                route: sum(case.question_route == route for case in cases) for route in _ROUTES
            },
            "turn_position_counts": {
                str(position): sum(case.turn_position == position for case in cases)
                for position in _POSITIONS
            },
            "private_case_rows_written": False,
        },
        "candidate_family": {
            "route_subset_count": (2 ** len(_ROUTES)) - 1,
            "position_subset_count": (2 ** len(_POSITIONS)) - 1,
            "candidate_spec_count": len(specs),
            "family_sha256": family_sha256,
            "selection_constraints": [
                "aggregate answerable refusal no worse than synthetic history",
                "aggregate answerable F1 no worse than synthetic history",
                "aggregate gold citations no worse than synthetic history",
                "aggregate unanswerable false answers no worse than synthetic history",
                "the same four nonregressions in every training fold",
                "at least one aggregate metric strictly improves",
            ],
            "selection_order": [
                "answerable_f1_gain",
                "gold_citation_gain",
                "answerable_refusal_reduction",
                "unanswerable_false_answer_reduction",
                "smaller_isolation_scope",
                "stable_spec_identity",
            ],
            "fit_model": False,
            "threshold_tuning": False,
            "rule_selection": True,
        },
        "outer_cv": {
            "fold_count": len(_FOLDS),
            "grouped_fold_assignment_reused_from_stage165": True,
            "outer_folds": outer_folds,
            "oof_metrics": oof.public_dict(),
            "oof_baseline": baseline.public_dict(),
            "oof_delta": _metric_delta(oof, baseline),
            "oof_by_fold": oof_by_fold,
            "oof_strict_nonregression": _strict_nonregression(oof, baseline),
            "oof_all_folds_strict_nonregression": all(
                values["strict_nonregression"] for values in oof_by_fold.values()
            ),
        },
        "controls": controls,
        "execution_counts": {
            "candidate_specs_per_outer_fold": len(specs),
            "candidate_evaluations": len(specs) * len(_FOLDS),
            "model_fit_runs": 0,
            "retrieval_runs": 0,
            "agent_runs": 0,
            "model_generation_runs": 0,
            "development_rows_loaded": 0,
            "test_rows_loaded": 0,
        },
        "closed_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
            "policy_selected": False,
            "runtime_registered_as_default": False,
            "fallback_strategies_enabled": False,
        },
    }
    report["guard_checks"] = _guard_checks(report)
    report["public_safe_contract"] = stage165._public_safe_contract(report)
    all_guards = all(check["passed"] for check in report["guard_checks"])
    cv_safe = bool(
        report["outer_cv"]["oof_strict_nonregression"]
        and report["outer_cv"]["oof_all_folds_strict_nonregression"]
    )
    report["decision"] = {
        "status": (
            "primeqa_hybrid_stage166_runtime_feature_gate_candidate_found"
            if all_guards and cv_safe
            else "primeqa_hybrid_stage166_runtime_feature_family_insufficient"
            if all_guards
            else "primeqa_hybrid_stage166_diagnostics_invalid"
        ),
        "all_process_guards_passed": all_guards,
        "failed_process_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "outer_cv_strict_nonregression": cv_safe,
        "candidate_selected": all_guards and cv_safe,
        "diagnostic_only": True,
        "development_gate_opened": False,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": (
            "freeze_selected_gate_before_any_development_validation"
            if all_guards and cv_safe
            else "enrich_pre_generation_runtime_evidence_features_on_train_only"
        ),
    }
    return report


def build_stage166_gate_specs() -> tuple[Stage166GateSpec, ...]:
    specs = []
    for route_count in range(1, len(_ROUTES) + 1):
        for routes in itertools.combinations(_ROUTES, route_count):
            for position_count in range(1, len(_POSITIONS) + 1):
                for positions in itertools.combinations(_POSITIONS, position_count):
                    specs.append(Stage166GateSpec(routes=routes, positions=positions))
    return tuple(specs)


def evaluate_stage166_policy(
    cases: Sequence[Stage166PairCase],
    spec: Stage166GateSpec | None,
) -> Stage166PolicyMetrics:
    selected_count = 0
    answerable_count = 0
    unanswerable_count = 0
    refusal_count = 0
    f1_sum = 0.0
    citation_count = 0
    false_answer_count = 0
    for case in cases:
        isolated = spec is not None and spec.selects(case)
        selected_count += int(isolated)
        refused = case.isolated_refused if isolated else case.synthetic_refused
        if case.answerable:
            answerable_count += 1
            refusal_count += int(refused)
            f1_sum += case.isolated_f1 if isolated else case.synthetic_f1
            citation_count += int(
                case.isolated_gold_cited if isolated else case.synthetic_gold_cited
            )
        else:
            unanswerable_count += 1
            false_answer_count += int(not refused)
    return Stage166PolicyMetrics(
        case_count=len(cases),
        isolated_selection_count=selected_count,
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        answerable_refusal_count=refusal_count,
        answerable_f1_sum=round(f1_sum, 6),
        answerable_average_f1=round(f1_sum / answerable_count, 6) if answerable_count else 0.0,
        answerable_gold_citation_count=citation_count,
        unanswerable_false_answer_count=false_answer_count,
    )


def write_stage166_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> list[Stage166Visualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = report["outer_cv"]["outer_folds"]
    specs = {
        "stage166_train_eligible_rule_counts.svg": (
            "Stage166 strict train-fold eligible rule counts",
            [
                _bar(f"heldout fold {row['heldout_fold']}", row["strict_train_eligible_spec_count"])
                for row in folds
            ],
            "eligible rules",
        ),
        "stage166_heldout_false_answer_deltas.svg": (
            "Stage166 heldout unanswerable false-answer deltas",
            [
                _bar(
                    f"fold {row['heldout_fold']}",
                    row["heldout_delta"]["unanswerable_false_answer_count"],
                )
                for row in folds
            ],
            "candidate minus synthetic false answers",
        ),
        "stage166_heldout_f1_deltas.svg": (
            "Stage166 heldout answerable F1-sum deltas",
            [
                _bar(f"fold {row['heldout_fold']}", row["heldout_delta"]["answerable_f1_sum"])
                for row in folds
            ],
            "candidate minus synthetic F1 sum",
        ),
        "stage166_oof_metric_deltas.svg": (
            "Stage166 outer-CV aggregate metric deltas",
            [
                _bar(
                    "answerable refusal count",
                    report["outer_cv"]["oof_delta"]["answerable_refusal_count"],
                ),
                _bar("answerable F1 sum", report["outer_cv"]["oof_delta"]["answerable_f1_sum"]),
                _bar(
                    "gold citations",
                    report["outer_cv"]["oof_delta"]["answerable_gold_citation_count"],
                ),
                _bar(
                    "unanswerable false answers",
                    report["outer_cv"]["oof_delta"]["unanswerable_false_answer_count"],
                ),
            ],
            "candidate minus synthetic",
        ),
        "stage166_guard_checks.svg": (
            "Stage166 process guard checks",
            [_bar(check["name"], int(check["passed"])) for check in report["guard_checks"]],
            "passed",
        ),
    }
    artifacts = []
    for filename, (title, bars, x_label) in specs.items():
        path = output_dir / filename
        path.write_text(
            render_horizontal_bar_chart_svg(
                title=title,
                bars=bars,
                x_label=x_label,
                margin_left=430,
            ),
            encoding="utf-8",
        )
        artifacts.append(Stage166Visualization(name=filename, path=str(path)))
    return artifacts


def _build_cases(rows: Sequence[Mapping[str, Any]]) -> tuple[Stage166PairCase, ...]:
    arms_by_identity: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        arms_by_identity.setdefault(str(row["private_identity_sha256"]), {})[str(row["arm"])] = row
    cases = []
    for identity, arms in arms_by_identity.items():
        isolated = arms["isolated"]
        synthetic = arms["synthetic_history"]
        if int(isolated["synthetic_turn_position"]) == 1:
            continue
        invariant_fields = (
            "diagnostic_group_sha256",
            "fold_id",
            "question_route",
            "synthetic_turn_position",
            "answerable",
            "top_candidate_score",
        )
        if any(isolated[field] != synthetic[field] for field in invariant_fields):
            raise ValueError("Stage166 requires exact arm-invariant runtime features")
        cases.append(
            Stage166PairCase(
                private_identity_sha256=identity,
                diagnostic_group_sha256=str(isolated["diagnostic_group_sha256"]),
                fold_id=int(isolated["fold_id"]),
                question_route=str(isolated["question_route"]),
                turn_position=int(isolated["synthetic_turn_position"]),
                answerable=bool(isolated["answerable"]),
                top_candidate_score=float(isolated["top_candidate_score"]),
                isolated_refused=bool(isolated["refused"]),
                synthetic_refused=bool(synthetic["refused"]),
                isolated_f1=float(isolated["answer_token_f1"] or 0.0),
                synthetic_f1=float(synthetic["answer_token_f1"] or 0.0),
                isolated_gold_cited=bool(isolated["gold_cited"]),
                synthetic_gold_cited=bool(synthetic["gold_cited"]),
            )
        )
    return tuple(sorted(cases, key=lambda case: case.private_identity_sha256))


def _strict_train_eligible(cases: Sequence[Stage166PairCase], spec: Stage166GateSpec) -> bool:
    candidate = evaluate_stage166_policy(cases, spec)
    baseline = evaluate_stage166_policy(cases, None)
    if candidate.isolated_selection_count == 0:
        return False
    if not _strict_nonregression(candidate, baseline):
        return False
    for fold in sorted({case.fold_id for case in cases}):
        fold_cases = tuple(case for case in cases if case.fold_id == fold)
        if not _strict_nonregression(
            evaluate_stage166_policy(fold_cases, spec),
            evaluate_stage166_policy(fold_cases, None),
        ):
            return False
    delta = _metric_delta(candidate, baseline)
    return (
        delta["answerable_refusal_count"] < 0
        or delta["answerable_f1_sum"] > _FLOAT_TOLERANCE
        or delta["answerable_gold_citation_count"] > 0
        or delta["unanswerable_false_answer_count"] < 0
    )


def _select_spec(
    cases: Sequence[Stage166PairCase], specs: Sequence[Stage166GateSpec]
) -> Stage166GateSpec:
    if not specs:
        raise ValueError("Stage166 outer fold has no strict train-eligible rule")
    baseline = evaluate_stage166_policy(cases, None)

    def key(spec: Stage166GateSpec) -> tuple[Any, ...]:
        candidate = evaluate_stage166_policy(cases, spec)
        return (
            candidate.answerable_f1_sum - baseline.answerable_f1_sum,
            candidate.answerable_gold_citation_count - baseline.answerable_gold_citation_count,
            baseline.answerable_refusal_count - candidate.answerable_refusal_count,
            baseline.unanswerable_false_answer_count - candidate.unanswerable_false_answer_count,
            -candidate.isolated_selection_count,
            spec.spec_id,
        )

    return max(specs, key=key)


def _strict_nonregression(
    candidate: Stage166PolicyMetrics, baseline: Stage166PolicyMetrics
) -> bool:
    return (
        candidate.answerable_refusal_count <= baseline.answerable_refusal_count
        and candidate.answerable_f1_sum + _FLOAT_TOLERANCE >= baseline.answerable_f1_sum
        and candidate.answerable_gold_citation_count >= baseline.answerable_gold_citation_count
        and candidate.unanswerable_false_answer_count <= baseline.unanswerable_false_answer_count
    )


def _metric_delta(
    candidate: Stage166PolicyMetrics, baseline: Stage166PolicyMetrics
) -> dict[str, int | float]:
    return {
        "isolated_selection_count": candidate.isolated_selection_count
        - baseline.isolated_selection_count,
        "answerable_refusal_count": candidate.answerable_refusal_count
        - baseline.answerable_refusal_count,
        "answerable_f1_sum": round(candidate.answerable_f1_sum - baseline.answerable_f1_sum, 6),
        "answerable_average_f1": round(
            candidate.answerable_average_f1 - baseline.answerable_average_f1, 6
        ),
        "answerable_gold_citation_count": candidate.answerable_gold_citation_count
        - baseline.answerable_gold_citation_count,
        "unanswerable_false_answer_count": candidate.unanswerable_false_answer_count
        - baseline.unanswerable_false_answer_count,
    }


def _evaluate_choices(
    cases: Sequence[Stage166PairCase], choices: Mapping[str, bool]
) -> Stage166PolicyMetrics:
    selected = _ChoiceSpec(choices)
    return evaluate_stage166_policy(cases, selected)  # type: ignore[arg-type]


class _ChoiceSpec:
    def __init__(self, choices: Mapping[str, bool]) -> None:
        self._choices = choices

    def selects(self, case: Stage166PairCase) -> bool:
        return bool(self._choices[case.private_identity_sha256])


def _spec_dict(spec: Stage166GateSpec) -> dict[str, Any]:
    return {"spec_id": spec.spec_id, "routes": list(spec.routes), "positions": list(spec.positions)}


def _all_spec() -> Stage166GateSpec:
    return Stage166GateSpec(routes=_ROUTES, positions=_POSITIONS)


def _spec_family_sha256(specs: Sequence[Stage166GateSpec]) -> str:
    payload = [asdict(spec) for spec in specs]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _authorize_fingerprints(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    if fingerprints["stage165_correction"]["sha256"] != _CORRECTION_SHA256:
        raise ValueError("Stage166 correction report fingerprint mismatch")
    if fingerprints["stage165_private"]["sha256"] != _PRIVATE_SHA256:
        raise ValueError("Stage166 private report fingerprint mismatch")


def _authorize_reports(*, correction: Mapping[str, Any], private: Mapping[str, Any]) -> None:
    if correction.get("decision", {}).get("status") != _CORRECTION_STATUS:
        raise ValueError("Stage166 requires the completed Stage165 correction")
    if correction.get("decision", {}).get("all_correction_guards_passed") is not True:
        raise ValueError("Stage166 requires all correction guards")
    if private.get("arm_row_count") != 1124 or len(private.get("rows", [])) != 1124:
        raise ValueError("Stage166 requires the exact 1124-row private report")


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    features = report["feature_contract"]
    cases = report["case_summary"]
    family = report["candidate_family"]
    outer = report["outer_cv"]
    counts = report["execution_counts"]
    closed = report["closed_boundaries"]
    return [
        _check("user_confirmed_stage166", report["user_confirmation"]["confirmed"] is True),
        _check(
            "corrected_stage165_sources_exact",
            report["source_authorization"]["stage165_correction"]["sha256"] == _CORRECTION_SHA256
            and report["source_authorization"]["stage165_private"]["sha256"] == _PRIVATE_SHA256,
        ),
        _check(
            "corrected_transition_semantics_consumed",
            report["corrected_stage165_contract"]["corrected_worsened_count"] == 22
            and report["corrected_stage165_contract"]["corrected_improved_count"] == 3,
        ),
        _check(
            "exact_post_first_case_coverage",
            cases["case_count"] == _EXPECTED_CASE_COUNT
            and cases["answerable_count"] == _EXPECTED_ANSWERABLE_COUNT
            and cases["unanswerable_count"] == _EXPECTED_UNANSWERABLE_COUNT,
        ),
        _check("arm_invariant_runtime_features", features["arm_invariant_feature_check"] is True),
        _check(
            "constant_top_score_not_used_as_gate",
            features["top_candidate_score_unique_count"] == 1
            and features["top_candidate_score_minimum"]
            == features["top_candidate_score_maximum"]
            == 1.0
            and features["top_candidate_score_excluded_from_rules"] is True,
        ),
        _check(
            "exact_exhaustive_candidate_family",
            family["candidate_spec_count"] == _EXPECTED_SPEC_COUNT
            and counts["candidate_evaluations"] == _EXPECTED_SPEC_COUNT * len(_FOLDS),
        ),
        _check(
            "five_outer_folds_complete",
            outer["fold_count"] == 5
            and len(outer["outer_folds"]) == 5
            and all(row["selected_spec"] for row in outer["outer_folds"]),
        ),
        _check(
            "no_private_case_rows_in_public_report",
            cases["private_case_rows_written"] is False,
        ),
        _check(
            "no_model_retrieval_agent_dev_or_test_execution",
            counts["model_fit_runs"] == 0
            and counts["retrieval_runs"] == 0
            and counts["agent_runs"] == 0
            and counts["model_generation_runs"] == 0
            and counts["development_rows_loaded"] == 0
            and counts["test_rows_loaded"] == 0,
        ),
        _check(
            "dev_test_runtime_and_fallback_closed",
            closed["development_loaded"] is False
            and closed["test_loaded"] is False
            and closed["policy_selected"] is False
            and closed["runtime_registered_as_default"] is False
            and closed["fallback_strategies_enabled"] is False,
        ),
    ]


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _bar(label: str, value: int | float) -> BarDatum:
    return BarDatum(label=label, value=float(value), value_label=str(value))


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}
