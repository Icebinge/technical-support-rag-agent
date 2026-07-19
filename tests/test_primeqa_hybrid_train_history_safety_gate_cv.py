from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.analyze_primeqa_hybrid_train_history_safety_gate_cv import app
from ts_rag_agent.application import primeqa_hybrid_train_history_safety_gate_cv as analysis


def test_candidate_family_is_exact_exhaustive_cross_product() -> None:
    specs = analysis.build_stage166_gate_specs()

    assert len(specs) == 1785
    assert len({spec.spec_id for spec in specs}) == len(specs)
    assert (
        analysis.Stage166GateSpec(
            routes=analysis._ROUTES,
            positions=analysis._POSITIONS,
        )
        in specs
    )


def test_policy_evaluation_selects_arm_per_gate() -> None:
    cases = (
        _case(1, answerable=True, isolated_refused=False, synthetic_refused=True),
        _case(2, answerable=False, isolated_refused=False, synthetic_refused=True),
    )
    spec = analysis.Stage166GateSpec(routes=("other",), positions=(4,))

    candidate = analysis.evaluate_stage166_policy(cases, spec)
    baseline = analysis.evaluate_stage166_policy(cases, None)

    assert candidate.isolated_selection_count == 2
    assert candidate.answerable_refusal_count == 0
    assert baseline.answerable_refusal_count == 1
    assert candidate.unanswerable_false_answer_count == 1
    assert baseline.unanswerable_false_answer_count == 0


def test_strict_train_eligibility_rejects_safety_regression() -> None:
    safe_cases = tuple(
        _case(
            fold + 1,
            fold_id=fold,
            answerable=True,
            isolated_refused=False,
            synthetic_refused=True,
        )
        for fold in range(4)
    ) + tuple(
        _case(
            fold + 10,
            fold_id=fold,
            answerable=False,
            isolated_refused=True,
            synthetic_refused=True,
        )
        for fold in range(4)
    )
    unsafe_cases = safe_cases + (
        _case(
            99,
            fold_id=0,
            answerable=False,
            isolated_refused=False,
            synthetic_refused=True,
        ),
    )
    spec = analysis.Stage166GateSpec(routes=("other",), positions=(4,))

    assert analysis._strict_train_eligible(safe_cases, spec) is True
    assert analysis._strict_train_eligible(unsafe_cases, spec) is False


def test_spec_selection_uses_frozen_quality_first_order() -> None:
    cases = tuple(
        _case(
            index,
            fold_id=index % 4,
            answerable=True,
            isolated_refused=False,
            synthetic_refused=True,
            route="other" if index < 4 else "error_or_log",
        )
        for index in range(8)
    )
    narrower = analysis.Stage166GateSpec(routes=("other",), positions=(4,))
    broader = analysis.Stage166GateSpec(
        routes=("other", "error_or_log"),
        positions=(4,),
    )

    selected = analysis._select_spec(cases, (narrower, broader))

    assert selected == broader


def test_visualizations_write_five_parseable_svgs(tmp_path: Path) -> None:
    report = _visual_report()

    visualizations = analysis.write_stage166_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 5
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_cli_exposes_no_development_or_test_inputs() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Stage166" in result.output
    assert "Required" in result.output
    assert "--dev" not in result.output
    assert "--test" not in result.output


def _case(
    index: int,
    *,
    fold_id: int = 0,
    answerable: bool,
    isolated_refused: bool,
    synthetic_refused: bool,
    route: str = "other",
) -> analysis.Stage166PairCase:
    return analysis.Stage166PairCase(
        private_identity_sha256=f"{index:064x}",
        diagnostic_group_sha256=f"{index + 100:064x}",
        fold_id=fold_id,
        question_route=route,
        turn_position=4,
        answerable=answerable,
        top_candidate_score=1.0,
        isolated_refused=isolated_refused,
        synthetic_refused=synthetic_refused,
        isolated_f1=1.0 if answerable and not isolated_refused else 0.0,
        synthetic_f1=1.0 if answerable and not synthetic_refused else 0.0,
        isolated_gold_cited=answerable and not isolated_refused,
        synthetic_gold_cited=answerable and not synthetic_refused,
    )


def _visual_report() -> dict:
    folds = [
        {
            "heldout_fold": fold,
            "strict_train_eligible_spec_count": fold + 1,
            "heldout_delta": {
                "unanswerable_false_answer_count": fold % 2,
                "answerable_f1_sum": float(fold) / 10,
            },
        }
        for fold in range(5)
    ]
    return {
        "outer_cv": {
            "outer_folds": folds,
            "oof_delta": {
                "answerable_refusal_count": -1,
                "answerable_f1_sum": 0.2,
                "answerable_gold_citation_count": 1,
                "unanswerable_false_answer_count": 2,
            },
        },
        "guard_checks": [{"name": "train_only", "passed": True}],
    }
