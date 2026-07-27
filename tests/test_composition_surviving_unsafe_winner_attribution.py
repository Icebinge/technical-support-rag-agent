from __future__ import annotations

from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application import composition_safety_first_frontier as frontier
from ts_rag_agent.application import composition_surviving_unsafe_winner_attribution as analysis
from ts_rag_agent.application.composition_action_audit import ActionAuditRow, CompositionAction


def test_unsafe_winner_partition_separates_gain_dominance_from_risk_ordering() -> None:
    totals = analysis._AttributionTotals()
    totals.add(_decision("gain", unsafe_risk=0.3, strict_risk=0.2))
    totals.add(_decision("risk", unsafe_risk=0.1, strict_risk=0.2))

    report = totals.report()

    assert report["unsafe_winner_context_count"] == 2
    assert report["mechanism_counts"] == {
        "final_gain_dominance": 1,
        "risk_ordering_failure": 1,
    }
    assert report["mechanism_partition_exact"] is True
    assert report["oracle_strict_repairable_count"] == 2
    assert report["risk_rank_bucket_counts"]["2"] == 1
    assert report["risk_rank_bucket_counts"]["3-4"] == 1


def test_loss_type_is_mutually_exclusive() -> None:
    assert analysis._loss_type(_row("q1", "a", citation_delta=-1)) == "citation_only"
    assert analysis._loss_type(_row("q2", "a", f1_delta=-0.1)) == "f1_only"
    assert (
        analysis._loss_type(_row("q3", "a", citation_delta=-1, f1_delta=-0.1)) == "citation_and_f1"
    )


def test_recommendation_maps_observed_mechanism_without_oracle_policy() -> None:
    assert analysis._recommendation("final_gain_dominance") == "risk-aware_final_winner_rule"
    assert analysis._recommendation("risk_ordering_failure") == "unsafe_head_discrimination"


def _decision(
    question_key: str, *, unsafe_risk: float, strict_risk: float
) -> frontier.SafetyFirstFrontierDecision:
    rows = (
        _row(question_key, "baseline", family="baseline"),
        _row(question_key, "strict", strict=True, citation_delta=1, f1_delta=0.1),
        _row(question_key, "unsafe", f1_delta=-0.1),
    )
    safety = tuple(stage194.SafetyPrediction(row, 0.1, 0.1) for row in rows)
    gain = tuple(
        frontier.GainPrediction(row, 1.0 if row.action.action_id == "unsafe" else 0.5)
        for row in rows
    )
    risk = tuple(
        frontier.UnsafePrediction(
            row,
            {
                "baseline": 0.0,
                "strict": strict_risk,
                "unsafe": unsafe_risk,
            }[row.action.action_id],
        )
        for row in rows
    )
    spec = frontier.SafetyFirstFrontierPolicySpec(
        "test",
        "raw_runtime",
        "class_balanced_logistic",
        "raw_runtime",
        "conservative",
        "raw_runtime",
        "conservative",
        1.0,
        4,
    )
    return frontier.build_safety_first_frontier_decisions(safety, gain, risk, spec)[0]


def _row(
    question_key: str,
    action_id: str,
    *,
    family: str = "test",
    strict: bool = False,
    citation_delta: int = 0,
    f1_delta: float = 0.0,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question_key,
        fold_id="fold_1",
        route="other",
        action=CompositionAction(
            action_id=action_id,
            family=family,
            aliases=(),
            selected_indices=(0,),
            matches_stage180=False,
        ),
        runtime_features={"score": float(strict)},
        outcome_class="test",
        strict_expected=strict,
        citation_delta=citation_delta,
        f1_delta=f1_delta,
    )
