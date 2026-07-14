from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg


@dataclass(frozen=True)
class StrategyVisualization:
    """One generated Stage 54 strategy visualization."""

    name: str
    path: str


def review_evaluation_strategy(
    readiness_review: Mapping[str, Any],
    leakage_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Review valid evaluation paths after the proposed held-out set is blocked."""

    leakage_counts = leakage_report["counts"]
    candidate_policy = str(readiness_review["candidate_policy"])
    nvidia_blocked = not bool(leakage_report["heldout_usable_without_exclusions"])
    options = _strategy_options(candidate_policy=candidate_policy)
    return {
        "review_scope": (
            "Evaluation strategy review after Stage 53 leakage audit. This report "
            "does not run held-out metrics, does not select a new evaluation path "
            "on behalf of the user, and does not change the runtime default policy."
        ),
        "current_facts": {
            "stage51_candidate_policy": candidate_policy,
            "stage51_dev_train_readiness_passed": bool(
                readiness_review["overall_decision"][
                    "candidate_passes_dev_train_readiness"
                ]
            ),
            "nvidia_train_json_blocked_as_heldout": nvidia_blocked,
            "nvidia_exact_overlap_questions": leakage_counts["exact_overlap_count"],
            "nvidia_heldout_questions": leakage_counts["heldout_questions"],
            "nvidia_unhandled_overlap_questions": leakage_counts[
                "unhandled_overlap_count"
            ],
            "default_runtime_policy": "unchanged",
        },
        "rejected_paths": [
            {
                "label": "use_nvidia_train_json_as_current_heldout",
                "status": "rejected",
                "reason": (
                    "Stage 53 found exact normalized overlap for all NVIDIA rows "
                    "against PrimeQA train/dev."
                ),
                "evidence": {
                    "heldout_questions": leakage_counts["heldout_questions"],
                    "exact_overlap_questions": leakage_counts["exact_overlap_count"],
                    "unhandled_overlap_questions": leakage_counts[
                        "unhandled_overlap_count"
                    ],
                },
            }
        ],
        "strategy_options": options,
        "decision_required": {
            "requires_user_confirmation": True,
            "reason": (
                "The next path changes evaluation design and project scope. The "
                "available choices have different cost, validity, and defaultization "
                "implications."
            ),
            "recommended_for_confirmation": "external_independent_eval_set",
            "no_action_without_confirmation": (
                "Do not change the default runtime, do not run pseudo-held-out "
                "metrics, and do not tune the Stage 51 candidate."
            ),
        },
        "next_stage_options": [
            {
                "if_user_confirms": "external_independent_eval_set",
                "next_stage": "Stage 55: external dataset discovery and schema fit audit",
            },
            {
                "if_user_confirms": "rebuild_leak_safe_primeqa_split",
                "next_stage": (
                    "Stage 55: split redesign plan and full rerun cost estimate"
                ),
            },
            {
                "if_user_confirms": "freeze_without_defaultization",
                "next_stage": "Stage 55: package current candidate as non-default research result",
            },
        ],
    }


def write_evaluation_strategy_visualizations(
    review: Mapping[str, Any],
    output_dir: Path,
) -> list[StrategyVisualization]:
    """Write compact SVG charts for Stage 54 evaluation strategy review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    options = review["strategy_options"]
    charts = {
        "stage54_option_validity_score.svg": render_horizontal_bar_chart_svg(
            title="Stage 54 option validity score",
            bars=[
                BarDatum(
                    label=option["label"],
                    value=float(option["validity_score"]),
                    value_label=str(option["validity_score"]),
                )
                for option in options
            ],
            x_label="validity score, higher is better",
            margin_left=360,
        ),
        "stage54_option_effort_score.svg": render_horizontal_bar_chart_svg(
            title="Stage 54 option effort score",
            bars=[
                BarDatum(
                    label=option["label"],
                    value=float(option["effort_score"]),
                    value_label=str(option["effort_score"]),
                )
                for option in options
            ],
            x_label="effort score, higher is more work",
            margin_left=360,
        ),
        "stage54_option_defaultization_support.svg": render_horizontal_bar_chart_svg(
            title="Stage 54 option can support defaultization",
            bars=[
                BarDatum(
                    label=option["label"],
                    value=float(option["can_support_defaultization"]),
                    value_label="yes" if option["can_support_defaultization"] else "no",
                )
                for option in options
            ],
            x_label="1 means can support a future defaultization decision",
            margin_left=360,
        ),
        "stage54_blocked_nvidia_overlap.svg": render_horizontal_bar_chart_svg(
            title="Stage 54 blocked NVIDIA overlap",
            bars=[
                BarDatum(
                    label="NVIDIA rows",
                    value=float(review["current_facts"]["nvidia_heldout_questions"]),
                    value_label=str(review["current_facts"]["nvidia_heldout_questions"]),
                ),
                BarDatum(
                    label="exact overlap rows",
                    value=float(
                        review["current_facts"]["nvidia_exact_overlap_questions"]
                    ),
                    value_label=str(
                        review["current_facts"]["nvidia_exact_overlap_questions"]
                    ),
                ),
                BarDatum(
                    label="unhandled overlap rows",
                    value=float(
                        review["current_facts"]["nvidia_unhandled_overlap_questions"]
                    ),
                    value_label=str(
                        review["current_facts"]["nvidia_unhandled_overlap_questions"]
                    ),
                ),
            ],
            x_label="question count",
            margin_left=260,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(StrategyVisualization(name=filename, path=str(path)))
    return artifacts


def _strategy_options(candidate_policy: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "external_independent_eval_set",
            "status": "available_after_user_confirmation",
            "validity_score": 3,
            "effort_score": 2,
            "can_support_defaultization": True,
            "keeps_stage51_candidate_frozen": True,
            "requires_full_pipeline_rerun": False,
            "description": (
                "Find or construct an evaluation set that was not used in the "
                "PrimeQA train/dev development loop, then run leakage audit before "
                "any metrics."
            ),
            "why_consider": (
                "This preserves the Stage 51 candidate and gives the cleanest path "
                "to a real defaultization decision."
            ),
            "risks": [
                "External data may not match the IBM technote corpus or answer schema.",
                "Licensing and public-safe redistribution must be checked.",
                "A schema adapter and leakage audit are required before metrics.",
            ],
            "first_actions_after_confirmation": [
                "List candidate external sources and license constraints.",
                "Check whether each source has questions, answers, answerability, and evidence.",
                "Run leakage audit against PrimeQA train/dev and Stage 51 artifacts.",
                (
                    "Evaluate the frozen candidate only after the source passes "
                    f"leakage checks: {candidate_policy}."
                ),
            ],
        },
        {
            "label": "rebuild_leak_safe_primeqa_split",
            "status": "available_after_user_confirmation",
            "validity_score": 2,
            "effort_score": 3,
            "can_support_defaultization": True,
            "keeps_stage51_candidate_frozen": False,
            "requires_full_pipeline_rerun": True,
            "description": (
                "Redesign train/dev/test splits inside PrimeQA, exclude leakage, "
                "then rebuild candidate-reranker datasets and rerun the policy path."
            ),
            "why_consider": (
                "This avoids needing a new dataset, but it invalidates existing "
                "Stage 31-53 model-selection evidence."
            ),
            "risks": [
                "The current Stage 51 candidate cannot be defaultized from old evidence.",
                "Candidate reranker training data and all comparisons must be rebuilt.",
                "Small split sizes may make results unstable.",
            ],
            "first_actions_after_confirmation": [
                "Design grouped split rules by normalized question and document identity.",
                "Rebuild candidate-reranker dataset only from the new train split.",
                "Rerun dev readiness from scratch on the new dev split.",
                "Use the new test split only once after the new protocol freezes.",
            ],
        },
        {
            "label": "freeze_without_defaultization",
            "status": "available_after_user_confirmation",
            "validity_score": 1,
            "effort_score": 1,
            "can_support_defaultization": False,
            "keeps_stage51_candidate_frozen": True,
            "requires_full_pipeline_rerun": False,
            "description": (
                "Keep Stage 51 as a non-default research candidate and stop before "
                "final evaluation."
            ),
            "why_consider": (
                "This is the safest low-effort option if no independent evaluation "
                "source is available now."
            ),
            "risks": [
                "No defaultization decision can be made.",
                "The project remains at dev/train evidence only.",
                "Future work still needs a real evaluation source.",
            ],
            "first_actions_after_confirmation": [
                "Document Stage 51 as non-default experimental runtime policy.",
                "Keep top-k as default runtime.",
                "Package artifacts and limitations for handoff.",
            ],
        },
    ]
