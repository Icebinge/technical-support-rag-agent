from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 60"
_CREATED_AT = "2026-07-14"
_RECOMMENDED = "recommended_for_user_confirmation"
_SECONDARY = "secondary_option"
_BLOCKED = "blocked"
_REJECTED = "rejected"
_UNCHANGED = "unchanged"


@dataclass(frozen=True)
class ProtocolOption:
    """One MSQA Stage 51 protocol design option."""

    label: str
    status: str
    coverage_percent: float
    coverage_source: str
    coverage_score: int
    citation_grounding_score: int
    stage51_feature_fit_score: int
    leakage_safety_score: int
    effort_score: int
    total_score: int
    decision: str
    strengths: tuple[str, ...]
    risks: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolVisualization:
    """One generated Stage 60 protocol design visualization."""

    name: str
    path: str


def design_msqa_stage51_protocol(
    *,
    schema_probe_report_path: Path,
    evaluation_split_report_path: Path,
    compatibility_review_path: Path,
) -> dict[str, Any]:
    """Design an MSQA source/citation protocol before any Stage 51 candidate run."""

    for path in [
        schema_probe_report_path,
        evaluation_split_report_path,
        compatibility_review_path,
    ]:
        _ensure_file(path)
    schema_probe = _load_json(schema_probe_report_path)
    evaluation_split = _load_json(evaluation_split_report_path)
    compatibility_review = _load_json(compatibility_review_path)
    _validate_inputs(schema_probe, evaluation_split, compatibility_review)

    coverage = schema_probe["source_link_coverage"]
    selected_count = int(
        evaluation_split["frozen_split"]["filter_counts"]["selected_question_count"]
    )
    source_options = _source_identity_options(coverage)
    candidate_options = _candidate_construction_options(coverage)
    recommended_source = _recommended_option(source_options)
    recommended_candidate = _recommended_option(candidate_options)

    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "MSQA source/citation adapter and Stage 51 comparison protocol design. "
            "This report does not run Stage 51, does not tune policies, does not "
            "fetch external pages, does not add fallback fields, and does not "
            "change the default runtime."
        ),
        "source_files": {
            "schema_probe_report": _fingerprint(schema_probe_report_path),
            "evaluation_split_report": _fingerprint(evaluation_split_report_path),
            "compatibility_review": _fingerprint(compatibility_review_path),
        },
        "current_constraints": {
            "frozen_split": evaluation_split["frozen_split"]["split_name"],
            "adapter_contract_version": evaluation_split["adapter_contract"][
                "contract_version"
            ],
            "selected_question_count": selected_count,
            "approved_answer_field": evaluation_split["adapter_contract"][
                "answer_field"
            ],
            "approved_source_url_field": evaluation_split["adapter_contract"][
                "source_url_field"
            ],
            "no_answer_field_fallback": True,
            "question_text_index_rejected": True,
            "stage59_blocker_count": compatibility_review["compatibility_gate"][
                "summary"
            ]["blocker_count"],
            "default_runtime_policy": _UNCHANGED,
        },
        "scoring_rubric": {
            "score_boundary": (
                "Generated protocol-fit rubric for Stage 60 design only. It is "
                "not a model-quality metric and does not report answer accuracy."
            ),
            "total_score_formula": (
                "coverage_score + citation_grounding_score + "
                "stage51_feature_fit_score + leakage_safety_score - effort_score"
            ),
            "coverage_score": {
                "3": "coverage >= 99%",
                "2": "coverage >= 60% and < 99%",
                "1": "coverage > 0% and < 60%",
                "0": "coverage == 0%",
            },
            "score_ranges": {
                "citation_grounding_score": (
                    "0-3, higher means closer to the citation identity needed "
                    "by evaluation"
                ),
                "stage51_feature_fit_score": (
                    "0-3, higher means easier to build Stage 51-style evidence "
                    "candidates"
                ),
                "leakage_safety_score": (
                    "0-3, higher means lower risk of query self-match or task "
                    "leakage"
                ),
                "effort_score": "1-3, higher means more implementation or audit work",
            },
        },
        "source_citation_identity_options": [asdict(option) for option in source_options],
        "candidate_construction_options": [
            asdict(option) for option in candidate_options
        ],
        "recommended_protocol": _recommended_protocol(
            recommended_source=recommended_source,
            recommended_candidate=recommended_candidate,
        ),
        "decision": {
            "status": "msqa_stage51_protocol_ready_for_user_confirmation",
            "requires_user_confirmation": True,
            "can_run_stage51_candidate_now": False,
            "can_defaultize_runtime_now": False,
            "default_runtime_policy": _UNCHANGED,
            "recommended_source_citation_identity": recommended_source.label,
            "recommended_candidate_construction": recommended_candidate.label,
            "no_action_without_confirmation": (
                "Do not implement the Stage 51 MSQA candidate adapter, do not run "
                "Stage 51 on MSQA, and do not change the runtime default before "
                "the recommended protocol is confirmed."
            ),
            "recommended_next_stage": (
                "Stage 61: after user confirmation, implement the MSQA row-source "
                "answer-sentence candidate adapter and dry-run contract tests"
            ),
        },
    }


def write_msqa_stage51_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[ProtocolVisualization]:
    """Write SVG charts for Stage 60 protocol design."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage60_source_identity_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage 60 source identity option scores",
            bars=_option_score_bars(report["source_citation_identity_options"]),
            x_label="protocol-fit score",
            margin_left=340,
        ),
        "stage60_candidate_construction_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage 60 candidate construction option scores",
            bars=_option_score_bars(report["candidate_construction_options"]),
            x_label="protocol-fit score",
            margin_left=360,
        ),
        "stage60_source_coverage.svg": render_horizontal_bar_chart_svg(
            title="Stage 60 source coverage from MSQA probe",
            bars=_coverage_bars(report["source_citation_identity_options"]),
            x_label="coverage percent",
            margin_left=340,
        ),
        "stage60_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage 60 protocol decision flags",
            bars=[
                BarDatum(
                    label="requires_user_confirmation",
                    value=float(report["decision"]["requires_user_confirmation"]),
                    value_label="yes",
                ),
                BarDatum(
                    label="can_run_stage51_candidate_now",
                    value=float(report["decision"]["can_run_stage51_candidate_now"]),
                    value_label="no",
                ),
                BarDatum(
                    label="can_defaultize_runtime_now",
                    value=float(report["decision"]["can_defaultize_runtime_now"]),
                    value_label="no",
                ),
            ],
            x_label="1 means true",
            margin_left=320,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(ProtocolVisualization(name=filename, path=str(path)))
    return artifacts


def _validate_inputs(
    schema_probe: Mapping[str, Any],
    evaluation_split: Mapping[str, Any],
    compatibility_review: Mapping[str, Any],
) -> None:
    if schema_probe.get("stage") != "Stage 56":
        raise ValueError(f"Expected Stage 56 schema probe, got: {schema_probe.get('stage')!r}")
    if evaluation_split.get("stage") != "Stage 57":
        raise ValueError(
            f"Expected Stage 57 evaluation split, got: {evaluation_split.get('stage')!r}"
        )
    if compatibility_review.get("stage") != "Stage 59":
        raise ValueError(
            "Expected Stage 59 compatibility review, got: "
            f"{compatibility_review.get('stage')!r}"
        )
    required_coverage = {
        "rows_with_row_url",
        "rows_with_processed_answer_link",
        "rows_with_processed_answer_learn_link",
        "rows_with_processed_answer_azure_docish_link",
    }
    missing_coverage = sorted(
        required_coverage.difference(schema_probe.get("source_link_coverage", {}))
    )
    if missing_coverage:
        raise ValueError(f"Stage 56 report missing coverage keys: {missing_coverage}")
    if (
        evaluation_split["adapter_contract"].get("answer_field")
        != "ProcessedAnswerText"
    ):
        raise ValueError("Stage 57 adapter must use ProcessedAnswerText")
    if compatibility_review["decision"].get("can_run_stage51_candidate_now") is not False:
        raise ValueError("Stage 60 expects Stage 59 to block direct Stage 51 comparison")


def _source_identity_options(coverage: Mapping[str, Any]) -> tuple[ProtocolOption, ...]:
    row_url_coverage = _coverage_percent(coverage, "rows_with_row_url")
    processed_link_coverage = _coverage_percent(
        coverage,
        "rows_with_processed_answer_link",
    )
    processed_learn_coverage = _coverage_percent(
        coverage,
        "rows_with_processed_answer_learn_link",
    )
    azure_docish_coverage = _coverage_percent(
        coverage,
        "rows_with_processed_answer_azure_docish_link",
    )
    return (
        _option(
            label="msqa_row_source_url",
            status=_RECOMMENDED,
            coverage_percent=row_url_coverage,
            coverage_source="rows_with_row_url",
            citation_grounding_score=2,
            stage51_feature_fit_score=2,
            leakage_safety_score=3,
            effort_score=1,
            decision=(
                "Use QuestionId + AnswerId + row Url as the MSQA row-source "
                "citation identity for the next adapter."
            ),
            strengths=(
                "100% local row-level URL coverage in Stage 56.",
                "Matches the Stage 57 approved source URL field.",
                "Does not require fetching external pages.",
                "Avoids using question text in the retrieval index.",
            ),
            risks=(
                "This is row-level source attribution, not document-span citation.",
                "It cannot claim complete Microsoft Learn documentation citation coverage.",
            ),
        ),
        _option(
            label="processed_answer_links",
            status=_BLOCKED,
            coverage_percent=processed_link_coverage,
            coverage_source="rows_with_processed_answer_link",
            citation_grounding_score=3,
            stage51_feature_fit_score=3,
            leakage_safety_score=2,
            effort_score=2,
            decision=(
                "Do not use as required citation identity because coverage is "
                "incomplete and no fallback field is approved."
            ),
            strengths=(
                "Links inside processed answers can point closer to documentation evidence.",
                "Could become a future optional analysis dimension after a coverage audit.",
            ),
            risks=(
                "Coverage is below 100%, so it cannot support a no-fallback protocol.",
                "Links are not guaranteed to be the exact evidence span for every answer.",
            ),
        ),
        _option(
            label="processed_answer_learn_links",
            status=_BLOCKED,
            coverage_percent=processed_learn_coverage,
            coverage_source="rows_with_processed_answer_learn_link",
            citation_grounding_score=3,
            stage51_feature_fit_score=3,
            leakage_safety_score=2,
            effort_score=2,
            decision=(
                "Do not use as required citation identity because Microsoft Learn "
                "link coverage is incomplete."
            ),
            strengths=(
                "Closer to Learn-document citation when present.",
                "Can inform future document-grounded dataset construction.",
            ),
            risks=(
                "Only a minority of local rows have processed-answer Learn links.",
                "Would require excluding many rows or adding an unapproved fallback.",
            ),
        ),
        _option(
            label="processed_answer_azure_docish_links",
            status=_BLOCKED,
            coverage_percent=azure_docish_coverage,
            coverage_source="rows_with_processed_answer_azure_docish_link",
            citation_grounding_score=3,
            stage51_feature_fit_score=3,
            leakage_safety_score=2,
            effort_score=2,
            decision=(
                "Do not use as required citation identity because Azure doc-like "
                "link coverage is too sparse."
            ),
            strengths=(
                "Most document-like option when available.",
                "Potential future seed for a smaller document-grounded subset.",
            ),
            risks=(
                "Coverage is far too low for the current frozen split protocol.",
                "Would no longer evaluate the full Stage 57 frozen split.",
            ),
        ),
        _option(
            label="question_answer_page_text",
            status=_REJECTED,
            coverage_percent=row_url_coverage,
            coverage_source="rows_with_row_url",
            citation_grounding_score=1,
            stage51_feature_fit_score=1,
            leakage_safety_score=0,
            effort_score=1,
            decision=(
                "Reject as a comparison identity because Stage 59 showed that "
                "indexing question text trivializes retrieval."
            ),
            strengths=("Easy to construct from local rows.",),
            risks=(
                "Includes the query text in the index.",
                "Stage 58 diagnostic hit@1/MRR/F1 reached 1.0 and is not fair.",
            ),
        ),
    )


def _candidate_construction_options(
    coverage: Mapping[str, Any],
) -> tuple[ProtocolOption, ...]:
    answer_coverage = 100.0
    processed_learn_coverage = _coverage_percent(
        coverage,
        "rows_with_processed_answer_learn_link",
    )
    row_url_coverage = _coverage_percent(coverage, "rows_with_row_url")
    return (
        _option(
            label="processed_answer_sentence_candidates",
            status=_RECOMMENDED,
            coverage_percent=answer_coverage,
            coverage_source="ProcessedAnswerText required by Stage 57",
            citation_grounding_score=2,
            stage51_feature_fit_score=3,
            leakage_safety_score=3,
            effort_score=2,
            decision=(
                "Split ProcessedAnswerText into answer sentences and attach each "
                "candidate to the row-source citation identity."
            ),
            strengths=(
                "Uses the approved Stage 57 answer field with full local coverage.",
                "Produces Stage 51-style sentence candidates.",
                "Can carry retrieval rank, candidate score, and source-row identity.",
            ),
            risks=(
                "Still evaluates row-source answer evidence, not document-span evidence.",
                "Requires a dedicated adapter before any candidate policy run.",
            ),
        ),
        _option(
            label="processed_answer_chunk_candidates",
            status=_SECONDARY,
            coverage_percent=answer_coverage,
            coverage_source="ProcessedAnswerText required by Stage 57",
            citation_grounding_score=2,
            stage51_feature_fit_score=2,
            leakage_safety_score=3,
            effort_score=2,
            decision=(
                "Keep as a secondary option if sentence splitting proves too noisy "
                "in the next adapter dry run."
            ),
            strengths=(
                "Uses only approved local fields.",
                "May be more stable for long or poorly punctuated answers.",
            ),
            risks=(
                "Chunk candidates are less aligned with Stage 51 sentence evidence.",
                "Chunk boundaries need another protocol choice before metrics.",
            ),
        ),
        _option(
            label="source_row_single_candidate",
            status=_SECONDARY,
            coverage_percent=row_url_coverage,
            coverage_source="rows_with_row_url",
            citation_grounding_score=2,
            stage51_feature_fit_score=1,
            leakage_safety_score=3,
            effort_score=1,
            decision=(
                "Keep as a smoke-test adapter only; it is too coarse for a real "
                "Stage 51 reranker comparison."
            ),
            strengths=(
                "Simplest adapter shape.",
                "Useful for contract and serialization tests.",
            ),
            risks=(
                "Provides little candidate competition for Stage 51 reranking.",
                "Mostly repeats the Stage 58 source-row retrieval task.",
            ),
        ),
        _option(
            label="linked_learn_document_candidates",
            status=_BLOCKED,
            coverage_percent=processed_learn_coverage,
            coverage_source="rows_with_processed_answer_learn_link",
            citation_grounding_score=3,
            stage51_feature_fit_score=3,
            leakage_safety_score=2,
            effort_score=3,
            decision=(
                "Do not implement as the next protocol because link coverage is "
                "incomplete and page fetching is outside Stage 60."
            ),
            strengths=(
                "Closest option to document-grounded evidence when links exist.",
                "May support a future smaller document-grounded MSQA subset.",
            ),
            risks=(
                "Would require external page retrieval or a new local corpus.",
                "Cannot cover the frozen split without an unapproved fallback.",
            ),
        ),
        _option(
            label="question_answer_text_candidates",
            status=_REJECTED,
            coverage_percent=row_url_coverage,
            coverage_source="rows_with_row_url",
            citation_grounding_score=1,
            stage51_feature_fit_score=2,
            leakage_safety_score=0,
            effort_score=1,
            decision=(
                "Reject because question text must not be part of the comparison "
                "candidate index."
            ),
            strengths=("Easy to construct.",),
            risks=(
                "Repeats the Stage 58 diagnostic leakage issue.",
                "Not a fair candidate comparison target.",
            ),
        ),
    )


def _recommended_protocol(
    *,
    recommended_source: ProtocolOption,
    recommended_candidate: ProtocolOption,
) -> dict[str, Any]:
    return {
        "protocol_status": "draft_requires_user_confirmation",
        "source_citation_identity": recommended_source.label,
        "candidate_construction": recommended_candidate.label,
        "retrieval_corpus_scope": "frozen_split_only",
        "retrieval_index_text": "ProcessedAnswerText only",
        "excluded_index_text": "QuestionText",
        "gold_source_identity": {
            "question_id": "QuestionId",
            "answer_id": "AnswerId",
            "source_url": "Url",
        },
        "candidate_identity": (
            "QuestionId::processed_answer_sentence::<one_based_sentence_index>"
        ),
        "candidate_fields_required_for_next_adapter": [
            "question_id",
            "answer_id",
            "source_url",
            "candidate_id",
            "candidate_sentence",
            "retrieval_rank",
            "retrieval_score",
            "candidate_score",
            "source_row_id",
        ],
        "metrics_allowed_after_confirmation": [
            "source_row_hit@k",
            "source_row_mrr",
            "top1_answer_token_f1",
            "oracle_answer_token_f1@k",
            "source_citation_preservation_delta",
        ],
        "explicit_exclusions": [
            "Do not use AnswerText fallback.",
            "Do not use DoubleProcessedAnswerText fallback.",
            "Do not index QuestionText for candidate comparison.",
            "Do not require processed-answer links as citation ground truth.",
            "Do not fetch external pages in this protocol.",
            "Do not change the default runtime.",
        ],
    }


def _option(
    *,
    label: str,
    status: str,
    coverage_percent: float,
    coverage_source: str,
    citation_grounding_score: int,
    stage51_feature_fit_score: int,
    leakage_safety_score: int,
    effort_score: int,
    decision: str,
    strengths: Sequence[str],
    risks: Sequence[str],
) -> ProtocolOption:
    coverage_score = _coverage_score(coverage_percent)
    total_score = (
        coverage_score
        + citation_grounding_score
        + stage51_feature_fit_score
        + leakage_safety_score
        - effort_score
    )
    return ProtocolOption(
        label=label,
        status=status,
        coverage_percent=round(coverage_percent, 3),
        coverage_source=coverage_source,
        coverage_score=coverage_score,
        citation_grounding_score=citation_grounding_score,
        stage51_feature_fit_score=stage51_feature_fit_score,
        leakage_safety_score=leakage_safety_score,
        effort_score=effort_score,
        total_score=total_score,
        decision=decision,
        strengths=tuple(strengths),
        risks=tuple(risks),
    )


def _coverage_percent(coverage: Mapping[str, Any], key: str) -> float:
    return float(coverage[key]["percent"])


def _coverage_score(percent: float) -> int:
    if percent >= 99:
        return 3
    if percent >= 60:
        return 2
    if percent > 0:
        return 1
    return 0


def _recommended_option(options: Sequence[ProtocolOption]) -> ProtocolOption:
    recommended = [option for option in options if option.status == _RECOMMENDED]
    if len(recommended) != 1:
        raise ValueError("Expected exactly one recommended option")
    return recommended[0]


def _option_score_bars(options: Sequence[Mapping[str, Any]]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(option["label"]),
            value=float(option["total_score"]),
            value_label=f"{option['total_score']} ({option['status']})",
        )
        for option in options
    ]


def _coverage_bars(options: Sequence[Mapping[str, Any]]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(option["label"]),
            value=float(option["coverage_percent"]),
            value_label=f"{option['coverage_percent']}%",
        )
        for option in options
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
