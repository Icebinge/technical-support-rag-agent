from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application.evidence_selection import SentenceEvidenceCandidate

_EQUALITY_TOLERANCE = 1e-12
_NUMERIC_CANDIDATE_FEATURES = (
    "retrieval_rank",
    "retrieval_score",
    "candidate_score",
    "candidate_token_count",
    "candidate_sentence_count",
    "query_overlap_count",
    "query_overlap_ratio",
    "candidate_query_coverage_ratio",
    "title_query_overlap_count",
    "title_query_overlap_ratio",
    "answer_signal_score",
    "problem_noise_score",
    "symbol_ratio",
)
_BOOLEAN_CANDIDATE_FEATURES = (
    "has_answer_heading",
    "has_problem_heading",
    "has_question_heading",
    "has_url",
    "has_trace_noise",
)


@dataclass(frozen=True)
class CompositionAction:
    """One unique, runtime-executable sentence-selection modification."""

    action_id: str
    family: str
    aliases: tuple[str, ...]
    selected_indices: tuple[int, ...]
    matches_stage180: bool


@dataclass(frozen=True)
class ActionAuditRow:
    """One labeled action used only by the train-only offline audit."""

    question_key: str
    fold_id: str
    route: str
    action: CompositionAction
    runtime_features: Mapping[str, Any]
    outcome_class: str
    strict_expected: bool
    citation_delta: int
    f1_delta: float


@dataclass(frozen=True)
class ActionOOFPrediction:
    """Held-out probability for one action in a frozen question fold."""

    row: ActionAuditRow
    probability: float


def enumerate_atomic_composition_actions(
    *,
    candidates: Sequence[SentenceEvidenceCandidate],
    stage180_selected_indices: Sequence[int],
    max_sentences: int = 3,
    alternate_limit: int = 12,
) -> tuple[CompositionAction, ...]:
    """Enumerate and deduplicate the frozen Stage 181 atomic action family."""

    if max_sentences <= 0:
        raise ValueError("max_sentences must be positive")
    if alternate_limit <= 0:
        raise ValueError("alternate_limit must be positive")
    if any(index < 0 or index >= len(candidates) for index in stage180_selected_indices):
        raise ValueError("Stage180 selected index is outside the candidate pool")

    baseline = tuple(range(min(max_sentences, len(candidates))))
    actions: dict[tuple[int, ...], dict[str, Any]] = {}

    def add(indices: Sequence[int], family: str, *, stage180: bool = False) -> None:
        selected = tuple(indices)
        empty_control = not selected and family in {"baseline", "stage180_selected"}
        if (
            (not selected and not empty_control)
            or len(selected) > max_sentences
            or len(set(selected)) != len(selected)
        ):
            return
        existing = actions.get(selected)
        if existing is None:
            actions[selected] = {
                "family": family,
                "aliases": [family],
                "matches_stage180": stage180,
            }
            return
        if family not in existing["aliases"]:
            existing["aliases"].append(family)
        existing["matches_stage180"] = existing["matches_stage180"] or stage180

    add(baseline, "baseline")
    for slot in range(len(baseline)):
        add(
            baseline[:slot] + baseline[slot + 1 :],
            f"delete_slot_{slot + 1}",
        )

    alternate_stop = min(len(candidates), alternate_limit)
    alternates = tuple(index for index in range(alternate_stop) if index not in baseline)
    for slot in range(len(baseline)):
        for alternate in alternates:
            replacement = list(baseline)
            replacement[slot] = alternate
            add(replacement, f"replace_slot_{slot + 1}")

    if len(baseline) < max_sentences:
        for alternate in alternates:
            add((*baseline, alternate), "append_candidate")

    if baseline:
        add(baseline[:1], "keep_prefix_1")
    if len(baseline) >= 2:
        add(baseline[:2], "keep_prefix_2")

    coverage = _distinct_document_indices(candidates, max_sentences=max_sentences)
    add(coverage, "document_coverage")
    lead_coverage = _lead_preserving_document_indices(
        candidates,
        baseline=baseline,
        max_sentences=max_sentences,
    )
    add(lead_coverage, "lead_preserving_document_coverage")
    add(stage180_selected_indices, "stage180_selected", stage180=True)

    return tuple(
        CompositionAction(
            action_id=f"action_{index:03d}",
            family=str(raw["family"]),
            aliases=tuple(str(alias) for alias in raw["aliases"]),
            selected_indices=indices,
            matches_stage180=bool(raw["matches_stage180"]),
        )
        for index, (indices, raw) in enumerate(actions.items())
    )


def build_action_runtime_features(
    *,
    action: CompositionAction,
    candidates: Sequence[SentenceEvidenceCandidate],
    candidate_runtime_features: Sequence[Mapping[str, Any]],
    route: str,
    max_sentences: int = 3,
) -> dict[str, Any]:
    """Aggregate runtime-safe sentence features into an action feature row."""

    if len(candidates) != len(candidate_runtime_features):
        raise ValueError("candidate features must align with candidates")
    baseline = tuple(range(min(max_sentences, len(candidates))))
    selected = action.selected_indices
    added = tuple(index for index in selected if index not in baseline)
    removed = tuple(index for index in baseline if index not in selected)
    selected_features = [candidate_runtime_features[index] for index in selected]
    added_features = [candidate_runtime_features[index] for index in added]
    selected_document_ids = {candidates[index].retrieval_result.document.id for index in selected}
    baseline_document_ids = {candidates[index].retrieval_result.document.id for index in baseline}
    features: dict[str, Any] = {
        "question_route": route,
        "action_family": action.family,
        "selected_sentence_count": len(selected),
        "added_sentence_count": len(added),
        "removed_sentence_count": len(removed),
        "modified_sentence_count": len(added) + len(removed),
        "preserves_baseline_lead": bool(baseline and baseline[0] in selected),
        "selected_document_count": len(selected_document_ids),
        "baseline_document_count": len(baseline_document_ids),
        "document_count_delta": len(selected_document_ids) - len(baseline_document_ids),
        "matches_stage180": action.matches_stage180,
        "is_delete": action.family.startswith("delete_slot"),
        "is_replace": action.family.startswith("replace_slot"),
        "is_prefix": action.family.startswith("keep_prefix"),
        "is_coverage": "coverage" in action.family,
    }
    for slot in range(max_sentences):
        features[f"removed_slot_{slot + 1}"] = slot in removed
    for alias in action.aliases:
        features[f"alias={alias}"] = True
    _add_rank_features(features, "selected", selected)
    _add_rank_features(features, "added", added)
    _add_aggregated_candidate_features(features, "selected", selected_features)
    _add_aggregated_candidate_features(features, "added", added_features)
    return features


def classify_action_outcome(*, citation_delta: int, f1_delta: float) -> tuple[str, bool]:
    """Assign the frozen strict Stage 181 outcome taxonomy."""

    f1_direction = _direction(f1_delta)
    if citation_delta > 0 and f1_direction > 0:
        return "dual_gain", True
    if citation_delta > 0 and f1_direction == 0:
        return "citation_gain_f1_tied", True
    if citation_delta == 0 and f1_direction > 0:
        return "f1_gain_citation_preserved", True
    if citation_delta > 0:
        return "citation_gain_f1_loss", False
    if citation_delta < 0:
        return "citation_loss", False
    if f1_direction < 0:
        return "citation_preserved_f1_loss", False
    return "neutral", False


def run_action_predictability_oof(
    rows: Sequence[ActionAuditRow],
    *,
    total_question_count: int | None = None,
) -> dict[str, Any]:
    """Fit the fixed classifier in five question-grouped OOF folds."""

    nonbaseline = [row for row in rows if row.action.family != "baseline"]
    fold_ids = sorted({row.fold_id for row in nonbaseline})
    if len(fold_ids) != 5:
        raise ValueError("action predictability requires five frozen folds")
    predictions: list[ActionOOFPrediction] = []
    fold_reports = {}
    for fold_id in fold_ids:
        train_rows = [row for row in nonbaseline if row.fold_id != fold_id]
        heldout_rows = [row for row in nonbaseline if row.fold_id == fold_id]
        train_labels = [int(row.strict_expected) for row in train_rows]
        if len(set(train_labels)) != 2:
            raise ValueError(f"training fold {fold_id} does not contain both classes")
        vectorizer = DictVectorizer(sparse=True)
        train_matrix = vectorizer.fit_transform([dict(row.runtime_features) for row in train_rows])
        heldout_matrix = vectorizer.transform([dict(row.runtime_features) for row in heldout_rows])
        scaler = StandardScaler(with_mean=False)
        train_matrix = scaler.fit_transform(train_matrix)
        heldout_matrix = scaler.transform(heldout_matrix)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=181,
            solver="liblinear",
        )
        model.fit(
            train_matrix,
            train_labels,
            sample_weight=_question_balanced_weights(train_rows),
        )
        probabilities = model.predict_proba(heldout_matrix)[:, 1]
        fold_predictions = [
            ActionOOFPrediction(row=row, probability=float(probability))
            for row, probability in zip(heldout_rows, probabilities, strict=True)
        ]
        predictions.extend(fold_predictions)
        fold_reports[fold_id] = _prediction_metrics(fold_predictions)

    question_count = total_question_count or len({row.question_key for row in rows})
    return {
        "model": {
            "family": "dict_vectorized_class_balanced_logistic_regression",
            "fit_count": len(fold_ids),
            "random_state": 181,
            "gold_features_used": False,
        },
        "aggregate": _prediction_metrics(predictions),
        "folds": fold_reports,
        "question_ranking": _question_ranking_metrics(
            predictions, total_question_count=question_count
        ),
        "coverage_curve": _coverage_curve(predictions, total_question_count=question_count),
        "predictions": tuple(predictions),
    }


def summarize_action_rows(
    rows: Sequence[ActionAuditRow], *, total_question_count: int | None = None
) -> dict[str, Any]:
    """Build public-safe aggregate action and oracle summaries."""

    nonbaseline = [row for row in rows if row.action.family != "baseline"]
    all_question_keys = {row.question_key for row in rows}
    by_question: dict[str, list[ActionAuditRow]] = {
        question_key: [] for question_key in all_question_keys
    }
    for row in nonbaseline:
        by_question[row.question_key].append(row)
    expected_by_question = {
        key: [row for row in question_rows if row.strict_expected]
        for key, question_rows in by_question.items()
    }
    oracle_rows = [
        _oracle_action(question_rows)
        for question_rows in expected_by_question.values()
        if question_rows
    ]
    question_count = total_question_count or len(all_question_keys)
    return {
        "question_count": question_count,
        "questions_with_nonbaseline_action": sum(bool(rows) for rows in by_question.values()),
        "nonbaseline_action_count": len(nonbaseline),
        "strict_expected_action_count": sum(row.strict_expected for row in nonbaseline),
        "strict_expected_action_rate": _ratio(
            sum(row.strict_expected for row in nonbaseline), len(nonbaseline)
        ),
        "questions_with_strict_expected_action": sum(
            bool(question_rows) for question_rows in expected_by_question.values()
        ),
        "outcome_class_counts": dict(
            sorted(Counter(row.outcome_class for row in nonbaseline).items())
        ),
        "family_summaries": _group_summaries(nonbaseline, key=lambda row: row.action.family),
        "route_summaries": _group_summaries(nonbaseline, key=lambda row: row.route),
        "pattern_summaries": _group_summaries(nonbaseline, key=_modification_pattern),
        "oracle": {
            "selected_question_count": len(oracle_rows),
            "question_coverage": _ratio(len(oracle_rows), question_count),
            "gold_citation_delta": sum(row.citation_delta for row in oracle_rows),
            "mean_answerable_f1_delta": _mean_over_questions(
                oracle_rows, total_questions=question_count
            ),
            "selected_action_mean_f1_delta": _mean(row.f1_delta for row in oracle_rows),
            "outcome_class_counts": dict(
                sorted(Counter(row.outcome_class for row in oracle_rows).items())
            ),
        },
    }


def stage180_action_summary(rows: Sequence[ActionAuditRow]) -> dict[str, Any]:
    """Summarize the uniquely flagged reconstructed Stage 180 action per question."""

    selected = [row for row in rows if row.action.matches_stage180]
    counts = Counter(row.question_key for row in selected)
    if any(count != 1 for count in counts.values()):
        raise ValueError("each audited question must have exactly one Stage180 action")
    return {
        "question_count": len(selected),
        "strict_expected_count": sum(row.strict_expected for row in selected),
        "strict_expected_rate": _ratio(sum(row.strict_expected for row in selected), len(selected)),
        "gold_citation_delta": sum(row.citation_delta for row in selected),
        "mean_answerable_f1_delta": _mean(row.f1_delta for row in selected),
        "outcome_class_counts": dict(
            sorted(Counter(row.outcome_class for row in selected).items())
        ),
        "family_counts": dict(sorted(Counter(row.action.family for row in selected).items())),
    }


def _distinct_document_indices(
    candidates: Sequence[SentenceEvidenceCandidate], *, max_sentences: int
) -> tuple[int, ...]:
    selected = []
    seen_documents = set()
    for index, candidate in enumerate(candidates):
        document_id = candidate.retrieval_result.document.id
        if document_id in seen_documents:
            continue
        selected.append(index)
        seen_documents.add(document_id)
        if len(selected) == max_sentences:
            break
    return tuple(selected)


def _lead_preserving_document_indices(
    candidates: Sequence[SentenceEvidenceCandidate],
    *,
    baseline: Sequence[int],
    max_sentences: int,
) -> tuple[int, ...]:
    if not baseline:
        return ()
    selected = [baseline[0]]
    seen_documents = {candidates[baseline[0]].retrieval_result.document.id}
    for index, candidate in enumerate(candidates):
        if index == baseline[0]:
            continue
        document_id = candidate.retrieval_result.document.id
        if document_id in seen_documents:
            continue
        selected.append(index)
        seen_documents.add(document_id)
        if len(selected) == max_sentences:
            break
    return tuple(selected)


def _add_rank_features(features: dict[str, Any], prefix: str, indices: Sequence[int]) -> None:
    ranks = [index + 1 for index in indices]
    features[f"{prefix}_candidate_rank_min"] = min(ranks, default=0)
    features[f"{prefix}_candidate_rank_max"] = max(ranks, default=0)
    features[f"{prefix}_candidate_rank_mean"] = _mean(ranks)


def _add_aggregated_candidate_features(
    features: dict[str, Any],
    prefix: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for name in _NUMERIC_CANDIDATE_FEATURES:
        values = [float(row.get(name, 0.0)) for row in rows]
        features[f"{prefix}_{name}_mean"] = _mean(values)
        features[f"{prefix}_{name}_max"] = max(values, default=0.0)
        features[f"{prefix}_{name}_min"] = min(values, default=0.0)
    for name in _BOOLEAN_CANDIDATE_FEATURES:
        features[f"{prefix}_{name}_count"] = sum(bool(row.get(name, False)) for row in rows)


def _direction(value: float) -> int:
    if value > _EQUALITY_TOLERANCE:
        return 1
    if value < -_EQUALITY_TOLERANCE:
        return -1
    return 0


def _question_balanced_weights(rows: Sequence[ActionAuditRow]) -> list[float]:
    counts = Counter(row.question_key for row in rows)
    return [1.0 / counts[row.question_key] for row in rows]


def _prediction_metrics(predictions: Sequence[ActionOOFPrediction]) -> dict[str, Any]:
    labels = [int(row.row.strict_expected) for row in predictions]
    probabilities = [row.probability for row in predictions]
    prevalence = _ratio(sum(labels), len(labels))
    return {
        "action_count": len(predictions),
        "strict_expected_count": sum(labels),
        "prevalence": prevalence,
        "roc_auc": round(roc_auc_score(labels, probabilities), 6)
        if len(set(labels)) == 2
        else None,
        "average_precision": round(average_precision_score(labels, probabilities), 6)
        if any(labels)
        else None,
    }


def _question_ranking_metrics(
    predictions: Sequence[ActionOOFPrediction],
    *,
    total_question_count: int,
) -> dict[str, Any]:
    grouped: dict[str, list[ActionOOFPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    ranked = {
        key: sorted(
            rows,
            key=lambda row: (-row.probability, row.row.action.action_id),
        )
        for key, rows in grouped.items()
    }
    questions_with_expected = sum(
        any(row.row.strict_expected for row in rows) for rows in ranked.values()
    )
    metrics: dict[str, Any] = {
        "question_count": total_question_count,
        "questions_with_nonbaseline_action": len(ranked),
        "questions_with_strict_expected_action": questions_with_expected,
    }
    for cutoff in (1, 3, 5):
        hit_count = sum(
            any(row.row.strict_expected for row in rows[:cutoff]) for rows in ranked.values()
        )
        metrics[f"strict_expected_hit_at_{cutoff}_count"] = hit_count
        metrics[f"strict_expected_hit_at_{cutoff}_rate"] = _ratio(hit_count, total_question_count)
        metrics[f"conditional_recall_at_{cutoff}"] = _ratio(hit_count, questions_with_expected)
    top_rows = [rows[0].row for rows in ranked.values()]
    metrics["top1_strict_expected_precision"] = _ratio(
        sum(row.strict_expected for row in top_rows), len(top_rows)
    )
    metrics["top1_gold_citation_delta"] = sum(row.citation_delta for row in top_rows)
    metrics["top1_mean_answerable_f1_delta"] = _mean(row.f1_delta for row in top_rows)
    return metrics


def _coverage_curve(
    predictions: Sequence[ActionOOFPrediction],
    *,
    total_question_count: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ActionOOFPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    top_by_question = [
        max(rows, key=lambda row: (row.probability, row.row.action.action_id))
        for rows in grouped.values()
    ]
    top_by_question.sort(key=lambda row: (-row.probability, row.row.question_key))
    curve = []
    for fraction in (0.10, 0.25, 0.50, 1.00):
        target_count = max(1, math.ceil(total_question_count * fraction))
        selected_count = min(len(top_by_question), target_count)
        selected = [row.row for row in top_by_question[:selected_count]]
        fold_f1 = {
            fold_id: _mean(row.f1_delta for row in selected if row.fold_id == fold_id)
            for fold_id in sorted({row.fold_id for row in selected})
        }
        curve.append(
            {
                "target_question_coverage": fraction,
                "actual_question_coverage": _ratio(selected_count, total_question_count),
                "selected_question_count": selected_count,
                "strict_expected_count": sum(row.strict_expected for row in selected),
                "strict_expected_precision": _ratio(
                    sum(row.strict_expected for row in selected), selected_count
                ),
                "gold_citation_delta": sum(row.citation_delta for row in selected),
                "mean_answerable_f1_delta_all_questions": round(
                    sum(row.f1_delta for row in selected) / total_question_count, 6
                ),
                "selected_action_mean_f1_delta": _mean(row.f1_delta for row in selected),
                "fold_selected_action_mean_f1_delta": fold_f1,
                "f1_nonregressing_fold_count": sum(value >= 0 for value in fold_f1.values()),
            }
        )
    return curve


def _group_summaries(
    rows: Sequence[ActionAuditRow],
    *,
    key: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ActionAuditRow]] = defaultdict(list)
    for row in rows:
        grouped[str(key(row))].append(row)
    return {
        name: {
            "action_count": len(group_rows),
            "question_count": len({row.question_key for row in group_rows}),
            "strict_expected_count": sum(row.strict_expected for row in group_rows),
            "strict_expected_rate": _ratio(
                sum(row.strict_expected for row in group_rows), len(group_rows)
            ),
            "mean_citation_delta": _mean(row.citation_delta for row in group_rows),
            "mean_f1_delta": _mean(row.f1_delta for row in group_rows),
            "outcome_class_counts": dict(
                sorted(Counter(row.outcome_class for row in group_rows).items())
            ),
        }
        for name, group_rows in sorted(grouped.items())
    }


def _modification_pattern(row: ActionAuditRow) -> str:
    features = row.runtime_features
    if bool(features["preserves_baseline_lead"]):
        lead = "lead_preserved"
    else:
        lead = "lead_changed"
    return (
        f"{lead}|added={features['added_sentence_count']}|"
        f"removed={features['removed_sentence_count']}"
    )


def _oracle_action(rows: Sequence[ActionAuditRow]) -> ActionAuditRow:
    return max(
        rows,
        key=lambda row: (
            row.citation_delta,
            row.f1_delta,
            -int(row.runtime_features["modified_sentence_count"]),
            row.action.action_id,
        ),
    )


def _mean(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(statistics.fmean(materialized), 6)


def _mean_over_questions(rows: Sequence[ActionAuditRow], *, total_questions: int) -> float:
    if total_questions <= 0:
        return 0.0
    return round(sum(row.f1_delta for row in rows) / total_questions, 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)
