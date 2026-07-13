from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from ts_rag_agent.application.text_metrics import token_f1

DIRECT_ANSWER_ROUTES = frozenset({"how_to_or_lookup", "install_upgrade_config"})
CITATION_SENSITIVE_ROUTES = frozenset(
    {
        "security_bulletin_vulnerability_detail",
        "security_bulletin_affected_product",
        "security_bulletin_remediation",
        "security_bulletin_post_fix_behavior",
        "limitation_or_restriction",
    }
)
ANSWER_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"answer|apply|configure|corrective action|disable|download|enable|fix|"
    r"install|local fix|must|resolution|resolving the problem|set|should|"
    r"solution|use|workaround"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompositionPolicyCandidate:
    """One selected evidence candidate from an answer-gap report."""

    document_id: str
    title: str
    retrieval_rank: int | None
    sentence: str
    candidate_score: float


@dataclass(frozen=True)
class CompositionPolicyDecision:
    """One route-aware answer-composition decision and its measured outcome."""

    question_id: str
    question_route: str
    strategy: str
    reason: str
    baseline_candidate_count: int
    policy_candidate_count: int
    baseline_document_count: int
    policy_document_count: int
    baseline_answer_token_f1: float
    policy_answer_token_f1: float
    f1_delta: float
    baseline_gold_cited: bool
    policy_gold_cited: bool
    citation_delta: int
    baseline_doc_ids: list[str]
    policy_doc_ids: list[str]
    question_title: str


@dataclass(frozen=True)
class RouteCompositionSummary:
    """Aggregate policy result for one question route."""

    route: str
    total_cases: int
    average_baseline_f1: float
    average_policy_f1: float
    average_f1_delta: float
    baseline_gold_citation_count: int
    policy_gold_citation_count: int
    citation_delta: int
    strategy_counts: dict[str, int]


@dataclass(frozen=True)
class RouteAwareCompositionSummary:
    """Aggregate policy result over the analyzed answer-gap report."""

    total_cases: int
    average_baseline_f1: float
    average_policy_f1: float
    average_f1_delta: float
    baseline_gold_citation_count: int
    policy_gold_citation_count: int
    citation_delta: int
    baseline_gold_citation_rate: float
    policy_gold_citation_rate: float
    changed_answer_count: int
    f1_improved_count: int
    f1_regressed_count: int
    citation_lost_count: int
    citation_gained_count: int
    strategy_counts: dict[str, int]
    route_summaries: list[RouteCompositionSummary]
    accepted_for_runtime_experiment: bool
    acceptance_reason: str


@dataclass(frozen=True)
class RouteAwareCompositionResult:
    """Full route-aware answer-composition policy analysis."""

    policy_name: str
    summary: RouteAwareCompositionSummary
    changed_cases: list[CompositionPolicyDecision]
    f1_regression_cases: list[CompositionPolicyDecision]
    citation_loss_cases: list[CompositionPolicyDecision]


class RouteAwareCompositionPolicy:
    """Conservative route-aware policy for choosing answer evidence sentences."""

    name = "route_aware_top1_direct_otherwise_top3"

    def __init__(
        self,
        strong_first_score_min: float = 100.0,
        strong_first_score_ratio_min: float = 1.15,
        strong_first_score_margin_min: float = 20.0,
        max_top1_retrieval_rank: int = 3,
        duplicate_threshold: float = 0.96,
    ) -> None:
        if strong_first_score_min < 0:
            raise ValueError("strong_first_score_min must be non-negative")
        if strong_first_score_ratio_min < 1:
            raise ValueError("strong_first_score_ratio_min must be at least 1")
        if strong_first_score_margin_min < 0:
            raise ValueError("strong_first_score_margin_min must be non-negative")
        if max_top1_retrieval_rank <= 0:
            raise ValueError("max_top1_retrieval_rank must be positive")
        if not 0 <= duplicate_threshold <= 1:
            raise ValueError("duplicate_threshold must be between 0 and 1")

        self.strong_first_score_min = strong_first_score_min
        self.strong_first_score_ratio_min = strong_first_score_ratio_min
        self.strong_first_score_margin_min = strong_first_score_margin_min
        self.max_top1_retrieval_rank = max_top1_retrieval_rank
        self.duplicate_threshold = duplicate_threshold

    def select(
        self,
        question_route: str,
        candidates: list[CompositionPolicyCandidate],
    ) -> tuple[list[CompositionPolicyCandidate], str, str]:
        """Select answer candidates without using gold labels."""

        baseline_candidates = candidates[:3]
        if not baseline_candidates:
            return [], "empty", "no candidates"

        if question_route in CITATION_SENSITIVE_ROUTES:
            return (
                baseline_candidates,
                "keep_top3_citation_sensitive",
                f"{question_route} keeps top3 to protect citation coverage",
            )

        if (
            question_route in DIRECT_ANSWER_ROUTES
            and self._has_strong_first_candidate(baseline_candidates)
        ):
            return (
                baseline_candidates[:1],
                "top1_direct_strong_signal",
                f"{question_route} top1 has strong score, rank, and answer signal",
            )

        deduplicated = _deduplicate_same_document_candidates(
            baseline_candidates,
            duplicate_threshold=self.duplicate_threshold,
        )
        if len(deduplicated) < len(baseline_candidates):
            return (
                deduplicated,
                "dedup_same_document",
                "removed near-duplicate same-document evidence sentences",
            )

        return baseline_candidates, "keep_top3_default", "default keeps top3"

    def _has_strong_first_candidate(
        self,
        candidates: list[CompositionPolicyCandidate],
    ) -> bool:
        first = candidates[0]
        if first.candidate_score < self.strong_first_score_min:
            return False
        if first.retrieval_rank is None or first.retrieval_rank > self.max_top1_retrieval_rank:
            return False
        if not ANSWER_SIGNAL_PATTERN.search(first.sentence):
            return False
        if len(candidates) == 1:
            return True

        second_score = max(candidates[1].candidate_score, 0.0)
        if second_score == 0:
            return True
        score_ratio = first.candidate_score / second_score
        score_margin = first.candidate_score - second_score
        return (
            score_ratio >= self.strong_first_score_ratio_min
            or score_margin >= self.strong_first_score_margin_min
        )


def analyze_route_aware_composition_policy(
    answer_gap_report: dict[str, Any],
    policy: RouteAwareCompositionPolicy | None = None,
    min_average_f1_gain: float = 0.002,
    max_allowed_citation_loss: int = 2,
    sample_limit_per_bucket: int = 20,
) -> RouteAwareCompositionResult:
    """Evaluate a route-aware composition policy over an answer-gap report."""

    if min_average_f1_gain < 0:
        raise ValueError("min_average_f1_gain must be non-negative")
    if max_allowed_citation_loss < 0:
        raise ValueError("max_allowed_citation_loss must be non-negative")
    if sample_limit_per_bucket < 0:
        raise ValueError("sample_limit_per_bucket must be non-negative")

    raw_cases = answer_gap_report.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("answer_gap_report must contain a list field named 'cases'")

    active_policy = policy or RouteAwareCompositionPolicy()
    decisions = [
        _evaluate_case(raw_case, active_policy)
        for raw_case in raw_cases
        if isinstance(raw_case, dict)
    ]
    summary = _build_summary(
        decisions=decisions,
        min_average_f1_gain=min_average_f1_gain,
        max_allowed_citation_loss=max_allowed_citation_loss,
    )
    return RouteAwareCompositionResult(
        policy_name=active_policy.name,
        summary=summary,
        changed_cases=_select_cases(
            decisions,
            predicate=lambda decision: decision.strategy
            not in {"keep_top3_default", "keep_top3_citation_sensitive"},
            sort_key=lambda decision: (-abs(decision.f1_delta), decision.question_id),
            sample_limit=sample_limit_per_bucket,
        ),
        f1_regression_cases=_select_cases(
            decisions,
            predicate=lambda decision: decision.f1_delta < 0,
            sort_key=lambda decision: (decision.f1_delta, decision.question_id),
            sample_limit=sample_limit_per_bucket,
        ),
        citation_loss_cases=_select_cases(
            decisions,
            predicate=lambda decision: decision.citation_delta < 0,
            sort_key=lambda decision: (decision.f1_delta, decision.question_id),
            sample_limit=sample_limit_per_bucket,
        ),
    )


def route_aware_composition_result_to_dict(
    result: RouteAwareCompositionResult,
) -> dict[str, Any]:
    """Convert route-aware policy analysis to a JSON-safe dictionary."""

    return {
        "policy_name": result.policy_name,
        "summary": {
            **asdict(result.summary),
            "route_summaries": [
                asdict(route_summary)
                for route_summary in result.summary.route_summaries
            ],
        },
        "changed_cases": [asdict(decision) for decision in result.changed_cases],
        "f1_regression_cases": [
            asdict(decision) for decision in result.f1_regression_cases
        ],
        "citation_loss_cases": [
            asdict(decision) for decision in result.citation_loss_cases
        ],
    }


def _evaluate_case(
    raw_case: dict[str, Any],
    policy: RouteAwareCompositionPolicy,
) -> CompositionPolicyDecision:
    question_route = str(raw_case.get("question_route", ""))
    candidates = _load_candidates(raw_case)
    baseline_candidates = candidates[:3]
    policy_candidates, strategy, reason = policy.select(question_route, candidates)
    gold_answer = str(raw_case.get("gold_answer", ""))
    gold_answer_doc_id = str(raw_case.get("gold_answer_doc_id", ""))

    baseline_answer = _join_candidate_sentences(baseline_candidates)
    policy_answer = _join_candidate_sentences(policy_candidates)
    baseline_f1 = _case_baseline_f1(raw_case, baseline_answer, gold_answer)
    policy_f1 = token_f1(policy_answer, gold_answer)
    baseline_gold_cited = _contains_document_id(baseline_candidates, gold_answer_doc_id)
    policy_gold_cited = _contains_document_id(policy_candidates, gold_answer_doc_id)

    return CompositionPolicyDecision(
        question_id=str(raw_case.get("question_id", "")),
        question_route=question_route,
        strategy=strategy,
        reason=reason,
        baseline_candidate_count=len(baseline_candidates),
        policy_candidate_count=len(policy_candidates),
        baseline_document_count=len(
            {candidate.document_id for candidate in baseline_candidates}
        ),
        policy_document_count=len({candidate.document_id for candidate in policy_candidates}),
        baseline_answer_token_f1=round(baseline_f1, 4),
        policy_answer_token_f1=round(policy_f1, 4),
        f1_delta=round(policy_f1 - baseline_f1, 4),
        baseline_gold_cited=baseline_gold_cited,
        policy_gold_cited=policy_gold_cited,
        citation_delta=int(policy_gold_cited) - int(baseline_gold_cited),
        baseline_doc_ids=[candidate.document_id for candidate in baseline_candidates],
        policy_doc_ids=[candidate.document_id for candidate in policy_candidates],
        question_title=str(raw_case.get("question_title", "")),
    )


def _load_candidates(raw_case: dict[str, Any]) -> list[CompositionPolicyCandidate]:
    raw_candidates = raw_case.get("selected_candidates", [])
    if not isinstance(raw_candidates, list):
        return []

    candidates = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        candidates.append(
            CompositionPolicyCandidate(
                document_id=str(raw_candidate.get("document_id", "")),
                title=str(raw_candidate.get("title", "")),
                retrieval_rank=_safe_int(raw_candidate.get("retrieval_rank")),
                sentence=str(raw_candidate.get("sentence", "")),
                candidate_score=float(raw_candidate.get("candidate_score", 0.0)),
            )
        )
    return candidates


def _build_summary(
    decisions: list[CompositionPolicyDecision],
    min_average_f1_gain: float,
    max_allowed_citation_loss: int,
) -> RouteAwareCompositionSummary:
    baseline_citation_count = sum(1 for decision in decisions if decision.baseline_gold_cited)
    policy_citation_count = sum(1 for decision in decisions if decision.policy_gold_cited)
    citation_delta = policy_citation_count - baseline_citation_count
    f1_gain = _average(decision.policy_answer_token_f1 for decision in decisions) - _average(
        decision.baseline_answer_token_f1 for decision in decisions
    )
    accepted = f1_gain >= min_average_f1_gain and citation_delta >= -max_allowed_citation_loss
    acceptance_reason = (
        "accepted for runtime experiment: F1 gain meets threshold and citation loss is bounded"
        if accepted
        else "rejected: F1 gain is too small or citation loss is too large"
    )

    return RouteAwareCompositionSummary(
        total_cases=len(decisions),
        average_baseline_f1=_average(
            decision.baseline_answer_token_f1 for decision in decisions
        ),
        average_policy_f1=_average(decision.policy_answer_token_f1 for decision in decisions),
        average_f1_delta=round(f1_gain, 4),
        baseline_gold_citation_count=baseline_citation_count,
        policy_gold_citation_count=policy_citation_count,
        citation_delta=citation_delta,
        baseline_gold_citation_rate=_safe_rate(baseline_citation_count, len(decisions)),
        policy_gold_citation_rate=_safe_rate(policy_citation_count, len(decisions)),
        changed_answer_count=sum(
            1
            for decision in decisions
            if decision.strategy not in {"keep_top3_default", "keep_top3_citation_sensitive"}
        ),
        f1_improved_count=sum(1 for decision in decisions if decision.f1_delta > 0),
        f1_regressed_count=sum(1 for decision in decisions if decision.f1_delta < 0),
        citation_lost_count=sum(1 for decision in decisions if decision.citation_delta < 0),
        citation_gained_count=sum(1 for decision in decisions if decision.citation_delta > 0),
        strategy_counts=dict(Counter(decision.strategy for decision in decisions)),
        route_summaries=_build_route_summaries(decisions),
        accepted_for_runtime_experiment=accepted,
        acceptance_reason=acceptance_reason,
    )


def _build_route_summaries(
    decisions: list[CompositionPolicyDecision],
) -> list[RouteCompositionSummary]:
    decisions_by_route: dict[str, list[CompositionPolicyDecision]] = defaultdict(list)
    for decision in decisions:
        decisions_by_route[decision.question_route].append(decision)

    route_summaries = []
    for route, route_decisions in sorted(decisions_by_route.items()):
        baseline_citation_count = sum(
            1 for decision in route_decisions if decision.baseline_gold_cited
        )
        policy_citation_count = sum(
            1 for decision in route_decisions if decision.policy_gold_cited
        )
        route_summaries.append(
            RouteCompositionSummary(
                route=route,
                total_cases=len(route_decisions),
                average_baseline_f1=_average(
                    decision.baseline_answer_token_f1 for decision in route_decisions
                ),
                average_policy_f1=_average(
                    decision.policy_answer_token_f1 for decision in route_decisions
                ),
                average_f1_delta=round(
                    _average(decision.policy_answer_token_f1 for decision in route_decisions)
                    - _average(
                        decision.baseline_answer_token_f1 for decision in route_decisions
                    ),
                    4,
                ),
                baseline_gold_citation_count=baseline_citation_count,
                policy_gold_citation_count=policy_citation_count,
                citation_delta=policy_citation_count - baseline_citation_count,
                strategy_counts=dict(Counter(decision.strategy for decision in route_decisions)),
            )
        )
    return route_summaries


def _select_cases(
    decisions: list[CompositionPolicyDecision],
    predicate,
    sort_key,
    sample_limit: int,
) -> list[CompositionPolicyDecision]:
    if sample_limit == 0:
        return []
    selected = [decision for decision in decisions if predicate(decision)]
    return sorted(selected, key=sort_key)[:sample_limit]


def _case_baseline_f1(raw_case: dict[str, Any], baseline_answer: str, gold_answer: str) -> float:
    return token_f1(baseline_answer, gold_answer)


def _join_candidate_sentences(candidates: list[CompositionPolicyCandidate]) -> str:
    return " ".join(candidate.sentence for candidate in candidates if candidate.sentence)


def _contains_document_id(
    candidates: list[CompositionPolicyCandidate],
    document_id: str,
) -> bool:
    return bool(document_id) and any(
        candidate.document_id == document_id for candidate in candidates
    )


def _deduplicate_same_document_candidates(
    candidates: list[CompositionPolicyCandidate],
    duplicate_threshold: float,
) -> list[CompositionPolicyCandidate]:
    kept = []
    kept_by_document: dict[str, list[set[str]]] = defaultdict(list)
    for candidate in candidates:
        token_set = set(_tokens(candidate.sentence))
        if not token_set:
            continue
        document_token_sets = kept_by_document[candidate.document_id]
        if any(
            _jaccard(token_set, existing) >= duplicate_threshold
            for existing in document_token_sets
        ):
            continue
        kept.append(candidate)
        document_token_sets.append(token_set)
    return kept


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_+#]+(?:[.+#-][a-z0-9_+#]+)*", text.lower())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _average(values) -> float:
    materialized_values = list(values)
    if not materialized_values:
        return 0.0
    return round(sum(materialized_values) / len(materialized_values), 4)


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
