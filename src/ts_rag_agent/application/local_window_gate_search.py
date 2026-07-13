from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean

from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    split_sentences,
    tokenize_text,
)
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.dataset import PrimeQAQuestion


@dataclass(frozen=True)
class LocalWindowGateConfig:
    """Runtime-visible safety gate for local-window candidate replacement."""

    name: str
    max_window_tokens: int
    max_window_sentences: int
    max_added_tokens: int
    max_length_ratio: float
    min_anchor_coverage: float
    min_answer_signal_delta: float
    block_problem_headings: bool
    block_question_heading: bool
    block_noise_growth: bool


@dataclass(frozen=True)
class LocalWindowGateCase:
    """One question evaluation for one local-window gate."""

    source_label: str
    question_id: str
    question_route: str
    baseline_f1: float
    forced_local_f1: float
    gated_f1: float
    gated_delta_vs_baseline: float
    replacement_count: int
    changed: bool
    baseline_gold_cited: bool
    gated_gold_cited: bool


@dataclass(frozen=True)
class LocalWindowGateSummary:
    """Aggregate local-window gate metrics."""

    gate_name: str
    source_label: str
    total_cases: int
    changed_cases: int
    replacement_count: int
    average_f1: float
    average_delta_vs_baseline: float
    average_delta_vs_forced_local: float
    baseline_gold_citation_count: int
    gated_gold_citation_count: int
    citation_delta_vs_baseline: int
    win_counts_vs_baseline: dict[str, int]


@dataclass(frozen=True)
class LocalWindowGateSearchAnalysis:
    """Cross-source local-window gate search result."""

    source_count: int
    total_cases: int
    baseline_average_f1_by_source: dict[str, float]
    forced_local_average_f1_by_source: dict[str, float]
    gate_summaries: list[LocalWindowGateSummary]
    stable_gate_candidates: list[str]
    top_summary: str


DEFAULT_LOCAL_WINDOW_GATE_CONFIGS = (
    LocalWindowGateConfig(
        name="strict_answer_gain_no_heading",
        max_window_tokens=80,
        max_window_sentences=2,
        max_added_tokens=30,
        max_length_ratio=1.6,
        min_anchor_coverage=0.75,
        min_answer_signal_delta=0.4,
        block_problem_headings=True,
        block_question_heading=True,
        block_noise_growth=True,
    ),
    LocalWindowGateConfig(
        name="moderate_answer_gain_no_problem",
        max_window_tokens=110,
        max_window_sentences=3,
        max_added_tokens=50,
        max_length_ratio=2.0,
        min_anchor_coverage=0.65,
        min_answer_signal_delta=0.2,
        block_problem_headings=True,
        block_question_heading=False,
        block_noise_growth=True,
    ),
    LocalWindowGateConfig(
        name="compact_same_signal_no_heading",
        max_window_tokens=75,
        max_window_sentences=2,
        max_added_tokens=25,
        max_length_ratio=1.5,
        min_anchor_coverage=0.8,
        min_answer_signal_delta=0.0,
        block_problem_headings=True,
        block_question_heading=True,
        block_noise_growth=True,
    ),
    LocalWindowGateConfig(
        name="answer_heading_gain",
        max_window_tokens=100,
        max_window_sentences=3,
        max_added_tokens=50,
        max_length_ratio=2.2,
        min_anchor_coverage=0.6,
        min_answer_signal_delta=1.0,
        block_problem_headings=False,
        block_question_heading=False,
        block_noise_growth=False,
    ),
    LocalWindowGateConfig(
        name="shorter_same_signal_no_problem",
        max_window_tokens=90,
        max_window_sentences=3,
        max_added_tokens=0,
        max_length_ratio=1.0,
        min_anchor_coverage=0.7,
        min_answer_signal_delta=0.0,
        block_problem_headings=True,
        block_question_heading=False,
        block_noise_growth=True,
    ),
)


def evaluate_local_window_gate_cases(
    source_label: str,
    questions: list[PrimeQAQuestion],
    baseline_candidates_by_question_id: dict[str, list[SentenceEvidenceCandidate]],
    forced_local_candidates_by_question_id: dict[str, list[SentenceEvidenceCandidate]],
    question_route_by_id: dict[str, str],
    gate_configs: tuple[LocalWindowGateConfig, ...] = DEFAULT_LOCAL_WINDOW_GATE_CONFIGS,
) -> dict[str, list[LocalWindowGateCase]]:
    """Evaluate gate configs on full selected-candidate text."""

    if not gate_configs:
        raise ValueError("gate_configs must not be empty")

    answerable_questions = [question for question in questions if question.answerable]
    cases_by_gate = {config.name: [] for config in gate_configs}
    for question in answerable_questions:
        baseline_candidates = baseline_candidates_by_question_id.get(question.id, [])
        forced_local_candidates = forced_local_candidates_by_question_id.get(
            question.id,
            [],
        )
        question_route = question_route_by_id.get(question.id, "")
        baseline_f1 = _answer_f1(baseline_candidates, question.answer)
        forced_local_f1 = _answer_f1(forced_local_candidates, question.answer)
        baseline_gold_cited = _gold_cited(baseline_candidates, question.answer_doc_id)

        for config in gate_configs:
            gated_candidates = apply_local_window_gate(
                baseline_candidates=baseline_candidates,
                forced_local_candidates=forced_local_candidates,
                question_route=question_route,
                config=config,
            )
            replacement_count = _replacement_count(baseline_candidates, gated_candidates)
            gated_f1 = _answer_f1(gated_candidates, question.answer)
            cases_by_gate[config.name].append(
                LocalWindowGateCase(
                    source_label=source_label,
                    question_id=question.id,
                    question_route=question_route,
                    baseline_f1=baseline_f1,
                    forced_local_f1=forced_local_f1,
                    gated_f1=gated_f1,
                    gated_delta_vs_baseline=round(gated_f1 - baseline_f1, 4),
                    replacement_count=replacement_count,
                    changed=replacement_count > 0,
                    baseline_gold_cited=baseline_gold_cited,
                    gated_gold_cited=_gold_cited(gated_candidates, question.answer_doc_id),
                )
            )

    return cases_by_gate


def apply_local_window_gate(
    baseline_candidates: list[SentenceEvidenceCandidate],
    forced_local_candidates: list[SentenceEvidenceCandidate],
    question_route: str,
    config: LocalWindowGateConfig,
) -> list[SentenceEvidenceCandidate]:
    """Apply one local-window safety gate to selected candidates."""

    if question_route != "other":
        return baseline_candidates

    gated_candidates = []
    for index, baseline_candidate in enumerate(baseline_candidates):
        forced_candidate = (
            forced_local_candidates[index]
            if index < len(forced_local_candidates)
            else None
        )
        if forced_candidate is None or not _same_candidate_anchor(
            baseline_candidate,
            forced_candidate,
        ):
            gated_candidates.append(baseline_candidate)
            continue

        if should_replace_with_local_window(
            baseline_sentence=baseline_candidate.sentence,
            local_sentence=forced_candidate.sentence,
            config=config,
        ):
            gated_candidates.append(forced_candidate)
        else:
            gated_candidates.append(baseline_candidate)

    return gated_candidates


def should_replace_with_local_window(
    baseline_sentence: str,
    local_sentence: str,
    config: LocalWindowGateConfig,
) -> bool:
    """Return whether a local-window replacement passes the safety gate."""

    if baseline_sentence == local_sentence:
        return False

    baseline_tokens = tokenize_text(baseline_sentence)
    local_tokens = tokenize_text(local_sentence)
    if len(local_tokens) > config.max_window_tokens:
        return False
    if len(split_sentences(local_sentence)) > config.max_window_sentences:
        return False
    if len(local_tokens) - len(baseline_tokens) > config.max_added_tokens:
        return False
    if len(local_tokens) > max(1, len(baseline_tokens)) * config.max_length_ratio:
        return False

    if _anchor_coverage(baseline_sentence, local_sentence) < config.min_anchor_coverage:
        return False

    signal_delta = _answer_signal_score(local_sentence) - _answer_signal_score(
        baseline_sentence
    )
    if signal_delta < config.min_answer_signal_delta:
        return False

    if config.block_problem_headings and _has_problem_heading_noise(local_sentence):
        return False
    if config.block_question_heading and _has_question_heading_noise(local_sentence):
        return False
    if config.block_noise_growth and _noise_score(local_sentence) > _noise_score(
        baseline_sentence
    ):
        return False

    return True


def summarize_local_window_gate_search(
    cases_by_gate_by_source: dict[str, dict[str, list[LocalWindowGateCase]]],
) -> LocalWindowGateSearchAnalysis:
    """Summarize gate cases across source labels."""

    source_labels = sorted(cases_by_gate_by_source)
    if not source_labels:
        raise ValueError("cases_by_gate_by_source must not be empty")

    baseline_average_f1_by_source = {}
    forced_local_average_f1_by_source = {}
    for source_label in source_labels:
        first_gate_cases = next(iter(cases_by_gate_by_source[source_label].values()))
        baseline_average_f1_by_source[source_label] = _average(
            [case.baseline_f1 for case in first_gate_cases]
        )
        forced_local_average_f1_by_source[source_label] = _average(
            [case.forced_local_f1 for case in first_gate_cases]
        )

    gate_summaries = []
    gate_names = sorted(next(iter(cases_by_gate_by_source.values())).keys())
    for gate_name in gate_names:
        all_gate_cases = []
        for source_label in source_labels:
            source_cases = cases_by_gate_by_source[source_label][gate_name]
            gate_summaries.append(
                _summarize_cases(
                    gate_name=gate_name,
                    source_label=source_label,
                    cases=source_cases,
                )
            )
            all_gate_cases.extend(source_cases)

        gate_summaries.append(
            _summarize_cases(
                gate_name=gate_name,
                source_label="all",
                cases=all_gate_cases,
            )
        )

    stable_gate_candidates = _stable_gate_candidates(gate_summaries, source_labels)
    return LocalWindowGateSearchAnalysis(
        source_count=len(source_labels),
        total_cases=sum(
            len(next(iter(cases_by_gate_by_source[source_label].values())))
            for source_label in source_labels
        ),
        baseline_average_f1_by_source=baseline_average_f1_by_source,
        forced_local_average_f1_by_source=forced_local_average_f1_by_source,
        gate_summaries=sorted(
            gate_summaries,
            key=lambda summary: (
                summary.source_label != "all",
                -summary.average_delta_vs_baseline,
                summary.gate_name,
            ),
        ),
        stable_gate_candidates=stable_gate_candidates,
        top_summary=_top_summary(gate_summaries, stable_gate_candidates),
    )


def local_window_gate_search_analysis_to_dict(
    analysis: LocalWindowGateSearchAnalysis,
) -> dict:
    """Convert gate search analysis to a JSON-safe dictionary."""

    return {
        "source_count": analysis.source_count,
        "total_cases": analysis.total_cases,
        "baseline_average_f1_by_source": analysis.baseline_average_f1_by_source,
        "forced_local_average_f1_by_source": analysis.forced_local_average_f1_by_source,
        "gate_search_note": (
            "Analysis only: gate features use runtime-visible baseline/local "
            "candidate text, while F1 is used only for offline evaluation."
        ),
        "stable_gate_candidates": analysis.stable_gate_candidates,
        "top_summary": analysis.top_summary,
        "gate_summaries": [asdict(summary) for summary in analysis.gate_summaries],
    }


def _summarize_cases(
    gate_name: str,
    source_label: str,
    cases: list[LocalWindowGateCase],
) -> LocalWindowGateSummary:
    winner_counts = Counter(_winner_vs_baseline(case) for case in cases)
    return LocalWindowGateSummary(
        gate_name=gate_name,
        source_label=source_label,
        total_cases=len(cases),
        changed_cases=sum(case.changed for case in cases),
        replacement_count=sum(case.replacement_count for case in cases),
        average_f1=_average([case.gated_f1 for case in cases]),
        average_delta_vs_baseline=_average(
            [case.gated_delta_vs_baseline for case in cases]
        ),
        average_delta_vs_forced_local=_average(
            [case.gated_f1 - case.forced_local_f1 for case in cases]
        ),
        baseline_gold_citation_count=sum(case.baseline_gold_cited for case in cases),
        gated_gold_citation_count=sum(case.gated_gold_cited for case in cases),
        citation_delta_vs_baseline=(
            sum(case.gated_gold_cited for case in cases)
            - sum(case.baseline_gold_cited for case in cases)
        ),
        win_counts_vs_baseline=dict(winner_counts),
    )


def _stable_gate_candidates(
    gate_summaries: list[LocalWindowGateSummary],
    source_labels: list[str],
) -> list[str]:
    summaries_by_gate: dict[str, dict[str, LocalWindowGateSummary]] = defaultdict(dict)
    for summary in gate_summaries:
        summaries_by_gate[summary.gate_name][summary.source_label] = summary

    stable = []
    for gate_name, summaries_by_source in summaries_by_gate.items():
        if not set(source_labels).issubset(summaries_by_source):
            continue
        source_summaries = [summaries_by_source[label] for label in source_labels]
        if all(
            summary.average_delta_vs_baseline >= 0
            and summary.citation_delta_vs_baseline >= 0
            for summary in source_summaries
        ) and sum(summary.changed_cases for summary in source_summaries) > 0:
            stable.append(gate_name)
    return sorted(stable)


def _top_summary(
    gate_summaries: list[LocalWindowGateSummary],
    stable_gate_candidates: list[str],
) -> str:
    if stable_gate_candidates:
        return f"Stable gate candidate(s): {', '.join(stable_gate_candidates)}."
    overall_summaries = [
        summary for summary in gate_summaries if summary.source_label == "all"
    ]
    if not overall_summaries:
        return "No gate summaries were produced."
    best = max(
        overall_summaries,
        key=lambda summary: (
            summary.average_delta_vs_baseline,
            summary.changed_cases,
            summary.gate_name,
        ),
    )
    return (
        "No gate had non-negative delta on every source. "
        f"Best overall gate was {best.gate_name} with average delta "
        f"{best.average_delta_vs_baseline:+.4f}."
    )


def _same_candidate_anchor(
    baseline_candidate: SentenceEvidenceCandidate,
    forced_candidate: SentenceEvidenceCandidate,
) -> bool:
    return (
        baseline_candidate.retrieval_result.document.id
        == forced_candidate.retrieval_result.document.id
    )


def _answer_f1(candidates: list[SentenceEvidenceCandidate], gold_answer: str) -> float:
    return round(token_f1(" ".join(candidate.sentence for candidate in candidates), gold_answer), 4)


def _gold_cited(
    candidates: list[SentenceEvidenceCandidate],
    answer_doc_id: str | None,
) -> bool:
    if not answer_doc_id:
        return False
    return any(
        candidate.retrieval_result.document.id == answer_doc_id
        for candidate in candidates
    )


def _replacement_count(
    baseline_candidates: list[SentenceEvidenceCandidate],
    gated_candidates: list[SentenceEvidenceCandidate],
) -> int:
    return sum(
        1
        for baseline_candidate, gated_candidate in zip(
            baseline_candidates,
            gated_candidates,
            strict=False,
        )
        if baseline_candidate.sentence != gated_candidate.sentence
    )


def _winner_vs_baseline(case: LocalWindowGateCase) -> str:
    if case.gated_delta_vs_baseline >= 0.03:
        return "gated"
    if case.gated_delta_vs_baseline <= -0.03:
        return "baseline"
    if case.gated_gold_cited and not case.baseline_gold_cited:
        return "gated"
    if case.baseline_gold_cited and not case.gated_gold_cited:
        return "baseline"
    return "tie"


def _answer_signal_score(text: str) -> float:
    normalized = text.lower()
    score = 0.0
    if re.search(r"\b(resolving the problem|resolution|solution|answer)\b", normalized):
        score += 2.0
    if re.search(r"\b(workaround|fix|corrective action|local fix)\b", normalized):
        score += 1.2
    if re.search(
        r"\b(install|upgrade|configure|restart|set|enable|disable|apply|run|use)\b",
        normalized,
    ):
        score += 0.8
    if re.search(r"\b(required|must|should|recommended|supported)\b", normalized):
        score += 0.4
    return score


def _anchor_coverage(baseline_sentence: str, local_sentence: str) -> float:
    baseline_terms = _content_terms(tokenize_text(baseline_sentence))
    local_terms = _content_terms(tokenize_text(local_sentence))
    if not baseline_terms:
        return 0.0
    return len(baseline_terms & local_terms) / len(baseline_terms)


def _content_terms(tokens: list[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in _STOPWORDS and len(token) > 1
    }


def _has_problem_heading_noise(text: str) -> bool:
    return bool(
        re.search(
            r"\b(problem\(abstract\)|symptom|environment|"
            r"diagnosing the problem|collecting data|steps to reproduce)\b",
            text.lower(),
        )
    )


def _has_question_heading_noise(text: str) -> bool:
    return bool(re.search(r"\b(question|technote \(faq\))\b", text.lower()))


def _noise_score(text: str) -> float:
    normalized = text.lower()
    score = 0.0
    if _has_problem_heading_noise(text):
        score += 1.0
    if _has_question_heading_noise(text):
        score += 0.7
    if re.search(r"\b(trace|stack|dump|exception|heapdump|javacore)\b", normalized):
        score += 0.5
    if len(tokenize_text(text)) > 120:
        score += 0.8
    return score


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "with",
    "you",
}
