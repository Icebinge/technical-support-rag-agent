from ts_rag_agent.application.evidence_selection import SentenceEvidenceCandidate
from ts_rag_agent.application.local_window_gate_search import (
    DEFAULT_LOCAL_WINDOW_GATE_CONFIGS,
    LocalWindowGateCase,
    LocalWindowGateConfig,
    apply_local_window_gate,
    evaluate_local_window_gate_cases,
    should_replace_with_local_window,
    summarize_local_window_gate_search,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_should_replace_with_local_window_accepts_compact_answer_gain():
    config = _test_gate()

    assert should_replace_with_local_window(
        baseline_sentence="RESOLUTION Install the missing libraries.",
        local_sentence=(
            "RESOLUTION Install the missing libraries. You must restart the profile tool."
        ),
        config=config,
    )


def test_should_replace_with_local_window_rejects_problem_heading_noise():
    config = DEFAULT_LOCAL_WINDOW_GATE_CONFIGS[0]

    assert not should_replace_with_local_window(
        baseline_sentence="RESOLUTION Install the missing libraries.",
        local_sentence=(
            "SYMPTOM The profile tool opens to a blank panel. "
            "RESOLUTION Install the missing libraries."
        ),
        config=config,
    )


def test_apply_local_window_gate_only_replaces_matching_other_route_candidate():
    config = _test_gate()
    document = PrimeQADocument(
        id="gold",
        title="Profile blank panel",
        text="",
    )
    retrieval_result = RetrievalResult(document=document, score=10.0, rank=1)
    baseline = [
        _candidate(
            sentence="RESOLUTION Install the missing libraries.",
            retrieval_result=retrieval_result,
        )
    ]
    forced_local = [
        _candidate(
            sentence=(
                "RESOLUTION Install the missing libraries. "
                "You must restart the profile tool."
            ),
            retrieval_result=retrieval_result,
        )
    ]

    gated = apply_local_window_gate(
        baseline_candidates=baseline,
        forced_local_candidates=forced_local,
        question_route="other",
        config=config,
    )
    non_target = apply_local_window_gate(
        baseline_candidates=baseline,
        forced_local_candidates=forced_local,
        question_route="how_to_or_lookup",
        config=config,
    )

    assert "restart the profile tool" in gated[0].sentence
    assert non_target[0].sentence == baseline[0].sentence


def test_evaluate_local_window_gate_cases_summarizes_cross_source_stability():
    document = PrimeQADocument(id="gold", title="Profile blank panel", text="")
    retrieval_result = RetrievalResult(document=document, score=10.0, rank=1)
    config = LocalWindowGateConfig(
        name="test_gate",
        max_window_tokens=20,
        max_window_sentences=2,
        max_added_tokens=10,
        max_length_ratio=2.0,
        min_anchor_coverage=0.7,
        min_answer_signal_delta=0.0,
        block_problem_headings=True,
        block_question_heading=True,
        block_noise_growth=True,
    )

    dev_cases = evaluate_local_window_gate_cases(
        source_label="dev",
        questions=[
            _question(
                question_id="dev-q1",
                answer="Install the missing libraries. Restart the profile tool.",
            )
        ],
        baseline_candidates_by_question_id={
            "dev-q1": [
                _candidate(
                    "RESOLUTION Install the missing libraries.",
                    retrieval_result,
                )
            ]
        },
        forced_local_candidates_by_question_id={
            "dev-q1": [
                _candidate(
                    (
                        "RESOLUTION Install the missing libraries. "
                        "Restart the profile tool."
                    ),
                    retrieval_result,
                )
            ]
        },
        question_route_by_id={"dev-q1": "other"},
        gate_configs=(config,),
    )
    train_cases = evaluate_local_window_gate_cases(
        source_label="train",
        questions=[
            _question(
                question_id="train-q1",
                answer="Install the missing libraries. Restart the profile tool.",
            )
        ],
        baseline_candidates_by_question_id={
            "train-q1": [
                _candidate(
                    "RESOLUTION Install the missing libraries.",
                    retrieval_result,
                )
            ]
        },
        forced_local_candidates_by_question_id={
            "train-q1": [
                _candidate(
                    (
                        "RESOLUTION Install the missing libraries. "
                        "Restart the profile tool."
                    ),
                    retrieval_result,
                )
            ]
        },
        question_route_by_id={"train-q1": "other"},
        gate_configs=(config,),
    )

    analysis = summarize_local_window_gate_search(
        {"dev": dev_cases, "train": train_cases}
    )

    assert analysis.stable_gate_candidates == ["test_gate"]
    assert analysis.baseline_average_f1_by_source["dev"] < analysis.gate_summaries[0].average_f1


def test_stable_gate_candidate_requires_at_least_one_total_change():
    analysis = summarize_local_window_gate_search(
        {
            "dev": {"test_gate": [_case("dev", changed=False, delta=0.0)]},
            "train": {"test_gate": [_case("train", changed=False, delta=0.0)]},
        }
    )

    assert analysis.stable_gate_candidates == []


def _question(question_id: str, answer: str) -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id=question_id,
        title="Profile blank panel",
        text="",
        answer=answer,
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )


def _test_gate() -> LocalWindowGateConfig:
    return LocalWindowGateConfig(
        name="test_gate",
        max_window_tokens=30,
        max_window_sentences=2,
        max_added_tokens=20,
        max_length_ratio=3.0,
        min_anchor_coverage=0.7,
        min_answer_signal_delta=0.0,
        block_problem_headings=True,
        block_question_heading=True,
        block_noise_growth=True,
    )


def _candidate(
    sentence: str,
    retrieval_result: RetrievalResult,
) -> SentenceEvidenceCandidate:
    return SentenceEvidenceCandidate(
        sentence=sentence,
        retrieval_result=retrieval_result,
        score=10.0,
        overlap_terms=(),
    )


def _case(
    source_label: str,
    changed: bool,
    delta: float,
) -> LocalWindowGateCase:
    return LocalWindowGateCase(
        source_label=source_label,
        question_id=f"{source_label}-q1",
        question_route="other",
        baseline_f1=0.5,
        forced_local_f1=0.5 + delta,
        gated_f1=0.5 + delta,
        gated_delta_vs_baseline=delta,
        replacement_count=1 if changed else 0,
        changed=changed,
        baseline_gold_cited=True,
        gated_gold_cited=True,
    )
