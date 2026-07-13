from ts_rag_agent.application.evidence_selection import (
    AnswerAwareBM25SentenceEvidenceSelector,
    BM25SentenceEvidenceSelector,
    HybridRoutingEvidenceSelector,
    OverlapSentenceEvidenceSelector,
    SectionSpanBM25SentenceEvidenceSelector,
    classify_question_route,
    create_sentence_evidence_selector,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_bm25_sentence_selector_uses_idf_weighting():
    question = PrimeQAQuestion(
        id="q1",
        title="How do I configure raretoken authentication?",
        text="",
        answer="Use raretoken authentication in the security panel.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    common_document = PrimeQADocument(
        id="common",
        title="Common setup",
        text=(
            "Configure authentication setup for authentication setup. "
            "Authentication setup configuration is common."
        ),
    )
    rare_document = PrimeQADocument(
        id="gold",
        title="Rare token auth",
        text="Use raretoken authentication in the security panel.",
    )
    retrieval_results = [
        RetrievalResult(document=common_document, score=10.0, rank=1),
        RetrievalResult(document=rare_document, score=9.0, rank=2),
        RetrievalResult(
            document=PrimeQADocument(
                id="common-2",
                title="Authentication setup",
                text="Configure authentication setup before product installation.",
            ),
            score=8.0,
            rank=3,
        ),
        RetrievalResult(
            document=PrimeQADocument(
                id="common-3",
                title="Authentication setup checklist",
                text="Authentication setup configuration depends on the user registry.",
            ),
            score=7.0,
            rank=4,
        ),
    ]

    overlap_candidates = OverlapSentenceEvidenceSelector(
        min_sentence_chars=8
    ).rank_sentence_candidates(question, retrieval_results)
    bm25_candidates = BM25SentenceEvidenceSelector(
        min_sentence_chars=8,
        max_candidates_per_document=1,
    ).rank_sentence_candidates(question, retrieval_results)

    assert overlap_candidates[0].retrieval_result.document.id == "common"
    assert bm25_candidates[0].retrieval_result.document.id == "gold"
    assert "raretoken" in bm25_candidates[0].overlap_terms


def test_bm25_sentence_selector_caps_candidates_per_document():
    question = PrimeQAQuestion(
        id="q1",
        title="Restart service A",
        text="",
        answer="Restart service A from the panel.",
        answerable=True,
        answer_doc_id="doc-a",
        doc_ids=["doc-a"],
    )
    dominant_document = PrimeQADocument(
        id="doc-a",
        title="Restart service A",
        text=(
            "Restart service A from the panel. "
            "Restart service A after upgrade. "
            "Restart service A when it is stopped."
        ),
    )
    other_document = PrimeQADocument(
        id="doc-b",
        title="Restart service B",
        text="Restart service B from a different panel.",
    )
    selector = BM25SentenceEvidenceSelector(
        min_sentence_chars=8,
        max_candidates_per_document=1,
    )

    candidates = selector.rank_sentence_candidates(
        question,
        [
            RetrievalResult(document=dominant_document, score=10.0, rank=1),
            RetrievalResult(document=other_document, score=9.0, rank=2),
        ],
    )

    doc_ids = [candidate.retrieval_result.document.id for candidate in candidates[:2]]
    assert doc_ids == ["doc-a", "doc-b"]


def test_answer_aware_selector_promotes_resolution_over_symptom():
    question = PrimeQAQuestion(
        id="q1",
        title="Unable to open profile on Redhat Linux",
        text="Getting GPF and javacore dump.",
        answer="Install the missing adwaita libraries.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    document = PrimeQADocument(
        id="gold",
        title="Profile fails to open",
        text=(
            "PROBLEM(ABSTRACT) Unable to open profile on Redhat Linux with GPF and "
            "javacore dump. RESOLVING THE PROBLEM Install the missing adwaita "
            "libraries."
        ),
    )
    retrieval_results = [RetrievalResult(document=document, score=10.0, rank=1)]

    bm25_candidates = BM25SentenceEvidenceSelector(
        min_sentence_chars=8,
        max_candidates_per_document=2,
    ).rank_sentence_candidates(question, retrieval_results)
    answer_aware_candidates = AnswerAwareBM25SentenceEvidenceSelector(
        min_sentence_chars=8,
        max_candidates_per_document=2,
    ).rank_sentence_candidates(question, retrieval_results)

    assert "PROBLEM(ABSTRACT)" in bm25_candidates[0].sentence
    assert "RESOLVING THE PROBLEM" in answer_aware_candidates[0].sentence


def test_selector_factory_creates_answer_aware_selector():
    selector = create_sentence_evidence_selector(
        selector_name="answer-aware",
        min_sentence_chars=8,
        max_candidates_per_document=2,
    )

    assert selector.name == "answer_aware_bm25_sentence"


def test_section_span_selector_prefers_answer_section_window():
    question = PrimeQAQuestion(
        id="q1",
        title="Unable to open profile on Redhat Linux",
        text="Getting GPF and javacore dump.",
        answer="Install the missing adwaita libraries and restart the profile tool.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    document = PrimeQADocument(
        id="gold",
        title="Profile fails to open",
        text=(
            "PROBLEM(ABSTRACT) Unable to open profile on Redhat Linux with GPF "
            "and javacore dump. "
            "RESOLVING THE PROBLEM Install the missing adwaita libraries. "
            "Restart the profile tool."
        ),
    )
    retrieval_results = [RetrievalResult(document=document, score=10.0, rank=1)]

    candidates = SectionSpanBM25SentenceEvidenceSelector(
        min_sentence_chars=8,
        max_candidates_per_document=3,
        max_window_sentences=2,
    ).rank_sentence_candidates(question, retrieval_results)

    assert "Install the missing adwaita libraries" in candidates[0].sentence
    assert "Restart the profile tool" in candidates[0].sentence
    assert "PROBLEM(ABSTRACT)" not in candidates[0].sentence


def test_selector_factory_creates_section_span_selector():
    selector = create_sentence_evidence_selector(
        selector_name="section-span",
        min_sentence_chars=8,
        max_candidates_per_document=2,
    )

    assert selector.name == "section_span_bm25_sentence"


def test_section_span_selector_promotes_security_bulletin_cve_span():
    question = PrimeQAQuestion(
        id="q1",
        title="Security Bulletin Java SDK CVE-2015-0410",
        text="",
        answer="CVEID: CVE-2015-0410 CVSS Base Score: 5.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    document = PrimeQADocument(
        id="gold",
        title="Security Bulletin",
        text=(
            "SUMMARY There are multiple vulnerabilities in IBM SDK Java Technology "
            "Edition used by the product. "
            "CVEID: CVE-2015-0410 DESCRIPTION: An unspecified vulnerability could "
            "allow a remote attacker to cause a denial of service. CVSS Base Score: 5."
        ),
    )
    retrieval_results = [RetrievalResult(document=document, score=10.0, rank=1)]

    candidates = SectionSpanBM25SentenceEvidenceSelector(
        min_sentence_chars=8,
        max_candidates_per_document=1,
        max_window_sentences=2,
    ).rank_sentence_candidates(question, retrieval_results)

    assert "SUMMARY" not in candidates[0].sentence
    assert (
        "CVEID: CVE-2015-0410" in candidates[0].sentence
        or "CVSS Base Score" in candidates[0].sentence
    )


def test_hybrid_routing_selector_uses_section_span_for_security_bulletins():
    question = PrimeQAQuestion(
        id="q1",
        title="Security Bulletin Java SDK CVE-2015-0410",
        text="",
        answer="CVEID: CVE-2015-0410 CVSS Base Score: 5.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    document = PrimeQADocument(
        id="gold",
        title="Security Bulletin",
        text=(
            "SUMMARY There are multiple vulnerabilities in IBM SDK Java Technology "
            "Edition used by the product. "
            "CVEID: CVE-2015-0410 DESCRIPTION: An unspecified vulnerability could "
            "allow a remote attacker to cause a denial of service. CVSS Base Score: 5."
        ),
    )
    retrieval_results = [RetrievalResult(document=document, score=10.0, rank=1)]

    candidates = HybridRoutingEvidenceSelector(
        min_sentence_chars=8,
        answer_aware_max_candidates_per_document=3,
        section_span_max_candidates_per_document=1,
    ).rank_sentence_candidates(question, retrieval_results)

    assert "SUMMARY" not in candidates[0].sentence
    assert (
        "CVEID: CVE-2015-0410" in candidates[0].sentence
        or "CVSS Base Score" in candidates[0].sentence
    )


def test_hybrid_routing_selector_uses_answer_aware_for_general_questions():
    question = PrimeQAQuestion(
        id="q1",
        title="Unable to open profile on Redhat Linux",
        text="Getting GPF and javacore dump.",
        answer="Install the missing adwaita libraries.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )
    document = PrimeQADocument(
        id="gold",
        title="Profile fails to open",
        text=(
            "PROBLEM(ABSTRACT) Unable to open profile on Redhat Linux with GPF "
            "and javacore dump. RESOLVING THE PROBLEM Install the missing "
            "adwaita libraries."
        ),
    )
    retrieval_results = [RetrievalResult(document=document, score=10.0, rank=1)]

    candidates = HybridRoutingEvidenceSelector(
        min_sentence_chars=8,
        answer_aware_max_candidates_per_document=3,
        section_span_max_candidates_per_document=1,
    ).rank_sentence_candidates(question, retrieval_results)

    assert "RESOLVING THE PROBLEM" in candidates[0].sentence


def test_hybrid_route_classification_does_not_use_gold_answer():
    question = PrimeQAQuestion(
        id="q1",
        title="General product question",
        text="",
        answer="CVEID: CVE-2015-0410 CVSS Base Score: 5.",
        answerable=True,
        answer_doc_id="gold",
        doc_ids=["gold"],
    )

    assert classify_question_route(question) == "other"


def test_selector_factory_creates_hybrid_routing_selector():
    selector = create_sentence_evidence_selector(
        selector_name="hybrid-routing",
        min_sentence_chars=8,
        max_candidates_per_document=3,
    )

    assert selector.name == "hybrid_routing_answer_aware_mcpd3_section_span_mcpd1"
