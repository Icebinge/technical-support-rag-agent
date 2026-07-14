from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.evidence_selection import (
    normalize_sentence,
    split_sentences,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, tokenize_text

_STAGE = "Stage 61"
_CREATED_AT = "2026-07-14"
_SPLIT_NAME = "msqa_stage57_project_eval_v1"
_ADAPTER_CONTRACT_VERSION = "msqa_eval_adapter_v1"
_RECOMMENDED_SOURCE_IDENTITY = "msqa_row_source_url"
_RECOMMENDED_CANDIDATE_CONSTRUCTION = "processed_answer_sentence_candidates"
_DEFAULT_TOP_K = 10
_DEFAULT_MIN_SENTENCE_CHARS = 1
_SUPPORTED_STAGE_NAMES = ("Stage 61", "Stage 63")
_STAGE61_PASS_STATUS = "msqa_stage51_candidate_adapter_dry_run_passed"
_STAGE61_BLOCKED_STATUS = "msqa_stage51_candidate_adapter_dry_run_blocked"
_STAGE63_PASS_STATUS = "msqa_stage31_aligned_candidate_adapter_dry_run_passed"
_STAGE63_BLOCKED_STATUS = "msqa_stage31_aligned_candidate_adapter_dry_run_blocked"
_REQUIRED_CANDIDATE_FIELDS = (
    "question_id",
    "answer_id",
    "source_url",
    "candidate_id",
    "candidate_sentence",
    "retrieval_rank",
    "retrieval_score",
    "candidate_score",
    "source_row_id",
)


@dataclass(frozen=True)
class MsqaStage51AdapterSample:
    """One frozen MSQA row used by the Stage 61 adapter dry run."""

    question_id: str
    answer_id: str
    question: str
    answer: str
    source_url: str


@dataclass(frozen=True)
class MsqaStage51CandidateRow:
    """One Stage 61 MSQA row-source answer-sentence candidate."""

    query_question_id: str
    query_answer_id: str
    gold_source_row_id: str
    gold_source_url: str
    question_id: str
    answer_id: str
    source_url: str
    source_row_id: str
    candidate_id: str
    candidate_row_id: str
    candidate_sentence: str
    retrieval_rank: int
    retrieval_score: float
    candidate_score: float
    overlap_terms: tuple[str, ...]


@dataclass(frozen=True)
class MsqaStage51CandidateAdapterDryRun:
    """Stage 61 adapter report plus full candidate rows."""

    report: dict[str, Any]
    candidate_rows: list[MsqaStage51CandidateRow]


@dataclass(frozen=True)
class MsqaStage51CandidateAdapterVisualization:
    """One generated Stage 61 adapter visualization."""

    name: str
    path: str


def build_msqa_stage51_candidate_adapter_dry_run(
    *,
    split_jsonl_path: Path,
    protocol_report_path: Path,
    confirmed_protocol: bool,
    top_k: int = _DEFAULT_TOP_K,
    min_sentence_chars: int = _DEFAULT_MIN_SENTENCE_CHARS,
    max_candidates_per_source_row: int | None = None,
    sample_limit: int = 20,
    stage_name: str = _STAGE,
) -> MsqaStage51CandidateAdapterDryRun:
    """Build the confirmed MSQA row-source answer-sentence adapter dry run."""

    _ensure_file(split_jsonl_path)
    _ensure_file(protocol_report_path)
    _validate_options(
        confirmed_protocol=confirmed_protocol,
        top_k=top_k,
        min_sentence_chars=min_sentence_chars,
        max_candidates_per_source_row=max_candidates_per_source_row,
        sample_limit=sample_limit,
        stage_name=stage_name,
    )
    protocol_report = json.loads(protocol_report_path.read_text(encoding="utf-8"))
    _validate_protocol_report(protocol_report)

    samples = load_msqa_stage51_adapter_samples(split_jsonl_path)
    rows_by_id = {sample.question_id: sample for sample in samples}
    retriever = BM25Retriever()
    retriever.fit(_documents_from_samples(samples))

    candidate_rows: list[MsqaStage51CandidateRow] = []
    retrieval_ranks: list[int] = []
    samples_with_candidates = 0
    samples_with_gold_source_candidate = 0
    sample_summaries = []
    max_candidate_rows_for_samples = max(0, sample_limit)

    for sample in samples:
        retrieval_results = retriever.search(sample.question, top_k=top_k)
        result_ids = [result.document.id for result in retrieval_results]
        gold_rank = (
            result_ids.index(sample.question_id) + 1
            if sample.question_id in result_ids
            else None
        )
        if gold_rank is not None:
            retrieval_ranks.append(gold_rank)

        sample_rows: list[MsqaStage51CandidateRow] = []
        for result in retrieval_results:
            source_row = rows_by_id[result.document.id]
            sample_rows.extend(
                _candidate_rows_for_source_row(
                    query_sample=sample,
                    source_sample=source_row,
                    retrieval_rank=result.rank,
                    retrieval_score=result.score,
                    min_sentence_chars=min_sentence_chars,
                    max_candidates_per_source_row=max_candidates_per_source_row,
                )
            )
        if sample_rows:
            samples_with_candidates += 1
        if any(row.source_row_id == sample.question_id for row in sample_rows):
            samples_with_gold_source_candidate += 1
        candidate_rows.extend(sample_rows)
        if len(sample_summaries) < max_candidate_rows_for_samples:
            sample_summaries.append(
                _sample_summary(
                    sample=sample,
                    gold_rank=gold_rank,
                    candidate_rows=sample_rows,
                    top_k=top_k,
                )
            )

    report = _report(
        split_jsonl_path=split_jsonl_path,
        protocol_report_path=protocol_report_path,
        protocol_report=protocol_report,
        samples=samples,
        candidate_rows=candidate_rows,
        retrieval_ranks=retrieval_ranks,
        samples_with_candidates=samples_with_candidates,
        samples_with_gold_source_candidate=samples_with_gold_source_candidate,
        sample_summaries=sample_summaries,
        top_k=top_k,
        min_sentence_chars=min_sentence_chars,
        max_candidates_per_source_row=max_candidates_per_source_row,
        stage_name=stage_name,
    )
    return MsqaStage51CandidateAdapterDryRun(
        report=report,
        candidate_rows=candidate_rows,
    )


def load_msqa_stage51_adapter_samples(
    split_jsonl_path: Path,
) -> list[MsqaStage51AdapterSample]:
    """Load Stage 57 MSQA JSONL samples with the fields needed by Stage 61."""

    _ensure_file(split_jsonl_path)
    samples = []
    for line_number, line in enumerate(
        split_jsonl_path.read_text(encoding="utf-8").split("\n"),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != _SPLIT_NAME:
            raise ValueError(
                f"Unexpected split at line {line_number}: {row.get('split')!r}"
            )
        if row.get("adapter_contract_version") != _ADAPTER_CONTRACT_VERSION:
            raise ValueError(
                "Unexpected adapter contract at line "
                f"{line_number}: {row.get('adapter_contract_version')!r}"
            )
        samples.append(
            MsqaStage51AdapterSample(
                question_id=str(row["question_id"]),
                answer_id=str(row["answer_id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                source_url=str(row["source_url"]),
            )
        )
    if not samples:
        raise ValueError(f"No MSQA samples loaded from {split_jsonl_path}")
    return samples


def write_msqa_stage51_candidate_jsonl(
    *,
    candidate_rows: Sequence[MsqaStage51CandidateRow],
    output_path: Path,
) -> None:
    """Write Stage 61 candidate rows as local ignored JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(candidate_row_to_dict(row), ensure_ascii=False))
            handle.write("\n")


def candidate_row_to_dict(row: MsqaStage51CandidateRow) -> dict[str, Any]:
    """Convert one Stage 61 candidate row to a JSON-safe dictionary."""

    return asdict(row)


def write_msqa_stage51_candidate_adapter_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[MsqaStage51CandidateAdapterVisualization]:
    """Write SVG charts for Stage 61 adapter dry-run results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_label = str(report.get("stage", _STAGE))
    stage_slug = _stage_slug(stage_label)
    charts = {
        f"{stage_slug}_adapter_candidate_counts.svg": render_horizontal_bar_chart_svg(
            title=f"{stage_label} MSQA candidate adapter counts",
            bars=_candidate_count_bars(report),
            x_label="count",
            margin_left=340,
        ),
        f"{stage_slug}_adapter_source_hit_rates.svg": render_horizontal_bar_chart_svg(
            title=f"{stage_label} MSQA source retrieval hit rates",
            bars=_source_hit_rate_bars(report),
            x_label="rate",
            margin_left=260,
        ),
        f"{stage_slug}_adapter_contract_checks.svg": render_horizontal_bar_chart_svg(
            title=f"{stage_label} MSQA adapter contract checks",
            bars=_contract_check_bars(report),
            x_label="1 means pass",
            margin_left=360,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            MsqaStage51CandidateAdapterVisualization(name=filename, path=str(path))
        )
    return artifacts


def _documents_from_samples(
    samples: Sequence[MsqaStage51AdapterSample],
) -> list[PrimeQADocument]:
    return [
        PrimeQADocument(id=sample.question_id, title="", text=sample.answer)
        for sample in samples
    ]


def _candidate_rows_for_source_row(
    *,
    query_sample: MsqaStage51AdapterSample,
    source_sample: MsqaStage51AdapterSample,
    retrieval_rank: int,
    retrieval_score: float,
    min_sentence_chars: int,
    max_candidates_per_source_row: int | None,
) -> list[MsqaStage51CandidateRow]:
    rows = []
    seen_sentences = set()
    for sentence_index, sentence in enumerate(
        _answer_sentences(source_sample.answer, min_sentence_chars=min_sentence_chars),
        start=1,
    ):
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        candidate_id = (
            f"{source_sample.question_id}::processed_answer_sentence::"
            f"{sentence_index:03d}"
        )
        overlap_terms = _overlap_terms(query_sample.question, sentence)
        rows.append(
            MsqaStage51CandidateRow(
                query_question_id=query_sample.question_id,
                query_answer_id=query_sample.answer_id,
                gold_source_row_id=query_sample.question_id,
                gold_source_url=query_sample.source_url,
                question_id=source_sample.question_id,
                answer_id=source_sample.answer_id,
                source_url=source_sample.source_url,
                source_row_id=source_sample.question_id,
                candidate_id=candidate_id,
                candidate_row_id=(
                    f"{query_sample.question_id}::{candidate_id}"
                ),
                candidate_sentence=sentence,
                retrieval_rank=retrieval_rank,
                retrieval_score=round(retrieval_score, 6),
                candidate_score=_candidate_score(
                    retrieval_rank=retrieval_rank,
                    retrieval_score=retrieval_score,
                    overlap_terms=overlap_terms,
                ),
                overlap_terms=overlap_terms,
            )
        )
    if max_candidates_per_source_row is None:
        return rows
    return sorted(rows, key=_candidate_row_priority)[:max_candidates_per_source_row]


def _answer_sentences(text: str, min_sentence_chars: int) -> list[str]:
    sentences = [
        normalize_sentence(sentence)
        for sentence in split_sentences(text)
        if len(normalize_sentence(sentence)) >= min_sentence_chars
    ]
    return [sentence for sentence in sentences if sentence]


def _overlap_terms(question: str, sentence: str) -> tuple[str, ...]:
    question_terms = set(tokenize_text(question))
    sentence_terms = set(tokenize_text(sentence))
    return tuple(sorted(question_terms & sentence_terms))


def _candidate_score(
    *,
    retrieval_rank: int,
    retrieval_score: float,
    overlap_terms: Sequence[str],
) -> float:
    retrieval_prior = 1 / math.log2(retrieval_rank + 1)
    score = retrieval_score * retrieval_prior + 0.25 * len(overlap_terms)
    return round(score, 6)


def _report(
    *,
    split_jsonl_path: Path,
    protocol_report_path: Path,
    protocol_report: Mapping[str, Any],
    samples: Sequence[MsqaStage51AdapterSample],
    candidate_rows: Sequence[MsqaStage51CandidateRow],
    retrieval_ranks: Sequence[int],
    samples_with_candidates: int,
    samples_with_gold_source_candidate: int,
    sample_summaries: Sequence[Mapping[str, Any]],
    top_k: int,
    min_sentence_chars: int,
    max_candidates_per_source_row: int | None,
    stage_name: str,
) -> dict[str, Any]:
    total_samples = len(samples)
    contract_checks = _contract_checks(
        candidate_rows=candidate_rows,
        total_samples=total_samples,
        samples_with_candidates=samples_with_candidates,
    )
    all_contract_checks_pass = all(check["passed"] for check in contract_checks)
    effective_candidate_pool_cap = (
        top_k * max_candidates_per_source_row
        if max_candidates_per_source_row is not None
        else None
    )
    return {
        "stage": stage_name,
        "created_at": _CREATED_AT,
        "analysis_scope": _analysis_scope(
            stage_name=stage_name,
            max_candidates_per_source_row=max_candidates_per_source_row,
        ),
        "source_files": {
            "split_jsonl": _fingerprint(split_jsonl_path),
            "protocol_report": _fingerprint(protocol_report_path),
        },
        "user_confirmation": {
            "confirmed_protocol_option": "A",
            "confirmed_source_citation_identity": _RECOMMENDED_SOURCE_IDENTITY,
            "confirmed_candidate_construction": (
                _RECOMMENDED_CANDIDATE_CONSTRUCTION
            ),
            "confirmation_source": "current Codex conversation on 2026-07-14",
        },
        "adapter_contract": {
            "protocol_source": protocol_report["recommended_protocol"],
            "split_name": _SPLIT_NAME,
            "adapter_contract_version": _ADAPTER_CONTRACT_VERSION,
            "source_citation_identity": _RECOMMENDED_SOURCE_IDENTITY,
            "candidate_construction": _RECOMMENDED_CANDIDATE_CONSTRUCTION,
            "retrieval_index_text": "ProcessedAnswerText only",
            "excluded_index_text": "QuestionText",
            "no_answer_field_fallback": True,
            "external_fetch_used": False,
            "top_k": top_k,
            "min_sentence_chars": min_sentence_chars,
            "max_candidates_per_source_row": max_candidates_per_source_row,
            "effective_candidate_pool_cap": effective_candidate_pool_cap,
            "candidate_pool_cap_rule": _candidate_pool_cap_rule(
                max_candidates_per_source_row
            ),
            "required_candidate_fields": list(_REQUIRED_CANDIDATE_FIELDS),
            "candidate_score_boundary": (
                "Dry-run adapter score only. It combines answer-only BM25 source "
                "retrieval score, retrieval-rank prior, and query-sentence token "
                "overlap. It is not a tuned Stage 51 model score."
            ),
        },
        "dry_run_summary": {
            "evaluation_samples": total_samples,
            "candidate_rows": len(candidate_rows),
            "samples_with_candidates": samples_with_candidates,
            "samples_without_candidates": total_samples - samples_with_candidates,
            "samples_with_gold_source_candidate": samples_with_gold_source_candidate,
            "average_candidates_per_sample": round(
                len(candidate_rows) / total_samples,
                4,
            ),
            "median_candidates_per_sample": _median_candidates_per_sample(
                candidate_rows,
                samples,
            ),
            "max_candidates_per_sample_contract": effective_candidate_pool_cap,
            "unique_source_rows_in_candidates": len(
                {row.source_row_id for row in candidate_rows}
            ),
        },
        "source_retrieval_summary": {
            "hit@1": _hit_at_k(retrieval_ranks, total_samples, 1),
            f"hit@{top_k}": _hit_at_k(retrieval_ranks, total_samples, top_k),
            "mrr": _mrr(retrieval_ranks, total_samples),
            f"gold_source_missing_at_{top_k}": total_samples - len(retrieval_ranks),
        },
        "candidate_contract_checks": contract_checks,
        "sample_candidate_summaries": list(sample_summaries),
        "decision": {
            "status": _dry_run_status(
                stage_name=stage_name,
                all_contract_checks_pass=all_contract_checks_pass,
            ),
            "can_run_stage51_candidate_now": False,
            "can_defaultize_runtime_now": False,
            "default_runtime_policy": "unchanged",
            "stage51_candidate_run_performed": False,
            "recommended_next_stage": _recommended_next_stage(
                stage_name=stage_name,
                max_candidates_per_source_row=max_candidates_per_source_row,
            ),
        },
    }


def _contract_checks(
    *,
    candidate_rows: Sequence[MsqaStage51CandidateRow],
    total_samples: int,
    samples_with_candidates: int,
) -> list[dict[str, Any]]:
    rows_as_dicts = [candidate_row_to_dict(row) for row in candidate_rows]
    missing_required_field_rows = sum(
        1
        for row in rows_as_dicts
        if any(row.get(field) in (None, "") for field in _REQUIRED_CANDIDATE_FIELDS)
    )
    question_text_in_candidate_rows = any("question" in row for row in rows_as_dicts)
    return [
        _check(
            "user_confirmed_stage60_protocol",
            True,
            "User selected Stage 60 option A before Stage 61 started.",
        ),
        _check(
            "protocol_matches_stage60_recommendation",
            True,
            "Adapter uses msqa_row_source_url and processed_answer_sentence_candidates.",
        ),
        _check(
            "no_question_text_indexed_or_written_to_candidates",
            not question_text_in_candidate_rows,
            "Candidate rows contain IDs and answer sentences, not question text.",
        ),
        _check(
            "no_answer_field_fallback_used",
            True,
            "Adapter reads only Stage 57 JSONL answer field from ProcessedAnswerText.",
        ),
        _check(
            "no_external_fetch_used",
            True,
            "Adapter uses local frozen split rows only.",
        ),
        _check(
            "all_candidates_have_required_fields",
            missing_required_field_rows == 0,
            f"Rows missing required fields: {missing_required_field_rows}.",
        ),
        _check(
            "all_samples_have_candidate_rows",
            samples_with_candidates == total_samples,
            (
                f"Samples with candidates: {samples_with_candidates} / "
                f"{total_samples}."
            ),
        ),
    ]


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "evidence": evidence,
    }


def _analysis_scope(
    *,
    stage_name: str,
    max_candidates_per_source_row: int | None,
) -> str:
    base = (
        "MSQA row-source answer-sentence candidate adapter dry run. This report "
        "implements the user-confirmed Stage 60 protocol for contract checking "
        "only. It does not run Stage 51, does not tune policies, does not fetch "
        "external pages, and does not change the default runtime."
    )
    if stage_name == "Stage 63" and max_candidates_per_source_row is not None:
        return (
            f"{base} Stage 63 additionally applies the user-confirmed option A "
            "candidate-pool cap: each retrieved source row first generates all "
            "answer-sentence candidates, then keeps the top candidates by the "
            "existing dry-run candidate_score."
        )
    return base


def _candidate_pool_cap_rule(max_candidates_per_source_row: int | None) -> str:
    if max_candidates_per_source_row is None:
        return "uncapped"
    return (
        "For each retrieved source row, generate all normalized answer-sentence "
        "candidates, rank them by candidate_score descending, retrieval_rank "
        "ascending, and candidate_id ascending, then keep at most "
        f"{max_candidates_per_source_row} candidates."
    )


def _dry_run_status(*, stage_name: str, all_contract_checks_pass: bool) -> str:
    if stage_name == "Stage 63":
        return _STAGE63_PASS_STATUS if all_contract_checks_pass else _STAGE63_BLOCKED_STATUS
    return _STAGE61_PASS_STATUS if all_contract_checks_pass else _STAGE61_BLOCKED_STATUS


def _recommended_next_stage(
    *,
    stage_name: str,
    max_candidates_per_source_row: int | None,
) -> str:
    if stage_name == "Stage 63" and max_candidates_per_source_row is not None:
        return (
            "Stage 64: review the capped candidate distribution and, only if it "
            "is aligned, run one capped Stage 51 adapter comparison against the "
            "same capped candidate pool"
        )
    return (
        "Stage 62: review MSQA adapter candidate distribution and decide "
        "whether a single Stage 51 adapter comparison is fair"
    )


def _sample_summary(
    *,
    sample: MsqaStage51AdapterSample,
    gold_rank: int | None,
    candidate_rows: Sequence[MsqaStage51CandidateRow],
    top_k: int,
) -> dict[str, Any]:
    return {
        "query_question_id": sample.question_id,
        "gold_source_rank": gold_rank,
        "gold_source_retrieved_at_top_k": (
            gold_rank is not None and gold_rank <= top_k
        ),
        "candidate_count": len(candidate_rows),
        "gold_source_candidate_count": sum(
            1 for row in candidate_rows if row.source_row_id == sample.question_id
        ),
        "top_candidates": [
            {
                "candidate_row_id": row.candidate_row_id,
                "source_row_id": row.source_row_id,
                "retrieval_rank": row.retrieval_rank,
                "candidate_score": row.candidate_score,
                "sentence_preview": _preview(row.candidate_sentence),
            }
            for row in sorted(
                candidate_rows,
                key=lambda row: (
                    -row.candidate_score,
                    row.retrieval_rank,
                    row.candidate_id,
                ),
            )[:3]
        ],
    }


def _candidate_row_priority(row: MsqaStage51CandidateRow) -> tuple[float, int, str]:
    return (-row.candidate_score, row.retrieval_rank, row.candidate_id)


def _median_candidates_per_sample(
    candidate_rows: Sequence[MsqaStage51CandidateRow],
    samples: Sequence[MsqaStage51AdapterSample],
) -> float:
    counts_by_query_id = Counter(row.query_question_id for row in candidate_rows)
    counts = [counts_by_query_id[sample.question_id] for sample in samples]
    return float(statistics.median(counts)) if counts else 0.0


def _hit_at_k(ranks: Sequence[int], total: int, top_k: int) -> float:
    if total <= 0:
        return 0.0
    return round(sum(1 for rank in ranks if rank <= top_k) / total, 4)


def _mrr(ranks: Sequence[int], total: int) -> float:
    if total <= 0:
        return 0.0
    return round(sum(1 / rank for rank in ranks) / total, 4)


def _candidate_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["dry_run_summary"]
    return [
        BarDatum(
            label="evaluation_samples",
            value=float(summary["evaluation_samples"]),
            value_label=str(summary["evaluation_samples"]),
        ),
        BarDatum(
            label="candidate_rows",
            value=float(summary["candidate_rows"]),
            value_label=str(summary["candidate_rows"]),
        ),
        BarDatum(
            label="samples_with_candidates",
            value=float(summary["samples_with_candidates"]),
            value_label=str(summary["samples_with_candidates"]),
        ),
        BarDatum(
            label="samples_with_gold_source_candidate",
            value=float(summary["samples_with_gold_source_candidate"]),
            value_label=str(summary["samples_with_gold_source_candidate"]),
        ),
    ]


def _source_hit_rate_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    retrieval = report["source_retrieval_summary"]
    return [
        BarDatum(label=key, value=float(value), value_label=str(value))
        for key, value in retrieval.items()
        if key.startswith("hit@") or key == "mrr"
    ]


def _contract_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="pass" if check["passed"] else "fail",
        )
        for check in report["candidate_contract_checks"]
    ]


def _validate_options(
    *,
    confirmed_protocol: bool,
    top_k: int,
    min_sentence_chars: int,
    max_candidates_per_source_row: int | None,
    sample_limit: int,
    stage_name: str,
) -> None:
    if not confirmed_protocol:
        raise ValueError("Stage 61 requires confirmed_protocol=True")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if min_sentence_chars <= 0:
        raise ValueError("min_sentence_chars must be positive")
    if (
        max_candidates_per_source_row is not None
        and max_candidates_per_source_row <= 0
    ):
        raise ValueError("max_candidates_per_source_row must be positive")
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    if stage_name not in _SUPPORTED_STAGE_NAMES:
        raise ValueError(
            "stage_name must be one of: "
            + ", ".join(repr(stage) for stage in _SUPPORTED_STAGE_NAMES)
        )


def _validate_protocol_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 60":
        raise ValueError(f"Expected Stage 60 protocol report, got: {report.get('stage')!r}")
    decision = report["decision"]
    if (
        decision.get("recommended_source_citation_identity")
        != _RECOMMENDED_SOURCE_IDENTITY
    ):
        raise ValueError("Stage 60 source identity does not match option A")
    if (
        decision.get("recommended_candidate_construction")
        != _RECOMMENDED_CANDIDATE_CONSTRUCTION
    ):
        raise ValueError("Stage 60 candidate construction does not match option A")
    if decision.get("requires_user_confirmation") is not True:
        raise ValueError("Stage 60 protocol must require user confirmation")


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


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _stage_slug(stage_label: str) -> str:
    return "".join(stage_label.lower().split())
