from __future__ import annotations

import inspect

from scripts.analyze_primeqa_hybrid_listwise_agent_failures import main
from ts_rag_agent.application import primeqa_hybrid_listwise_agent_failure_attribution as analysis


def test_failure_attribution_selects_context_to_citation_bottleneck() -> None:
    rows = []
    for index in range(370):
        context_gain = index < 10
        rows.append(
            analysis.Stage179DiagnosticRow(
                fold_id=f"fold_{index % 5 + 1}",
                prefix_gold_hit=True,
                union_gold_hit=True,
                baseline_context_hit=not context_gain,
                candidate_context_hit=True,
                baseline_gold_cited=not context_gain,
                candidate_gold_cited=not context_gain or index < 4,
                baseline_f1=0.5,
                candidate_f1=0.6 if index % 2 else 0.4,
                candidate_gold_context_rank=index % 10 + 1,
            )
        )

    result = analysis.analyze_diagnostic_rows(rows)

    assert result["context_gain_attribution"]["count"] == 10
    assert result["context_gain_attribution"]["candidate_gold_cited_count"] == 4
    assert result["context_gain_attribution"]["candidate_gold_uncited_count"] == 6
    assert result["diagnostic_decision"]["primary_bottleneck"] == "context_to_citation_conversion"
    assert len(result["fold_reports"]) == 5


def test_stage179_cli_has_no_development_test_or_model_inputs() -> None:
    assert set(inspect.signature(main).parameters) == {
        "output",
        "visualization_dir",
        "encoder_batch_size",
    }
    assert analysis._EXPECTED_FOLDS == 5
    assert analysis._EXPECTED_PAIR_SCORES == 9_714
