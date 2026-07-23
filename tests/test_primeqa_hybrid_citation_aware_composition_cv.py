from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET

from scripts.analyze_primeqa_hybrid_citation_aware_composition_cv import main
from ts_rag_agent.application import primeqa_hybrid_citation_aware_composition_cv as stage180
from ts_rag_agent.application.citation_aware_composition_policy import stage180_policy_specs
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg


def test_selection_order_prioritizes_worst_fold_citation() -> None:
    specs = stage180_policy_specs()[:2]
    summaries = {
        specs[0].policy_id: _summary(min_citation=0, citation=8, min_f1=-0.1, f1=0.2),
        specs[1].policy_id: _summary(min_citation=1, citation=2, min_f1=-0.2, f1=-0.1),
    }

    selected = stage180._select_spec(specs=specs, summaries=summaries)

    assert selected.policy_id == specs[1].policy_id


def test_selection_order_uses_f1_after_citation_ties() -> None:
    specs = stage180_policy_specs()[:2]
    summaries = {
        specs[0].policy_id: _summary(min_citation=0, citation=2, min_f1=0.0, f1=0.01),
        specs[1].policy_id: _summary(min_citation=0, citation=2, min_f1=0.0, f1=0.02),
    }

    selected = stage180._select_spec(specs=specs, summaries=summaries)

    assert selected.policy_id == specs[1].policy_id


def test_stage180_cli_has_no_development_test_or_model_inputs() -> None:
    assert set(inspect.signature(main).parameters) == {
        "output",
        "visualization_dir",
        "encoder_batch_size",
    }
    assert stage180._EXPECTED_MODEL_HEAD_FITS == 50
    assert stage180._BOOTSTRAP_REPLICATES == 2_000


def test_stage180_forbidden_key_check_uses_extended_contract() -> None:
    found = stage180._stage180_forbidden_keys_found(
        {"safe": [{"runtime_features": {"candidate_score": 1.0}}]}
    )

    assert found == {"runtime_features"}


def test_stage180_chart_uses_shared_svg_api() -> None:
    svg = stage180._chart(
        "Stage 180 test chart",
        (BarDatum("candidate", 1.0, "1"),),
        "count",
    )

    assert ET.fromstring(svg).tag.endswith("svg")


def test_shared_svg_chart_moves_colliding_negative_value_inside_bar() -> None:
    svg = render_horizontal_bar_chart_svg(
        title="Signed values",
        bars=(
            BarDatum("minimum", -1.0, "-1.0000"),
            BarDatum("small negative", -0.1, "-0.1000"),
            BarDatum("positive", 0.5, "+0.5000"),
        ),
        x_label="delta",
        width=920,
        margin_left=230,
    )
    root = ET.fromstring(svg)
    texts = {element.text: element for element in root.findall("{http://www.w3.org/2000/svg}text")}

    assert texts["-1.0000"].attrib["class"] == "bar-value-inverted"
    assert "class" not in texts["-0.1000"].attrib
    assert "class" not in texts["+0.5000"].attrib


def _summary(
    *,
    min_citation: int,
    citation: int,
    min_f1: float,
    f1: float,
) -> dict[str, int | float]:
    return {
        "minimum_fold_gold_citation_delta": min_citation,
        "gold_citation_delta": citation,
        "minimum_fold_answerable_f1_delta": min_f1,
        "answerable_f1_delta": f1,
        "f1_regressed_count": 1,
        "changed_verified_count": 1,
    }
