from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET

from scripts.analyze_primeqa_hybrid_composition_action_audit import main
from ts_rag_agent.application import primeqa_hybrid_composition_action_audit as stage181


def test_stage181_cli_has_no_development_test_or_runtime_inputs() -> None:
    assert set(inspect.signature(main).parameters) == {
        "output",
        "visualization_dir",
        "encoder_batch_size",
    }
    assert stage181._EXPECTED_ROWS == 562
    assert stage181._EXPECTED_ANSWERABLE == 370
    assert stage181._EXPECTED_FOLDS == 5


def test_stage181_public_contract_rejects_private_action_fields() -> None:
    found = stage181._forbidden_keys_found(
        {"safe": [{"question_key": "private"}, {"selected_indices": [0, 1]}]}
    )

    assert found == {"question_key", "selected_indices"}


def test_stage181_family_scatter_is_parseable_svg() -> None:
    svg = stage181._family_scatter_svg(
        {
            "replace_slot_1": {
                "mean_citation_delta": 0.1,
                "mean_f1_delta": -0.01,
                "action_count": 10,
            },
            "delete_slot_3": {
                "mean_citation_delta": 0.0,
                "mean_f1_delta": 0.02,
                "action_count": 5,
            },
        }
    )

    assert ET.fromstring(svg).tag.endswith("svg")
    assert "replace_slot_1" in svg
    assert "action family" in svg
    assert "citation +0.1000, F1 -0.0100" in svg


def test_stage181_historical_baseline_f1_tolerance_is_bounded() -> None:
    assert stage181._baseline_f1_within_tolerance(0.199876, 0.2004) is True
    assert stage181._baseline_f1_within_tolerance(0.1994, 0.2004) is True
    assert stage181._baseline_f1_within_tolerance(0.199399, 0.2004) is False
