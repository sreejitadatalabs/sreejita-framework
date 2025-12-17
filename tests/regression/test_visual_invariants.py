from pathlib import Path
import pandas as pd
import pytest

from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine
from sreejita.reporting.orchestrator import generate_report_payload


# -------------------------------------------------
# Fixtures
# -------------------------------------------------

@pytest.fixture
def retail_df():
    """
    Retail-like dataframe sufficient to trigger visuals.
    """
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "sales": [100, 200, 300, 400, 500],
            "profit": [20, 40, 60, 80, 100],
            "discount": [0.1, 0.15, 0.2, 0.1, 0.05],
            "shipping_cost": [8, 18, 25, 30, 42],
            "category": ["A", "B", "A", "C", "B"],
        }
    )


@pytest.fixture
def output_dir(tmp_path):
    """
    Temporary directory for visual outputs.
    """
    return tmp_path / "visuals"


# -------------------------------------------------
# Visual Regression Tests
# -------------------------------------------------

def test_visuals_are_generated(retail_df, output_dir):
    """
    Visuals must be generated for Retail domain.
    """
    decision = decide_domain(retail_df)
    policy = PolicyEngine(min_confidence=0.0).evaluate(decision)

    payload = generate_report_payload(
        retail_df,
        decision,
        policy,
        output_dir=output_dir,
    )

    visuals = payload.get("visuals", [])
    assert isinstance(visuals, list)
    assert len(visuals) > 0


def test_minimum_visual_count(retail_df, output_dir):
    """
    Retail reports must have at least 3 visuals.
    """
    decision = decide_domain(retail_df)
    policy = PolicyEngine(min_confidence=0.0).evaluate(decision)

    payload = generate_report_payload(
        retail_df,
        decision,
        policy,
        output_dir=output_dir,
    )

    visuals = payload.get("visuals", [])
    assert len(visuals) >= 3


def test_visual_files_exist_and_valid(retail_df, output_dir):
    """
    Generated visual files must exist and not be empty.
    """
    decision = decide_domain(retail_df)
    policy = PolicyEngine(min_confidence=0.0).evaluate(decision)

    payload = generate_report_payload(
        retail_df,
        decision,
        policy,
        output_dir=output_dir,
    )

    for visual in payload.get("visuals", []):
        path = Path(visual["path"])
        assert path.exists()
        assert path.stat().st_size > 5_000  # sanity threshold (~5KB)


def test_visuals_have_captions(retail_df, output_dir):
    """
    Every visual must include a human-readable caption.
    """
    decision = decide_domain(retail_df)
    policy = PolicyEngine(min_confidence=0.0).evaluate(decision)

    payload = generate_report_payload(
        retail_df,
        decision,
        policy,
        output_dir=output_dir,
    )

    for visual in payload.get("visuals", []):
        assert "caption" in visual
        assert isinstance(visual["caption"], str)
        assert len(visual["caption"].strip()) > 0