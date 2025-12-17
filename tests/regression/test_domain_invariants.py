import pytest
import pandas as pd

from sreejita.domains.router import decide_domain
from sreejita.policy.engine import PolicyEngine
from sreejita.reporting.orchestrator import generate_report_payload


# -------------------------------------------------
# Fixtures
# -------------------------------------------------

@pytest.fixture
def retail_df():
    """
    Minimal dataframe that should clearly be detected as Retail.
    """
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "sales": [100.0, 200.0, 300.0],
            "profit": [20.0, 40.0, 60.0],
            "discount": [0.1, 0.2, 0.15],
            "shipping_cost": [8.0, 18.0, 25.0],
        }
    )


@pytest.fixture
def unknown_df():
    """
    Dataframe with no clear domain signals.
    """
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )


# -------------------------------------------------
# Domain Regression Tests
# -------------------------------------------------

def test_retail_domain_detection_stable(retail_df):
    """
    Retail domain must always be detected for retail-like data.
    """
    decision = decide_domain(retail_df)
    assert decision.selected_domain == "retail"
    assert decision.confidence > 0


def test_domain_confidence_in_range(retail_df):
    """
    Domain confidence must always be between 0 and 1.
    """
    decision = decide_domain(retail_df)
    assert 0.0 <= decision.confidence <= 1.0


def test_policy_blocks_low_confidence_domains(unknown_df):
    """
    Low-confidence domain detection should be blocked by policy.
    """
    decision = decide_domain(unknown_df)
    policy = PolicyEngine(min_confidence=0.7).evaluate(decision)

    assert policy.status in {"blocked", "review", "allowed"}


def test_unknown_domain_handled_safely(unknown_df):
    """
    Unknown or weak domains must not crash the system.
    """
    decision = decide_domain(unknown_df)
    assert decision.selected_domain in {"unknown", None, ""}


def test_report_payload_generated_for_retail(retail_df):
    """
    Retail domain must always produce a valid report payload.
    """
    decision = decide_domain(retail_df)
    policy = PolicyEngine(min_confidence=0.0).evaluate(decision)

    payload = generate_report_payload(retail_df, decision, policy)

    assert isinstance(payload, dict)
    assert "kpis" in payload
    assert "insights" in payload
    assert "recommendations" in payload