import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """
    Deterministic sample dataset for domain detection tests.
    Matches Retail + Customer signals.
    """
    return pd.DataFrame({
        "order_id": [1, 2, 3],
        "product": ["A", "B", "C"],
        "quantity": [2, 1, 5],
        "sales": [200, 100, 500],
        "customer_id": [10, 11, 12],
        "location": ["IN", "US", "UK"]
    })
