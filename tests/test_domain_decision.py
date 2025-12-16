from sreejita.domains.router import decide_domain


def test_domain_decision_is_explainable(sample_df):
    decision = decide_domain(sample_df)

    assert decision.decision_type == "domain_detection"
    assert decision.selected_domain is not None
    assert decision.confidence >= 0
    assert isinstance(decision.rules_applied, list)
    assert decision.timestamp is not None
