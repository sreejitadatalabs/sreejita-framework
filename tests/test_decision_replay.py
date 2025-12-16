from sreejita.domains.router import decide_domain


def test_domain_decision_is_deterministic(sample_df):
    d1 = decide_domain(sample_df)
    d2 = decide_domain(sample_df)

    assert d1.selected_domain == d2.selected_domain
    assert d1.confidence == d2.confidence
    assert d1.fingerprint == d2.fingerprint
