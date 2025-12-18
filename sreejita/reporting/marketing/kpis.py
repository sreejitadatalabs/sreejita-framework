from sreejita.reporting.utils import safe_mean, safe_sum, safe_ratio


def compute_marketing_kpis(df):
    spend = safe_sum(df, "spend")
    revenue = safe_sum(df, "revenue")

    return {
        "total_spend": spend,
        "total_revenue": revenue,
        "roas": safe_ratio(revenue, spend),
        "avg_ctr": safe_mean(df, "ctr"),
        "avg_conversion_rate": safe_mean(df, "conversion_rate"),
        "avg_cac": safe_mean(df, "cac"),
    }
