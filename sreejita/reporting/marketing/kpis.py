def compute_marketing_kpis(df):
    total_cost = df["cost"].sum()
    conversion_rate = (df["converted"] == True).mean()

    return {
        "total_spend": total_cost,
        "conversion_rate": conversion_rate,
        "campaign_count": df["campaign_id"].nunique(),
    }
