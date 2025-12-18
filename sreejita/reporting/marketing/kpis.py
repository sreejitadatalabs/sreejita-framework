def compute_marketing_kpis(df):
    return {
        "conversion_rate": (df["converted"] == True).mean(),
        "campaign_cost": df["cost"].sum(),
    }
