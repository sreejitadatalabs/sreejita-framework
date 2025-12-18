from sreejita.reporting.utils import safe_mean, safe_sum


def compute_ops_kpis(df):
    return {
        "avg_processing_time": safe_mean(df, "processing_time"),
        "avg_packing_time": safe_mean(df, "packing_time"),
        "total_shipping_cost": safe_sum(df, "shipping_cost"),
        "total_labor_cost": safe_sum(df, "labor_cost"),
        "return_rate": safe_mean(df, "returned"),
        "on_time_delivery_rate": safe_mean(df, "delivery_status"),
    }
