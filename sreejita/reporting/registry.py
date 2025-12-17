from pathlib import Path
from sreejita.core.schema import detect_schema

# -------------------------
# REPORT ENGINES
# -------------------------
from sreejita.reporting.retail.kpis import compute_retail_kpis
from sreejita.reporting.retail.insights import generate_retail_insights
from sreejita.reporting.retail.recommendations import generate_retail_recommendations

# -------------------------
# VISUAL ENGINES (POLISHED)
# -------------------------
from sreejita.visuals.time_series import plot_monthly
from sreejita.visuals.categorical import bar
from sreejita.visuals.correlation import heatmap, shipping_cost_vs_sales


# =====================================================
# VISUAL ADAPTERS (THIS IS THE KEY)
# =====================================================

def _retail_sales_trend(df, output_dir: Path):
    """
    Sales trend over time.
    """
    schema = detect_schema(df)

    if not schema["datetime"] or not schema["numeric_measures"]:
        return None

    date_col = schema["datetime"][0]
    value_col = schema["numeric_measures"][0]

    out = output_dir / "sales_trend.png"
    plot_monthly(df, date_col, value_col, out)
    return out if out.exists() else None


def _retail_sales_by_category(df, output_dir: Path):
    """
    Sales contribution by category.
    """
    schema = detect_schema(df)

    if not schema["categorical"]:
        return None

    category_col = schema["categorical"][0]
    out = output_dir / "sales_by_category.png"

    bar(df, category_col, out)
    return out if out.exists() else None


def _retail_shipping_cost_vs_sales(df, output_dir: Path):
    """
    Shipping cost vs order value relationship.
    """
    schema = detect_schema(df)

    if len(schema["numeric_measures"]) < 2:
        return None

    sales_col = schema["numeric_measures"][0]
    shipping_col = schema["numeric_measures"][1]

    out = output_dir / "shipping_cost_vs_sales.png"
    shipping_cost_vs_sales(df, sales_col, shipping_col, out)
    return out if out.exists() else None


def _retail_correlation_heatmap(df, output_dir: Path):
    """
    Correlation between key numeric metrics.
    """
    out = output_dir / "correlation_heatmap.png"
    return heatmap(df, out)


# =====================================================
# DOMAIN REGISTRIES
# =====================================================

DOMAIN_REPORT_ENGINES = {
    "retail": {
        "kpis": compute_retail_kpis,
        "insights": generate_retail_insights,
        "recommendations": generate_retail_recommendations,
    }
}

DOMAIN_VISUALS = {
    "retail": {
        "__always__": [
            _retail_sales_trend,
            _retail_sales_by_category,
            _retail_shipping_cost_vs_sales,
            _retail_correlation_heatmap,
        ]
    }
}
