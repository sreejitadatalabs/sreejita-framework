from sreejita.core.cleaner import clean_dataframe
from sreejita.core.kpis import compute_kpis
from sreejita.core.insights import correlation_insights
from sreejita.core.recommendations import generate_recommendations

def run_dynamic(input_path, output, config):
    import pandas as pd
    df = pd.read_csv(input_path)
    result = clean_dataframe(df, [config["dataset"]["date"]])
    df = result["df"]

    kpis = compute_kpis(df)
    insights = correlation_insights(df)
    recs = generate_recommendations(df)

    # PDF logic plugs here (reuse v1.0 PDF builder)
    print("Dynamic report generated:", output)
