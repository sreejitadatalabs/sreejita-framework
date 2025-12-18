from sreejita.reporting.utils import safe_mean, safe_sum, safe_ratio
from sreejita.domains.column_mapping import ColumnMapping

def compute_ops_kpis(df):
    """
    Compute operations KPIs with flexible column matching.
    Supports multiple naming conventions for operational metrics.
    """
    # Auto-detect and map columns
    mapping = ColumnMapping.auto_detect(df)
    
    # Define column alternatives for flexible matching
    cycle_time_cols = ['cycle_time', 'processing_time', 'duration', 'elapsed_time', 'lead_time']
    on_time_cols = ['on_time', 'on_schedule', 'on_time_delivery', 'delivered_on_time', 'on_time_flag']
    delay_cols = ['delay', 'delay_days', 'late', 'delay_hours', 'lateness', 'days_late']
    
    # Find the actual columns in the dataframe
    cycle_time_col = next((col for col in cycle_time_cols if col in df.columns), None)
    on_time_col = next((col for col in on_time_cols if col in df.columns), None)
    delay_col = next((col for col in delay_cols if col in df.columns), None)
    
    kpis = {}
    
    # Cycle time metrics
    if cycle_time_col:
        kpis["avg_cycle_time"] = safe_mean(df, cycle_time_col)
        kpis["total_cycle_time"] = safe_sum(df, cycle_time_col)
    
    # On-time metrics
    if on_time_col:
        kpis["on_time_rate"] = safe_ratio(df[on_time_col].sum() if on_time_col in df.columns else 0, len(df))
    
    # Delay metrics
    if delay_col:
        kpis["avg_delay"] = safe_mean(df, delay_col)
        kpis["total_delay"] = safe_sum(df, delay_col)
    
    # Fallback: return empty dict if no relevant columns found
    if not kpis:
        kpis = {
            "avg_cycle_time": None,
            "on_time_rate": None,
            "avg_delay": None
        }
    
    return kpis
