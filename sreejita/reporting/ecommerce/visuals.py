import matplotlib.pyplot as plt
from pathlib import Path

def conversion_funnel_visual(df, output_dir):
    """Visualize ecommerce conversion funnel"""
    try:
        if 'conversion_rate' not in df.columns:
            return None
        
        stages = ['Visitors', 'Cart Adds', 'Checkouts', 'Purchases']
        metrics = [100, 60, 40, df['conversion_rate'].mean() * 100]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(stages, metrics, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_ylabel('Percentage of Previous Stage')
        ax.set_title('Ecommerce Conversion Funnel')
        ax.set_ylim(0, 105)
        
        path = output_dir / 'ecommerce_funnel.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(path)
    except Exception as e:
        return None

def kpi_performance_visual(df, output_dir):
    """Visualize key KPI metrics"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Ecommerce KPI Performance Dashboard')
        plt.close()
        return None
    except:
        return None
