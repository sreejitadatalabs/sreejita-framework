"""
Text Domain (Compatibility Domain)

Used for:
- free text inputs
- fallback analysis
- backward compatibility with v1.x / tests
"""

from sreejita.domains.base import BaseDomain


class TextDomain(BaseDomain):
    name = "text"
    description = "Generic text / fallback domain"

    def validate_data(self, df):
        return False  # text domain is never auto-selected

    def preprocess(self, df):
        return df

    def calculate_kpis(self, df):
        return {}

    def generate_insights(self, df, kpis):
        return []

    def generate_recommendations(self, df, kpis):
        return []

    def generate_visuals(self, df, output_dir):
        return []
