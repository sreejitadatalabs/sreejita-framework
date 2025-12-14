import pandas as pd
import numpy as np

class DataProfiler:
    def profile(self, df: pd.DataFrame):
        profile = {}

        profile["shape"] = df.shape
        profile["dtypes"] = df.dtypes.astype(str).to_dict()
        profile["missing"] = df.isnull().sum().to_dict()

        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty:
            profile["numeric_summary"] = numeric.describe().to_dict()

        return profile
