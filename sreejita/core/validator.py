from dataclasses import dataclass
import pandas as pd

@dataclass
class DataQualityValidator:
    strict: bool = False

    def validate(self, df: pd.DataFrame):
        results = {}

        results["rows"] = len(df)
        results["columns"] = len(df.columns)
        results["missing_values"] = df.isnull().sum().to_dict()
        results["duplicate_rows"] = df.duplicated().sum()

        passed = True
        if self.strict:
            passed = results["duplicate_rows"] == 0

        return passed, results
