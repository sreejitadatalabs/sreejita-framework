import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Assuming these are from your local environment (keep them as is)
from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# =====================================================
# GLOBAL CONFIG: UNIVERSAL DICTIONARIES
# =====================================================

NEGATIVE_KEYWORDS = {
    "abnormal", "failed", "deceased", "critical",
    "positive", "poor", "expired", "high", "severe"
}

OUTCOME_COLUMN_HINTS = {
    "result", "status", "outcome", "test", "finding", "evaluation"
}

# =====================================================
# HELPER: DATA NORMALIZATION
# =====================================================

def _normalize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No, Y/N, True/False columns to 1/0.
    """
    for col in df.columns:
        if df[col].dtype == object:
            values = set(
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
            )
            if values and values.issubset({"yes", "no", "y", "n", "true", "false"}):
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({
                        "yes": 1, "y": 1, "true": 1,
                        "no": 0, "n": 0, "false": 0,
                    })
                )
    return df

# =====================================================
# HELPER: DATE INTELLIGENCE (SMART DATE FIX)
# =====================================================

def _derive_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive length_of_stay if missing but admission & discharge dates exist.
    Uses 'fuzzy matching' to find date columns even if names change.
    """
    # 1. If it already exists, do nothing
    if resolve_column(df, "length_of_stay"):
        return df

    # 2. Smart Search: Look for ANY column containing specific keywords
    admit = None
    discharge = None
    
    for col in df.columns:
        c_low = col.lower()
        if "admission" in c_low and "date" in c_low:
            admit = col
        if "discharge" in c_low and "date" in c_low:
            discharge = col
            
    # 3. Calculate if both found
    if admit and discharge:
        try:
            # Convert to datetime safely
            df[admit] = pd.to_datetime(df[admit], errors="coerce")
            df[discharge] = pd.to_datetime(df[discharge], errors="coerce")
            
            # Calculate difference
            df["derived_length_of_stay"] = (
                df[discharge] - df[admit]
            ).dt.days
            
            # Clean up: Remove negative days (data errors)
            df.loc[df["derived_length_of_stay"] < 0, "derived_length_of_stay"] = None
            
        except Exception:
            pass # Fail silently if date formats are unrecognizable

    return df

# =====================================================
# HELPER: DYNAMIC RISK SCANNER
# =====================================================

def _scan_categorical_risks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Scans ALL columns for negative keywords (Universal Risk Detection).
    """
    insights: List[Dict[str, Any]] = []

    for col in df.columns:
        col_l = col.lower()
        
        # Only check columns that sound like outcomes (to save time)
        if not any(h in col_l for h in OUTCOME_COLUMN_HINTS):
            continue
        if df[col].dtype != object:
            continue

        # Check content
        values = df[col].dropna().astype(str).str.lower()
        if values.empty:
            continue

        negatives = values[values.isin(NEGATIVE_KEYWORDS)]
        rate = len(negatives) / len(values)

        # Threshold: If >15% are bad, flag it
        if rate >= 0.15:
            insights.append({
                "level": "RISK" if rate >= 0.25 else "WARNING",
                "title": f"High Rate of Negative Outcomes in '{col}'",
                "so_what": (
                    f"{rate:.0%} of records contain negative values "
                    f"(e.g., {', '.join(negatives.unique()[:3])})."
                ),
                # Metadata to help the main engine know which column to blame
                "detected_col": col 
            })

    return insights

# =====================================================
# MAIN CLASS: HEALTHCARE DOMAIN
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"
    description = "Universal healthcare analytics (clinical + financial + operational)"

    def validate_data(self, df: pd.DataFrame) -> bool:
        # Relaxed validation: Just needs SOME identifiable healthcare column
        return (resolve_column(df, "patient_id") is not None) or \
               (resolve_column(df, "billing_amount") is not None)

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _normalize_binary_columns(df)
        df = _derive_length_of_stay(df)
        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        pid = resolve_column(df, "patient_id")

        # Dynamic LoS (Real or Derived)
        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]):
            kpis["avg_length_of_stay"] = df[los].mean()

        # Dynamic Readmission (might not exist in Dataset B)
        readm = resolve_column(df, "readmitted")
        if readm and pd.api.types.is_numeric_dtype(df[readm]):
            kpis["readmission_rate"] = df[readm].mean()

        # Financials
        bill = resolve_column(df, "billing_amount")
        if bill and pd.api.types.is_numeric_dtype(df[bill]):
            kpis["total_billing"] = df[bill].sum()
            if pid:
                kpis["avg_billing_per_patient"] = (
                    df.groupby(pid)[bill].sum().mean()
                )

        if pid:
            kpis["patient_volume"] = df[pid].nunique()

        return kpis

    # ---------------- VISUALS ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        def human_axis(x, _):
            if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if x >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Length of Stay (Real or Derived)
        los = resolve_column(df, "length_of_stay") or resolve_column(df, "derived_length_of_stay")
        if los and pd.api.types.is_numeric_dtype(df[los]):
            path = output_dir / "length_of_stay.png"
            df[los].dropna().hist(bins=15)
            plt.title("Length of Stay Distribution")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({"path": path, "caption": "Length of stay distribution"})

        # 2. Billing by Insurance
        bill = resolve_column(df, "billing_amount")
        ins = resolve_column(df, "insurance")
        if bill and ins and pd.api.types.is_numeric_dtype(df[bill]):
            path = output_dir / "billing_by_insurance.png"
            ax = df.groupby(ins)[bill].sum().sort_values(ascending=False).plot(kind="bar")
            ax.yaxis.set_major_formatter(FuncFormatter(human_axis))
            plt.title("Billing by Insurance Provider")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            visuals.append({"path": path, "caption": "Billing by insurance provider"})

        return visuals

    # ---------------- INSIGHTS (UNIVERSAL LOGIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights: List[Dict[str, Any]] = []

        # 1. RUN THE UNIVERSAL SCANNER FIRST
        # This detects "Abnormal" tests even if no Readmission column exists
        risk_findings = _scan_categorical_risks(df)
        insights.extend(risk_findings)

        # 2. DETERMINE PRIMARY RISK COLUMN (The "Source of Truth" for this dataset)
        primary_risk_col = resolve_column(df, "readmitted")
        
        # If 'readmitted' is missing (Dataset B), use the column found by the scanner
        if not primary_risk_col and risk_findings:
            # Look for the column name in the scanner results
            # We grab the first "RISK" level finding
            for finding in risk_findings:
                if "detected_col" in finding:
                    col_name = finding["detected_col"]
                    # We need to turn this text column (Normal/Abnormal) into a number (0/1) for analysis
                    # Create a temporary flag for the "Worst Doctor" calculation
                    df["_temp_risk_flag"] = df[col_name].astype(str).str.lower().isin(NEGATIVE_KEYWORDS).astype(int)
                    primary_risk_col = "_temp_risk_flag"
                    break

        # 3. IDENTIFY WORST PERFORMERS (Universal)
        # Now we calculate "Worst Doctor" regardless of whether the risk is "Readmission" or "Abnormal Tests"
        if primary_risk_col:
            overall_rate = df[primary_risk_col].mean()
            
            # Check Doctors and Hospitals
            for segment_key, segment_label in [("doctor", "Doctor"), ("hospital_branch", "Hospital")]:
                segment_col = resolve_column(df, segment_key)
                
                if segment_col and overall_rate > 0:
                    grp = df.groupby(segment_col)[primary_risk_col].mean()
                    worst_name = grp.idxmax()
                    worst_rate = grp.max()

                    # Only report if it's significantly worse (e.g., 20% worse than average)
                    if worst_rate > (overall_rate * 1.2):
                        
                        # Formatting the message based on source
                        source_msg = "Readmission Rate"
                        if primary_risk_col == "_temp_risk_flag":
                            source_msg = "Adverse Outcome Rate (e.g., Abnormal/Failed)"

                        insights.append({
                            "level": "RISK",
                            "title": f"Worst Performing {segment_label}: {worst_name}",
                            "so_what": (
                                f"{source_msg} is {worst_rate:.1%} "
                                f"(Average: {overall_rate:.1%}). Investigate clinical drivers."
                            )
                        })

        # 4. FINANCIAL CONCENTRATION (Fallback if no risks)
        if not any(i['level'] == 'RISK' for i in insights):
            bill = resolve_column(df, "billing_amount")
            if bill:
                key_col = resolve_column(df, "diagnosis") or resolve_column(df, "insurance")
                if key_col:
                    grp = df.groupby(key_col)[bill].sum()
                    top = grp.idxmax()
                    insights.append({
                        "level": "INFO",
                        "title": f"Highest Cost Driver: {top}",
                        "so_what": f"Accounts for {grp[top]/grp.sum():.0%} of total billing."
                    })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        insights = self.generate_insights(df, kpis)

        for i in insights:
            if i["level"] == "RISK":
                recs.append({
                    "action": f"Investigate immediately: {i['title']}",
                    "priority": "HIGH",
                    "timeline": "2–4 weeks",
                })
            elif i["level"] == "WARNING":
                recs.append({
                    "action": f"Review protocols for: {i['title']}",
                    "priority": "MEDIUM",
                    "timeline": "4–6 weeks",
                })

        if not recs:
            recs.append({
                "action": "Continue routine monitoring",
                "priority": "LOW",
                "timeline": "Ongoing",
            })

        return recs

# =====================================================
# REGISTRATION
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"
    # Extended dictionary to catch columns from BOTH datasets
    HEALTHCARE_COLUMNS: Set[str] = {
        "patient", "patient_id", "pid",
        "los", "length_of_stay", "admission_date", "date_of_admission",
        "readmitted", "mortality", "test", "result",
        "billing", "insurance",
        "doctor", "diagnosis", "medical_condition"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        # Check for partial matches to be safe (e.g. "date of admission" contains "admission")
        matched_count = 0
        matches = []
        for target in self.HEALTHCARE_COLUMNS:
            for col in cols:
                if target in col:
                    matched_count += 1
                    matches.append(col)
                    break
        
        confidence = min(matched_count / 4, 1.0)
        return DomainDetectionResult(
            domain="healthcare",
            confidence=confidence,
            signals={"matched_columns": matches},
        )

def register(registry):
    registry.register(
        name="healthcare",
        domain_cls=HealthcareDomain,
        detector_cls=HealthcareDomainDetector,
    )
