import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sreejita.core.column_resolver import resolve_column
from .base import BaseDomain
from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult


# =====================================================
# HELPERS
# =====================================================

def _safe_div(n, d):
    """Safely divides n by d, returning None if d is 0 or NaN."""
    if d in (0, None) or pd.isna(d):
        return None
    return n / d


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    HR-safe time detector:
    Prioritize Hire/Exit dates over generic dates.
    """
    # Priority 1: Lifecycle Dates (Strongest HR Signal)
    priority_candidates = [
        "hire date", "joining date", "start date",
        "exit date", "termination date", "term date",
        "dob", "birth date"
    ]
    
    # Priority 2: Generic Dates
    generic_candidates = ["date", "month", "year", "period"]

    cols = {c.lower(): c for c in df.columns}

    # Pass 1: Lifecycle
    for key in priority_candidates:
        for low, real in cols.items():
            if key in low and not df[real].isna().all():
                try:
                    pd.to_datetime(df[real].dropna().iloc[:10], errors="raise")
                    return real
                except Exception:
                    continue
    
    # Pass 2: Generic
    for key in generic_candidates:
        for low, real in cols.items():
            if key == low and not df[real].isna().all(): # Strict match for generic
                try:
                    pd.to_datetime(df[real].dropna().iloc[:10], errors="raise")
                    return real
                except Exception:
                    continue
    return None


# =====================================================
# HR / WORKFORCE DOMAIN
# =====================================================

class HRDomain(BaseDomain):
    name = "hr"
    description = "Human Resources & Workforce Analytics (Headcount, Attrition, Performance)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        HR data must have PEOPLE signals (Employee, Salary, Dept).
        """
        return any(
            resolve_column(df, c) is not None
            for c in [
                "employee", "employee_id", "staff",
                "department", "designation", "role", "title",
                "salary", "compensation", "pay", "payzone",
                "attrition", "exit", "termination",
                "performance", "rating", "score"
            ]
        )

    # ---------------- PREPROCESS ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        self.has_time_series = False

        if self.time_col:
            df = df.copy()
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.dropna(subset=[self.time_col])
            df = df.sort_values(self.time_col)
            self.has_time_series = df[self.time_col].nunique() >= 2

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}

        employee = (
            resolve_column(df, "employee")
            or resolve_column(df, "employee_id")
            or resolve_column(df, "staff")
        )
        salary = resolve_column(df, "salary") or resolve_column(df, "compensation")
        performance = (
            resolve_column(df, "performance") 
            or resolve_column(df, "rating") 
            or resolve_column(df, "score")
        )
        attrition = resolve_column(df, "attrition") or resolve_column(df, "termination")
        status = resolve_column(df, "status") or resolve_column(df, "employeestatus")
        
        # 1. Headcount
        if employee:
            kpis["headcount"] = df[employee].nunique()
        else:
            kpis["headcount"] = len(df)

        # 2. Attrition
        kpis["target_attrition_rate"] = 0.10
        kpis["attrition_rate"] = 0.0

        if attrition:
            # Check if numeric (0/1) or text
            if pd.api.types.is_numeric_dtype(df[attrition]):
                kpis["attrition_rate"] = df[attrition].mean()
            else:
                # Text based: "Voluntary", "Involuntary" implies exit
                # Anything not null/empty often means they left
                kpis["attrition_rate"] = df[attrition].notna().mean()

        elif status:
            # Fallback: Infer churn from status text (Regex)
            # HR specific terms: Terminated, Resigned, Voluntary
            status_series = df[status].astype(str).str.lower()
            exits = status_series.str.contains("term|exit|resign|left|inactive|vol", na=False)
            kpis["attrition_rate"] = exits.mean()

        # 3. Compensation
        if salary and pd.api.types.is_numeric_dtype(df[salary]):
            kpis["avg_salary"] = df[salary].mean()
            kpis["salary_std_dev"] = df[salary].std()

        # 4. Performance (Safe Normalization)
        if performance and pd.api.types.is_numeric_dtype(df[performance]):
            perf_series = df[performance].dropna()
            
            # Auto-detect 1-10 scale and normalize to 1-5
            if perf_series.mean() > 5:
                perf_series = perf_series / 2
            
            kpis["avg_performance_score"] = perf_series.mean()
            kpis["low_performance_rate"] = (perf_series < 3).mean()

        return kpis

    # ---------------- VISUALS (MAX 4) ----------------

    def generate_visuals(
        self, df: pd.DataFrame, output_dir: Path
    ) -> List[Dict[str, Any]]:

        visuals: List[Dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate KPIs once
        kpis = self.calculate_kpis(df)

        employee = (
            resolve_column(df, "employee")
            or resolve_column(df, "employee_id")
            or resolve_column(df, "staff")
        )
        department = resolve_column(df, "department") or resolve_column(df, "dept")
        salary = resolve_column(df, "salary") or resolve_column(df, "compensation")
        performance = (
            resolve_column(df, "performance") 
            or resolve_column(df, "rating") 
            or resolve_column(df, "score")
        )
        status = resolve_column(df, "status") or resolve_column(df, "employeestatus")
        title = resolve_column(df, "title") or resolve_column(df, "designation")

        def human_fmt(x, _):
            if abs(x) >= 1_000_000: return f"{x/1_000_000:.1f}M"
            if abs(x) >= 1_000: return f"{x/1_000:.0f}K"
            return str(int(x))

        # -------- Visual 1: Headcount by Department --------
        if department:
            p = output_dir / "headcount_by_department.png"
            
            # Use employee ID for count if available, else just row count
            if employee:
                counts = df.groupby(department)[employee].nunique()
            else:
                counts = df[department].value_counts()
                
            counts = counts.sort_values(ascending=False).head(10)

            plt.figure(figsize=(7, 4))
            counts.plot(kind="bar", color="#1f77b4")
            plt.title("Headcount by Department")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({"path": p, "caption": "Employee distribution across departments"})

        # -------- Visual 2: Salary Distribution --------
        if salary and pd.api.types.is_numeric_dtype(df[salary]):
            p = output_dir / "salary_distribution.png"

            plt.figure(figsize=(7, 4))
            df[salary].dropna().plot(kind="hist", bins=15, color="#ff7f0e", edgecolor='white')
            plt.title("Salary Distribution")
            plt.gca().xaxis.set_major_formatter(FuncFormatter(human_fmt))
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Distribution of employee compensation"
            })

        # -------- Visual 3: Performance Distribution OR Status Pie --------
        # Priority 1: Performance Histogram
        if performance and pd.api.types.is_numeric_dtype(df[performance]):
            p = output_dir / "performance_distribution.png"

            plt.figure(figsize=(7, 4))
            df[performance].dropna().plot(kind="hist", bins=10, color="#2ca02c", edgecolor='white')
            plt.title("Performance Score Distribution")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()

            visuals.append({
                "path": p,
                "caption": "Employee performance score spread"
            })
        # Priority 2: Status Pie
        elif status:
            p = output_dir / "status_breakdown.png"
            counts = df[status].value_counts().head(5)
            
            plt.figure(figsize=(6, 4))
            counts.plot(kind="pie", autopct='%1.1f%%')
            plt.ylabel("")
            plt.title("Employment Status Breakdown")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            
            visuals.append({"path": p, "caption": "Ratio of active vs terminated employees" })

        # -------- Visual 4: Designation / Title Breakdown --------
        # (Replaces Absence if Absence is missing, which is common in messy HR data)
        if title:
            p = output_dir / "title_breakdown.png"
            
            counts = df[title].value_counts().head(7)
            
            plt.figure(figsize=(7, 4))
            counts.plot(kind="barh", color="#9467bd")
            plt.title("Top Job Titles")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            
            visuals.append({
                "path": p,
                "caption": "Most common roles in the organization"
            })

        return visuals[:4]

    # ---------------- INSIGHTS ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        attrition = kpis.get("attrition_rate")
        perf = kpis.get("avg_performance_score")
        low_perf = kpis.get("low_performance_rate")
        
        # 1. Attrition
        if attrition is not None:
            if attrition > 0.20:
                insights.append({
                    "level": "RISK",
                    "title": "High Employee Attrition",
                    "so_what": f"Turnover is {attrition:.1%}, critical risk to stability (Target < 15%)."
                })
            elif attrition > 0.10:
                insights.append({
                    "level": "WARNING",
                    "title": "Rising Attrition",
                    "so_what": f"Turnover is {attrition:.1%}, slightly above healthy limits."
                })

        # 2. Performance
        if perf is not None and perf < 3.0:
            insights.append({
                "level": "WARNING",
                "title": "Low Average Performance",
                "so_what": f"Average performance score is {perf:.2f}/5."
            })

        if low_perf is not None and low_perf > 0.20:
            insights.append({
                "level": "WARNING",
                "title": "Large Low-Performance Segment",
                "so_what": f"{low_perf:.1%} of employees are underperforming."
            })

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Workforce Stable",
                "so_what": "Headcount, performance, and attrition metrics are within healthy ranges."
            })

        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        
        attrition = kpis.get("attrition_rate")
        low_perf = kpis.get("low_performance_rate")

        if attrition is not None and attrition > 0.15:
            recs.append({
                "action": "Conduct exit interviews and compensation benchmarking",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
        
        if low_perf is not None and low_perf > 0.20:
            recs.append({
                "action": "Initiate targeted training and performance improvement plans",
                "priority": "MEDIUM",
                "timeline": "This Quarter"
            })

        if not recs:
            recs.append({
                "action": "Continue workforce monitoring and engagement initiatives",
                "priority": "LOW",
                "timeline": "Ongoing"
            })

        return recs


# =====================================================
# DOMAIN DETECTOR (COLLISION PROOF)
# =====================================================

class HRDomainDetector(BaseDomainDetector):
    domain_name = "hr"

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        
        # 1. ANCHOR TOKENS (The Veto Power)
        # If these exist, it is almost certainly HR, not Customer/Retail
        anchors = {
            "salary", "compensation", "payroll", "hourly_rate", "payzone",
            "department", "designation", "job_title", "title",
            "hire_date", "termination_date", "dob", "marital", "race", "gender",
            "employee_id", "emp_id"
        }
        
        # 2. SHARED TOKENS (Ambiguous)
        # These appear in HR but also in Customer/Retail
        shared = {
            "performance", "rating", "score", "status", "id", "name", 
            "active", "date", "training"
        }

        anchor_hits = [c for c in cols if any(t in c for t in anchors)]
        shared_hits = [c for c in cols if any(t in c for t in shared)]

        # SCORING LOGIC
        # Anchor hits are worth 5x shared hits
        confidence = 0.0
        
        if len(anchor_hits) >= 1:
            # Strong signal found. 
            confidence = 0.8 + (len(anchor_hits) * 0.05)
        elif len(shared_hits) >= 3:
            # Only weak signals found. Check for specific "Employee" keyword context
            if any("employee" in c for c in cols):
                confidence = 0.85
            else:
                # If only "Score" and "Status" exist without "Employee", likely Customer data
                confidence = 0.2 
        
        return DomainDetectionResult(
            domain="hr",
            confidence=min(confidence, 1.0),
            signals={"anchor_hits": anchor_hits, "shared_hits": shared_hits},
        )


# =====================================================
# REGISTRATION
# =====================================================

def register(registry):
    registry.register(
        name="hr",
        domain_cls=HRDomain,
        detector_cls=HRDomainDetector,
    )
