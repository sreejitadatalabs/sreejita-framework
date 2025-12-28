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
    hiring, exit, attendance, evaluation timelines.
    """
    candidates = [
        "hire date", "joining date", "start date",
        "exit date", "termination date",
        "date", "month", "year"
    ]

    cols = {c.lower(): c for c in df.columns}

    for key in candidates:
        for low, real in cols.items():
            if key in low and not df[real].isna().all():
                try:
                    pd.to_datetime(df[real].dropna().iloc[:10], errors="raise")
                    return real
                except Exception:
                    continue
    return None


# =====================================================
# HR / WORKFORCE DOMAIN (v3.0 - FULL AUTHORITY)
# =====================================================

class HRDomain(BaseDomain):
    name = "hr"
    description = "Human Resources & Workforce Analytics (Headcount, Attrition, Performance)"

    # ---------------- VALIDATION ----------------

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        HR data must have people signals (Employee ID, Salary, Performance, etc.)
        """
        return any(
            resolve_column(df, c) is not None
            for c in [
                "employee", "employee_id", "staff",
                "department", "designation", "role",
                "salary", "compensation",
                "attrition", "exit", "termination",
                "attendance", "absence", "leave",
                "performance", "rating"
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
        performance = resolve_column(df, "performance") or resolve_column(df, "rating")
        attrition = resolve_column(df, "attrition")
        status = resolve_column(df, "status")
        absence = resolve_column(df, "absence") or resolve_column(df, "leave")

        # 1. Headcount
        if employee:
            kpis["headcount"] = df[employee].nunique()
        else:
            kpis["headcount"] = len(df)

        # 2. Attrition
        kpis["target_attrition_rate"] = 0.10

        if attrition and pd.api.types.is_numeric_dtype(df[attrition]):
            kpis["attrition_rate"] = df[attrition].mean()

        elif status:
            # Fallback: Infer churn from status text (Regex)
            status_series = df[status].astype(str).str.lower()
            exits = status_series.str.contains("exit|left|terminated|resigned|inactive", na=False)
            kpis["attrition_rate"] = exits.mean()

        # 3. Compensation
        if salary and pd.api.types.is_numeric_dtype(df[salary]):
            kpis["avg_salary"] = df[salary].mean()
            kpis["salary_std_dev"] = df[salary].std()
            
            # Compensation Ratio (High Earners vs Avg)
            if kpis["avg_salary"] > 0:
                kpis["top_10_percent_salary_ratio"] = df[salary].quantile(0.90) / kpis["avg_salary"]

        # 4. Performance (Safe Normalization)
        if performance and pd.api.types.is_numeric_dtype(df[performance]):
            perf_series = df[performance].dropna()
            
            # Auto-detect 1-10 scale and normalize to 1-5
            if perf_series.mean() > 5:
                perf_series = perf_series / 2
            
            kpis["avg_performance_score"] = perf_series.mean()
            # Low performer < 3.0 on 5.0 scale
            kpis["low_performance_rate"] = (perf_series < 3).mean()
            # High Performer Rate (> 4.5)
            kpis["high_performance_rate"] = (perf_series > 4.5).mean()

        # 5. Absenteeism
        if absence and pd.api.types.is_numeric_dtype(df[absence]):
            kpis["avg_absence_days"] = df[absence].mean()

        return kpis

    # ---------------- VISUALS (MAX 4) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
    visuals: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve columns safely
    emp_id = resolve_column(df, "employee_id")
    dept = resolve_column(df, "department")
    hire_date = resolve_column(df, "hire_date")
    exit_date = resolve_column(df, "exit_date")
    salary = resolve_column(df, "salary")
    gender = resolve_column(df, "gender")
    performance = resolve_column(df, "performance")
    revenue = resolve_column(df, "revenue")

    today = pd.Timestamp.today()

    # ---------------- 1. HEADCOUNT BY DEPARTMENT ----------------
    if dept:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[dept].value_counts().plot(kind="bar", ax=ax, color="#1f77b4")
        ax.set_title("Headcount by Department")
        ax.set_ylabel("Employees")

        p = output_dir / "headcount_by_department.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

        visuals.append({
            "path": str(p),
            "caption": "Distribution of employees across departments",
            "category": "workforce",
            "importance": 0.85,
        })

    # ---------------- 2. TURNOVER RATE ----------------
    if hire_date and exit_date:
        df_dates = df.copy()
        df_dates[hire_date] = pd.to_datetime(df_dates[hire_date], errors="coerce")
        df_dates[exit_date] = pd.to_datetime(df_dates[exit_date], errors="coerce")

        leavers = df_dates[exit_date].notna().sum()
        headcount = len(df_dates)
        turnover = leavers / headcount if headcount else 0

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Turnover"], [turnover * 100], color="#d62728")
        ax.set_title("Employee Turnover Rate")
        ax.set_ylabel("%")

        p = output_dir / "turnover_rate.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

        visuals.append({
            "path": str(p),
            "caption": f"Turnover rate: {turnover:.1%}",
            "category": "retention",
            "importance": 0.95 if turnover > 0.15 else 0.70,
        })

    # ---------------- 3. AVERAGE TENURE DISTRIBUTION ----------------
    if hire_date:
        df_tenure = df.copy()
        df_tenure[hire_date] = pd.to_datetime(df_tenure[hire_date], errors="coerce")
        df_tenure["tenure_years"] = (today - df_tenure[hire_date]).dt.days / 365

        fig, ax = plt.subplots(figsize=(6, 4))
        df_tenure["tenure_years"].dropna().plot(kind="hist", bins=10, ax=ax, color="#2ca02c")
        ax.set_title("Employee Tenure Distribution")
        ax.set_xlabel("Years")

        p = output_dir / "tenure_distribution.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

        visuals.append({
            "path": str(p),
            "caption": "Distribution of employee tenure",
            "category": "retention",
            "importance": 0.75,
        })

    # ---------------- 4. SALARY DISTRIBUTION ----------------
    if salary:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[salary].dropna().plot(kind="hist", bins=15, ax=ax, color="#9467bd")
        ax.set_title("Salary Distribution")
        ax.set_xlabel("Salary")

        p = output_dir / "salary_distribution.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

        visuals.append({
            "path": str(p),
            "caption": "Employee compensation distribution",
            "category": "compensation",
            "importance": 0.70,
        })

    # ---------------- 5. GENDER PAY GAP ----------------
    if salary and gender:
        pay_gap = df.groupby(gender)[salary].mean()

        fig, ax = plt.subplots(figsize=(5, 3))
        pay_gap.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
        ax.set_title("Average Pay by Gender")
        ax.set_ylabel("Average Salary")

        p = output_dir / "gender_pay_gap.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

        visuals.append({
            "path": str(p),
            "caption": "Gender pay comparison",
            "category": "diversity",
            "importance": 0.90,
        })

    # ---------------- 6. PERFORMANCE DISTRIBUTION ----------------
    if performance:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[performance].value_counts().plot(kind="bar", ax=ax, color="#8c564b")
        ax.set_title("Performance Rating Distribution")

        p = output_dir / "performance_distribution.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

        visuals.append({
            "path": str(p),
            "caption": "Employee performance spread",
            "category": "performance",
            "importance": 0.80,
        })

    # ---------------- 7. REVENUE PER EMPLOYEE ----------------
    if revenue and emp_id:
        rev_per_emp = df[revenue].sum() / df[emp_id].nunique()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Revenue / Employee"], [rev_per_emp], color="#17becf")
        ax.set_title("Revenue per Employee")

        p = output_dir / "revenue_per_employee.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

        visuals.append({
            "path": str(p),
            "caption": "Workforce productivity indicator",
            "category": "productivity",
            "importance": 0.88,
        })

    # ---------------- FINAL RANKING ----------------
    visuals.sort(key=lambda v: v.get("importance", 0), reverse=True)
    return visuals

    # ---------------- ATOMIC INSIGHTS (WITH DOMINANCE RULE) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []

        attrition = kpis.get("attrition_rate")
        perf = kpis.get("avg_performance_score")
        low_perf = kpis.get("low_performance_rate")
        absence = kpis.get("avg_absence_days")

        # === STEP 1: Generate Composite Insights FIRST ===
        composite_insights = self.generate_composite_insights(df, kpis)
        
        dominant_titles = {
            i["title"] for i in composite_insights
            if i["level"] in {"RISK", "WARNING"}
        }

        # Suppression Rules
        suppress_attrition = any(
            t in dominant_titles 
            for t in {"Talent Drain (Regrettable Attrition)", "Retention Risk Likely Linked to Compensation"}
        )
        
        suppress_absence = any("Burnout Risk" in t for t in dominant_titles)

        # === STEP 2: Atomic Insights (Guarded) ===
        
        # Attrition
        if attrition is not None and not suppress_attrition:
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

        # Performance
        if perf is not None and perf < 3.5:
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

        # Absenteeism
        if absence is not None and absence > 10 and not suppress_absence:
            insights.append({
                "level": "INFO",
                "title": "Elevated Absenteeism",
                "so_what": f"Average absence is {absence:.1f} days per employee."
            })

        # === STEP 3: Append Composite Insights LAST ===
        insights += composite_insights

        if not insights:
            insights.append({
                "level": "INFO",
                "title": "Workforce Stable",
                "so_what": "Headcount, performance, and attrition metrics are within healthy ranges."
            })

        return insights

    # ---------------- COMPOSITE INSIGHTS (HR v3.0) ----------------

    def generate_composite_insights(
        self, df: pd.DataFrame, kpis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        HR v3 Composite Intelligence Layer.
        Connects multiple signals to detect burnout, flight risk, and alignment.
        """
        insights: List[Dict[str, Any]] = []

        attrition = kpis.get("attrition_rate")
        high_perf = kpis.get("high_performance_rate")
        avg_sal = kpis.get("avg_salary")
        absence = kpis.get("avg_absence_days")
        top_sal_ratio = kpis.get("top_10_percent_salary_ratio")
        headcount = kpis.get("headcount", 0)

        # 1. Burnout Risk: High Performance + High Absence
        if high_perf is not None and absence is not None:
            if high_perf > 0.20 and absence > 12:
                insights.append({
                    "level": "RISK",
                    "title": "High Performer Burnout Risk",
                    "so_what": (
                        f"20%+ of staff are high performers, but absenteeism is high "
                        f"({absence:.1f} days). This pattern often precedes burnout-driven exit."
                    )
                })

        # 2. "Regrettable Attrition" Risk: High Performance + High Attrition
        if high_perf is not None and attrition is not None:
            if high_perf > 0.30 and attrition > 0.15:
                insights.append({
                    "level": "RISK",
                    "title": "Talent Drain (Regrettable Attrition)",
                    "so_what": (
                        f"You have a strong talent pool ({high_perf:.1%} high performers), "
                        f"but attrition is high ({attrition:.1%}). You may be losing your best people."
                    )
                })

        # 3. Compensation Misalignment: Low Pay + High Attrition
        if attrition is not None and avg_sal is not None:
            # Heuristic: If attrition is extremely high (>25%), pay is often a factor
            if attrition > 0.25:
                insights.append({
                    "level": "WARNING",
                    "title": "Retention Risk Likely Linked to Compensation",
                    "so_what": (
                        f"Attrition is critical ({attrition:.1%}). Review compensation "
                        f"benchmarks immediately."
                    )
                })

        # 4. Pay Equity / Structure Skew (Smart Guarded)
        if top_sal_ratio is not None:
            # GUARD: Only trigger if team is large enough (>30) to avoid small team noise
            if top_sal_ratio > 4.0 and headcount > 30:
                 insights.append({
                    "level": "INFO",
                    "title": "Steep Compensation Hierarchy",
                    "so_what": (
                        f"Top 10% earners make {top_sal_ratio:.1f}x the average. "
                        f"Ensure this aligns with your organizational philosophy."
                    )
                })

        return insights

    # ---------------- RECOMMENDATIONS (AUTHORITY BASED) ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        
        # 1. Check Composite Context for Action Authority
        composite = self.generate_composite_insights(df, kpis)
        composite_titles = [i["title"] for i in composite]

        # AUTHORITY RULES: Root causes mandate specific actions
        if any("Talent Drain" in t for t in composite_titles):
            recs.append({
                "action": "Conduct immediate retention audit and compensation review for high performers",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
            return recs

        if any("Burnout Risk" in t for t in composite_titles):
            recs.append({
                "action": "Initiate workload balancing and mandatory leave review for high-risk teams",
                "priority": "HIGH",
                "timeline": "This Month"
            })
            return recs

        if any("Retention Risk" in t for t in composite_titles):
            recs.append({
                "action": "Review compensation benchmarks against market rates immediately",
                "priority": "HIGH",
                "timeline": "Immediate"
            })
            return recs

        if any("Steep Compensation Hierarchy" in t for t in composite_titles):
            recs.append({
                "action": "Review pay bands and equity distribution for fairness",
                "priority": "MEDIUM",
                "timeline": "Next Quarter"
            })
            # Don't return, allow other recs as this is INFO/MEDIUM

        # 2. Fallback to Atomic Recommendations
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
# DOMAIN DETECTOR (SMART / DOMINANCE BASED)
# =====================================================

class HRDomainDetector(BaseDomainDetector):
    domain_name = "hr"

    HR_TOKENS: Set[str] = {
        # Identity
        "employee", "employee_id", "staff",
        
        # Org structure (HR-exclusive)
        "department", "designation", "role", "manager",
        
        # Compensation (HR-exclusive)
        "salary", "compensation", "ctc", "payroll", "bonus",
        
        # Lifecycle (HR-exclusive)
        "attrition", "exit", "resign", "termination", "joining", "hire",
        
        # Performance & attendance
        "performance", "rating", "kpi", 
        "leave", "absence", "attendance"
    }

    def detect(self, df) -> DomainDetectionResult:
        cols = {str(c).lower() for c in df.columns}
        
        hits = [c for c in cols if any(t in c for t in self.HR_TOKENS)]
        
        # Base confidence
        confidence = min(len(hits) / 3, 1.0)
        
        # 🔑 HR DOMINANCE RULE (MANDATORY FIX)
        # If strong HR exclusive signals exist, HR overrides Customer
        HR_STRONG_SIGNALS = {
            "salary", "compensation", "payroll", "ctc",
            "attrition", "termination", "resignation",
            "performance", "rating", "leave", "attendance"
        }

        strong_hits = [
            c for c in cols
            if any(sig in c for sig in HR_STRONG_SIGNALS)
        ]
        
        if strong_hits:
            confidence = max(confidence, 0.9)

        return DomainDetectionResult(
            domain="hr",
            confidence=confidence,
            signals={"matched_columns": hits},
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
