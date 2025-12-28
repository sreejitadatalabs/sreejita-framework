import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
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
    """HR-safe time detector (Hiring, Exit, etc.)."""
    candidates = ["hire date", "joining date", "start date", "date", "year"]
    for c in df.columns:
        if any(x in c.lower() for x in candidates):
            try:
                pd.to_datetime(df[c].dropna().iloc[:5], errors="raise")
                return c
            except:
                continue
    return None


# =====================================================
# HR DOMAIN (UNIVERSAL 10/10)
# =====================================================

class HRDomain(BaseDomain):
    name = "hr"
    description = "Universal HR Intelligence (Workforce, Retention, Performance, Diversity)"

    # ---------------- PREPROCESS (CENTRALIZED STATE) ----------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.time_col = _detect_time_column(df)
        self.today = pd.Timestamp.today()

        # 1. Resolve columns ONCE here. Access via self.cols elsewhere.
        self.cols = {
            "emp": resolve_column(df, "employee_id") or resolve_column(df, "employee"),
            "dept": resolve_column(df, "department") or resolve_column(df, "team"),
            "salary": resolve_column(df, "salary") or resolve_column(df, "compensation"),
            "gender": resolve_column(df, "gender") or resolve_column(df, "sex"),
            "rating": resolve_column(df, "rating") or resolve_column(df, "performance"),
            "hire": resolve_column(df, "hire_date") or resolve_column(df, "joining_date"),
            "exit": resolve_column(df, "exit_date") or resolve_column(df, "termination_date"),
            "status": resolve_column(df, "status") or resolve_column(df, "active_status"),
            "revenue": resolve_column(df, "revenue"),
            "absence": resolve_column(df, "absence") or resolve_column(df, "leave_days"),
            "tenure": resolve_column(df, "tenure") or resolve_column(df, "years_of_service")
        }

        # 2. Date Cleaning
        if self.time_col:
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(self.time_col)

        return df

    # ---------------- KPIs ----------------

    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        kpis: Dict[str, Any] = {}
        c = self.cols

        # 1. Headcount
        kpis["headcount"] = df[c["emp"]].nunique() if c["emp"] else len(df)

        # 2. Attrition
        if c["exit"]:
            # Valid exits are non-null dates
            leavers = df[c["exit"]].notna().sum()
            kpis["attrition_rate"] = _safe_div(leavers, len(df))
            
            # FIX 1: Strict Early Attrition Logic (Handling NaNs correctly)
            if c["hire"]:
                hure = pd.to_datetime(df[c["hire"]], errors='coerce')
                exite = pd.to_datetime(df[c["exit"]], errors='coerce')
                tenure_days = (exite - hure).dt.days
                
                # Explicit boolean mask: must be notna AND < 90
                early_mask = (tenure_days.notna()) & (tenure_days < 90)
                kpis["early_attrition_rate"] = _safe_div(early_mask.sum(), leavers)

        elif c["status"]:
            # Regex fallback
            exits = df[c["status"]].astype(str).str.lower().str.contains("exit|left|term|resign", na=False)
            kpis["attrition_rate"] = exits.mean()

        # 3. Salary & Productivity
        if c["salary"]:
            # Clean currency symbols if necessary
            if df[c["salary"]].dtype == object:
                 df[c["salary"]] = pd.to_numeric(df[c["salary"]].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
            
            kpis["avg_salary"] = df[c["salary"]].mean()
            
            if c["revenue"] and c["emp"]:
                rev_per = df[c["revenue"]].sum() / df[c["emp"]].nunique()
                kpis["revenue_per_employee"] = rev_per
                kpis["productivity_index"] = _safe_div(rev_per, kpis["avg_salary"])

        # 4. Performance
        if c["rating"]:
            perf = pd.to_numeric(df[c["rating"]], errors='coerce').dropna()
            # Normalize 10-scale to 5
            if perf.mean() > 5: perf = perf / 2
            
            kpis["avg_performance"] = perf.mean()
            kpis["high_performance_rate"] = (perf >= 4.5).mean()
            kpis["low_performance_rate"] = (perf < 3.0).mean()

        # 5. Diversity (Gender)
        if c["gender"]:
            kpis["gender_diversity_ratio"] = df[c["gender"]].value_counts(normalize=True).min()
            
            if c["salary"]:
                # Pay Gap: (Max Avg - Min Avg) / Max Avg
                pay = df.groupby(c["gender"])[c["salary"]].mean()
                if len(pay) > 1:
                    kpis["pay_gap"] = (pay.max() - pay.min()) / pay.max()

        # 6. Absence
        if c["absence"]:
             kpis["avg_absence_days"] = df[c["absence"]].mean()

        return kpis

    # ---------------- VISUALS (SMART SELECTION) ----------------

    def generate_visuals(self, df: pd.DataFrame, output_dir: Path) -> List[Dict[str, Any]]:
        visuals = []
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self.cols
        kpis = self.calculate_kpis(df)

        def save(fig, name, caption, imp, cat):
            p = output_dir / name
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            visuals.append({
                "path": str(p),
                "caption": caption,
                "importance": imp,
                "category": cat
            })

        # 1. Headcount by Dept
        if c["dept"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            df[c["dept"]].value_counts().head(8).plot(kind="bar", ax=ax, color="#1f77b4")
            ax.set_title("Headcount by Dept")
            save(fig, "headcount.png", "Workforce distribution", 0.9, "workforce")

        # 2. Attrition by Dept (High Priority if Exits exist)
        if c["dept"] and (c["exit"] or c["status"]):
            # Logic to define 'is_exit' boolean series
            if c["exit"]:
                is_exit = df[c["exit"]].notna()
            else:
                is_exit = df[c["status"]].astype(str).str.contains("exit|left", case=False, na=False)
            
            # FIX 2: Safety fillna and .loc indexing
            is_exit = is_exit.fillna(False)
            exits = df.loc[is_exit]
            
            if not exits.empty:
                fig, ax = plt.subplots(figsize=(7, 4))
                exits[c["dept"]].value_counts().head(6).plot(kind="bar", ax=ax, color="red")
                ax.set_title("Exits by Department")
                # Boost importance if attrition is high
                imp = 0.95 if kpis.get("attrition_rate", 0) > 0.15 else 0.7
                save(fig, "attrition_dept.png", "Turnover hotspots", imp, "retention")

        # 3. Performance Distribution
        if c["rating"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            pd.to_numeric(df[c["rating"]], errors='coerce').hist(ax=ax, bins=5, color="orange")
            ax.set_title("Performance Ratings")
            save(fig, "performance.png", "Talent quality", 0.85, "performance")

        # 4. Salary Distribution
        if c["salary"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[c["salary"]].hist(ax=ax, bins=15, color="green")
            ax.set_title("Salary Bands")
            save(fig, "salary.png", "Compensation structure", 0.75, "compensation")

        # 5. Gender Pay Gap (Critical D&I)
        if kpis.get("pay_gap"):
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(c["gender"])[c["salary"]].mean().plot(kind="bar", ax=ax, color=["pink", "blue"])
            ax.set_title("Avg Pay by Gender")
            # Spike importance if gap > 15%
            save(fig, "pay_gap.png", "Pay equity", 1.0 if kpis["pay_gap"] > 0.15 else 0.8, "diversity")

        # 6. Hiring Trend
        if c["hire"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            try:
                df.set_index(pd.to_datetime(df[c["hire"]])).resample('M').size().plot(ax=ax)
                ax.set_title("Hiring Velocity")
                save(fig, "hiring.png", "Growth rate", 0.7, "recruiting")
            except: pass

        # 7. Revenue per Employee
        if kpis.get("revenue_per_employee"):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(["Rev/Emp"], [kpis["revenue_per_employee"]], color="teal")
            ax.set_title("Productivity")
            save(fig, "productivity.png", "Revenue per employee", 0.88, "productivity")

        # 8. Performance vs Pay (Correlation)
        if c["rating"] and c["salary"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df[c["rating"]], df[c["salary"]], alpha=0.5, color="purple")
            ax.set_title("Pay vs Performance")
            save(fig, "pay_perf.png", "Meritocracy check", 0.85, "fairness")

        # Sort & Pick Top 4
        visuals.sort(key=lambda v: v["importance"], reverse=True)
        return visuals[:4]

    # ---------------- INSIGHTS (COMPOSITE + ATOMIC) ----------------

    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        insights = []
        
        attrition = kpis.get("attrition_rate", 0)
        high_perf = kpis.get("high_performance_rate", 0)
        absence = kpis.get("avg_absence_days", 0)
        pay_gap = kpis.get("pay_gap", 0)
        prod = kpis.get("productivity_index", 2.0)

        # --- COMPOSITE SIGNALS (The "Smart" Layer) ---
        
        # A. Talent Drain (High Performance + High Attrition)
        if high_perf > 0.25 and attrition > 0.15:
            insights.append({
                "level": "CRITICAL",
                "title": "Talent Drain Detected",
                "so_what": f"Regrettable attrition! {high_perf:.1%} are top performers, but turnover is high ({attrition:.1%})."
            })

        # B. Burnout Risk (High Performance + High Absence)
        elif high_perf > 0.20 and absence > 12:
            insights.append({
                "level": "RISK",
                "title": "High Performer Burnout",
                "so_what": f"Top talent is taking excessive leave ({absence:.1f} days avg), often a precursor to exit."
            })

        # --- ATOMIC SIGNALS (The Standard Layer) ---

        # 1. Attrition (General)
        if attrition > 0.15:
            # Only flag if not already covered by "Talent Drain"
            if not any("Talent Drain" in i["title"] for i in insights):
                insights.append({
                    "level": "RISK",
                    "title": "High Attrition", 
                    "so_what": f"Turnover is {attrition:.1%}, threatening workforce stability."
                })
        
        # 2. Pay Equity
        if pay_gap > 0.15:
            insights.append({
                "level": "WARNING", 
                "title": "Pay Equity Risk", 
                "so_what": f"Gender pay gap is {pay_gap:.1%}, posing compliance and morale risks."
            })

        # 3. Productivity
        if prod < 1.2:
            insights.append({
                "level": "WARNING", 
                "title": "Low ROI on Compensation", 
                "so_what": "Revenue per employee is barely covering salary costs."
            })

        # 4. Success State
        if not insights:
            insights.append({
                "level": "INFO", 
                "title": "Workforce Stable", 
                "so_what": "Retention, performance, and diversity metrics are healthy."
            })
            
        return insights

    # ---------------- RECOMMENDATIONS ----------------

    def generate_recommendations(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        
        # 1. Check Insight Titles for Context
        titles = [i["title"] for i in self.generate_insights(df, kpis)]

        if any("Talent Drain" in t for t in titles):
            recs.append({"action": "Conduct immediate stay interviews with top 20% performers.", "priority": "HIGH"})
        
        if any("Burnout" in t for t in titles):
            recs.append({"action": "Audit workload distribution and mandate time-off.", "priority": "HIGH"})

        if kpis.get("pay_gap", 0) > 0.15:
            recs.append({"action": "Initiate formal compensation equity audit.", "priority": "HIGH"})

        if kpis.get("low_performance_rate", 0) > 0.15:
            recs.append({"action": "Review training & PIP programs.", "priority": "MEDIUM"})
            
        if not recs:
            recs.append({"action": "Maintain current engagement programs.", "priority": "LOW"})

        return recs


# =====================================================
# DOMAIN DETECTOR (WITH CONFIDENCE FLOOR FIX)
# =====================================================

class HRDomainDetector(BaseDomainDetector):
    domain_name = "hr"
    TOKENS = {"employee", "salary", "department", "rating", "hiring", "attrition", "leave"}

    def detect(self, df) -> DomainDetectionResult:
        hits = [c for c in df.columns if any(t in c.lower() for t in self.TOKENS)]
        confidence = min(len(hits)/3, 1.0)
        
        # FIX 3: Dominance Rule for Strong Signals
        STRONG_SIGNALS = ["salary", "attrition", "performance", "leave"]
        if any(any(x in c.lower() for x in STRONG_SIGNALS) for c in df.columns):
            confidence = max(confidence, 0.9)
            
        return DomainDetectionResult("hr", confidence, {"matched_columns": hits})

def register(registry):
    registry.register("hr", HRDomain, HRDomainDetector)
