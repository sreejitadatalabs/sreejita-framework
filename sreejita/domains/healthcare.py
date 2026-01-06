# =====================================================
# UNIVERSAL HEALTHCARE DOMAIN — FOUNDATIONAL DEFINITIONS
# Sreejita Framework v3.5.x
# =====================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from sreejita.core.capabilities import Capability
from sreejita.core.column_resolver import resolve_column
from sreejita.core.dataset_shape import detect_dataset_shape
from .base import BaseDomain

from sreejita.domains.contracts import BaseDomainDetector, DomainDetectionResult

# =====================================================
# CONSTANTS
# =====================================================

MIN_SAMPLE_SIZE = 30

# NOTE:
# 'flag' is a generalized binary outcome proxy used across:
# mortality, alerts, specimen rejection, screening, immunization, etc.
# This is intentional for universal-domain support in v3.x.

# =====================================================
# SUB-DOMAINS (CANONICAL ENUM)
# =====================================================

class HealthcareSubDomain(str, Enum):
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    DIAGNOSTICS = "diagnostics"
    PHARMACY = "pharmacy"
    PUBLIC_HEALTH = "public_health"
    MIXED = "mixed"
    UNKNOWN = "unknown"

# =====================================================
# VISUAL INTELLIGENCE MAP (LOCKED CONTRACT)
# =====================================================

HEALTHCARE_VISUAL_MAP: Dict[str, List[str]] = {
    HealthcareSubDomain.HOSPITAL.value: [
        "avg_los_trend",
        "bed_turnover",
        "readmission_risk",
        "discharge_hour",
        "acuity_vs_staffing",
        "ed_boarding",
        "mortality_trend",
    ],
    HealthcareSubDomain.CLINIC.value: [
        "no_show_by_day",
        "wait_time_split",
        "appointment_lag",
        "provider_utilization",
        "demographic_reach",
        "referral_funnel",
        "telehealth_mix",
    ],
    HealthcareSubDomain.DIAGNOSTICS.value: [
        "tat_percentiles",
        "critical_alert_time",
        "specimen_rejection",
        "device_downtime",
        "order_heatmap",
        "repeat_scan",
        "ordering_variance",
    ],
    HealthcareSubDomain.PHARMACY.value: [
        "spend_velocity",
        "refill_gap",
        "therapeutic_spend",
        "generic_rate",
        "prescribing_variance",
        "inventory_turn",
        "drug_alerts",
    ],
    HealthcareSubDomain.PUBLIC_HEALTH.value: [
        "incidence_geo",
        "cohort_growth",
        "prevalence_age",
        "access_gap",
        "program_effect",
        "sdoh_overlay",
        "immunization_rate",
    ],
}

# =====================================================
# HEALTHCARE KPI INTELLIGENCE MAP (LOCKED CONTRACT)
# =====================================================

HEALTHCARE_KPI_MAP: Dict[str, List[str]] = {
    HealthcareSubDomain.HOSPITAL.value: [
        "avg_los",
        "readmission_rate",
        "bed_occupancy_rate",
        "case_mix_index",
        "hcahps_score",
        "mortality_rate",
        "er_boarding_time",
        "labor_cost_per_day",
        "surgical_complication_rate",
    ],
    HealthcareSubDomain.CLINIC.value: [
        "no_show_rate",
        "avg_wait_time",
        "provider_productivity",
        "third_next_available",
        "referral_conversion_rate",
        "visit_cycle_time",
        "patient_acquisition_cost",
        "telehealth_mix",
        "net_collection_ratio",
    ],
    HealthcareSubDomain.DIAGNOSTICS.value: [
        "avg_tat",
        "critical_alert_time",
        "specimen_rejection_rate",
        "equipment_downtime_rate",
        "repeat_test_rate",
        "tests_per_fte",
        "supply_cost_per_test",
        "order_completeness_ratio",
        "outpatient_market_share",
    ],
    HealthcareSubDomain.PHARMACY.value: [
        "days_supply_on_hand",
        "generic_dispensing_rate",
        "refill_adherence_rate",
        "cost_per_rx",
        "med_error_rate",
        "pharmacist_intervention_rate",
        "inventory_turnover",
        "spend_velocity",
        "avg_patient_wait_time",
    ],
    HealthcareSubDomain.PUBLIC_HEALTH.value: [
        "incidence_per_100k",
        "sdoh_risk_score",
        "screening_coverage_rate",
        "chronic_readmission_rate",
        "immunization_rate",
        "provider_access_gap",
        "ed_visits_per_1k",
        "cost_per_member",
        "healthy_days_index",
    ],
}

# =====================================================
# HEALTHCARE INSIGHT INTELLIGENCE MAP (LOCKED)
# =====================================================

HEALTHCARE_INSIGHT_MAP: Dict[str, List[str]] = {
    HealthcareSubDomain.HOSPITAL.value: [
        "throughput_bottleneck",
        "clinical_safety_alert",
        "bed_capacity_strain",
        "acuity_labor_mismatch",
        "revenue_leakage",
        "quality_stability",
        "patient_experience_gap",
        "physician_variance",
        "supply_chain_variance",
    ],
    HealthcareSubDomain.CLINIC.value: [
        "access_barrier",
        "productivity_variance",
        "referral_leakage",
        "workflow_inefficiency",
        "revenue_risk",
        "telehealth_shift",
        "demographic_gap",
        "front_desk_variance",
        "financial_health",
    ],
    HealthcareSubDomain.DIAGNOSTICS.value: [
        "service_level_gap",
        "life_safety_risk",
        "technical_waste",
        "pre_analytical_failure",
        "capacity_overload",
        "asset_depreciation",
        "efficiency_plateau",
        "quality_variance",
        "market_opportunity",
    ],
    HealthcareSubDomain.PHARMACY.value: [
        "economic_pressure",
        "adherence_risk",
        "safety_barrier",
        "inventory_inefficiency",
        "intervention_impact",
        "payer_mix_shift",
        "throughput_constraint",
        "prescribing_variance",
        "inventory_waste",
    ],
    HealthcareSubDomain.PUBLIC_HEALTH.value: [
        "equity_gap",
        "prevention_failure",
        "access_desert",
        "chronic_cost_driver",
        "outbreak_risk",
        "environmental_influence",
        "program_success",
        "member_engagement_gap",
        "governance_risk",
    ],
}

# =====================================================
# HEALTHCARE RECOMMENDATION MAP (LOCKED)
# =====================================================

HEALTHCARE_RECOMMENDATION_MAP: Dict[str, List[str]] = {
    HealthcareSubDomain.HOSPITAL.value: [
        "discharge_huddle",
        "clinical_pathway_standardization",
        "bed_assignment_automation",
        "post_discharge_review",
        "acuity_based_staffing",
        "patient_feedback_rounding",
        "demand_forecasting",
        "or_turnover_optimization",
        "implant_contract_review",
    ],
    HealthcareSubDomain.CLINIC.value: [
        "appointment_reminders",
        "open_access_scheduling",
        "telehealth_standardization",
        "checkin_workflow_lean",
        "provider_rvu_dashboard",
        "referral_centralization",
        "patient_portal",
        "rooming_velocity_optimization",
        "targeted_marketing",
    ],
    HealthcareSubDomain.DIAGNOSTICS.value: [
        "analyzer_upgrade",
        "critical_alert_software",
        "specimen_training",
        "preventive_maintenance",
        "ehr_interface_automation",
        "stat_track",
        "managed_services",
        "barcode_chain",
        "physician_portal",
    ],
    HealthcareSubDomain.PHARMACY.value: [
        "refill_reminders",
        "meds_to_beds",
        "central_fill",
        "formulary_standardization",
        "drug_interaction_software",
        "dispensing_robot",
        "inventory_audit",
        "manufacturer_contracts",
        "staffing_optimization",
    ],
    HealthcareSubDomain.PUBLIC_HEALTH.value: [
        "mobile_health_units",
        "screening_campaign",
        "food_programs",
        "chronic_protocols",
        "immunization_registry",
        "disparity_audit",
        "community_health_workers",
        "sdoh_referrals",
        "whole_person_platform",
    ],
}

# =====================================================
# SAFE SIGNAL DETECTION (STRICT, DOMAIN-AWARE)
# =====================================================

def _has_signal(
    df: pd.DataFrame,
    col: Optional[str],
    min_coverage: float = 0.3,
) -> bool:
    """
    Column must exist AND meet minimum non-null coverage.
    Generic columns are intentionally handled upstream.
    """
    if not col or col not in df.columns or len(df) == 0:
        return False

    return (df[col].notna().sum() / len(df)) >= min_coverage

# =====================================================
# SUB-DOMAIN ELIGIBILITY CONTRACT (HARD GATE)
# =====================================================

SUBDOMAIN_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    HealthcareSubDomain.HOSPITAL.value: [
        "date", "discharge_date", "los"
    ],
    HealthcareSubDomain.CLINIC.value: [
        "doctor", "duration"
    ],
    HealthcareSubDomain.DIAGNOSTICS.value: [
        "duration", "encounter"
    ],
    HealthcareSubDomain.PHARMACY.value: [
        "fill_date", "supply"
    ],
    HealthcareSubDomain.PUBLIC_HEALTH.value: [
        "population"
    ],
}

def _eligible_subdomain(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    sub: str,
) -> bool:
    """
    A sub-domain is eligible ONLY if ALL required
    exclusive columns are present with signal.
    """
    required = SUBDOMAIN_REQUIRED_COLUMNS.get(sub, [])
    if not required:
        return False

    return all(
        _has_signal(df, cols.get(col))
        for col in required
    )


# =====================================================
# UNIVERSAL SUB-DOMAIN INFERENCE — HEALTHCARE (FIXED)
# =====================================================

def infer_healthcare_subdomains(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
) -> Dict[str, float]:
    """
    Deterministic, evidence-gated healthcare sub-domain inference.

    Rules:
    - One sub-domain by default
    - Multiple only if independently eligible
    - Pharmacy requires fill_date + supply (NEVER cost-only)
    """

    scores: Dict[str, float] = {}

    # -------------------------------
    # HOSPITAL (PRIMARY BY DEFAULT)
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.HOSPITAL.value):
        signals = sum([
            _has_signal(df, cols.get("los")),
            _has_signal(df, cols.get("bed_id")),
            _has_signal(df, cols.get("admit_type")),
            _has_signal(df, cols.get("date")) and _has_signal(df, cols.get("discharge_date")),
        ])
        scores[HealthcareSubDomain.HOSPITAL.value] = round(
            min(1.0, 0.35 + 0.15 * signals), 2
        )

    # -------------------------------
    # CLINIC
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.CLINIC.value):
        signals = sum([
            _has_signal(df, cols.get("doctor")),
            _has_signal(df, cols.get("duration")),
            _has_signal(df, cols.get("facility")),
        ])
        scores[HealthcareSubDomain.CLINIC.value] = round(
            min(0.85, 0.30 + 0.15 * signals), 2
        )

    # -------------------------------
    # DIAGNOSTICS
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.DIAGNOSTICS.value):
        signals = sum([
            _has_signal(df, cols.get("duration")),
            _has_signal(df, cols.get("encounter")),
            _has_signal(df, cols.get("flag")),
        ])
        scores[HealthcareSubDomain.DIAGNOSTICS.value] = round(
            min(0.85, 0.30 + 0.15 * signals), 2
        )

    # -------------------------------
    # PHARMACY (STRICT — NO GENERIC LEAKAGE)
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.PHARMACY.value):
        signals = sum([
            _has_signal(df, cols.get("fill_date")),
            _has_signal(df, cols.get("supply")),
        ])
        scores[HealthcareSubDomain.PHARMACY.value] = round(
            min(0.80, 0.35 + 0.20 * signals), 2
        )

    # -------------------------------
    # PUBLIC HEALTH
    # -------------------------------
    if _eligible_subdomain(df, cols, HealthcareSubDomain.PUBLIC_HEALTH.value):
        signals = sum([
            _has_signal(df, cols.get("population")),
            _has_signal(df, cols.get("flag")),
        ])
        scores[HealthcareSubDomain.PUBLIC_HEALTH.value] = round(
            min(0.90, 0.40 + 0.20 * signals), 2
        )

    # -------------------------------
    # FINAL RESOLUTION
    # -------------------------------
    if not scores:
        return {HealthcareSubDomain.UNKNOWN.value: 1.0}

    # If only one sub-domain → return it
    if len(scores) == 1:
        return scores

    # If multiple → ensure meaningful separation
    strongest = max(scores.values())
    filtered = {
        k: v for k, v in scores.items()
        if v >= max(0.45, strongest - 0.20)
    }

    return filtered    

# =====================================================
# HEALTHCARE DOMAIN (FIXED, SUBDOMAIN-SAFE)
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"

    # -------------------------------------------------
    # KPI ACCESS HELPERS (SAFE)
    # -------------------------------------------------
    @staticmethod
    def get_kpi(kpis: Dict[str, Any], sub: str, key: str):
        namespaced = f"{sub}_{key}"
        return kpis.get(namespaced, kpis.get(key))

    @staticmethod
    def get_kpi_confidence(kpis: Dict[str, Any], sub: str, key: str) -> float:
        conf_map = kpis.get("_confidence", {})
        return conf_map.get(f"{sub}_{key}", conf_map.get(key, 0.6))

    # -------------------------------------------------
    # PREPROCESS (UNIVERSAL + SAFE)
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Universal Healthcare preprocessing.

        Guarantees:
        - No sub-domain leakage
        - No pharmacy hallucination
        - No generic column misuse
        - Deterministic column semantics
        """

        df = df.copy()
        self.shape_info = detect_dataset_shape(df)

        # -------------------------------------------------
        # CANONICAL COLUMN RESOLUTION (SCOPED)
        # -------------------------------------------------
        self.cols = {
            # ---------------- IDENTITY ----------------
            "pid": resolve_column(df, "patient_id"),
            "encounter": (
                resolve_column(df, "encounter_id")
                or resolve_column(df, "visit_id")
            ),

            # ---------------- TIME ----------------
            "date": resolve_column(df, "admission_date"),
            "discharge_date": resolve_column(df, "discharge_date"),

            # Pharmacy-only (do NOT infer meaning yet)
            "fill_date": resolve_column(df, "fill_date"),

            # ---------------- DURATION ----------------
            "los": resolve_column(df, "length_of_stay"),
            "duration": resolve_column(df, "duration"),

            # ---------------- COST (NEUTRAL) ----------------
            # Cost is intentionally neutral; sub-domain decides meaning
            "cost": resolve_column(df, "cost"),

            # ---------------- FLAGS (RAW ONLY) ----------------
            "readmitted": resolve_column(df, "readmitted"),
            "flag": resolve_column(df, "flag"),

            # ---------------- OPERATIONS ----------------
            "facility": resolve_column(df, "facility"),
            "doctor": resolve_column(df, "doctor"),
            "admit_type": resolve_column(df, "admission_type"),
            "bed_id": resolve_column(df, "bed_id"),

            # ---------------- PHARMACY / POPULATION ----------------
            "supply": resolve_column(df, "supply"),
            "population": resolve_column(df, "population"),
        }

        # -------------------------------------------------
        # NUMERIC NORMALIZATION (STRICT)
        # -------------------------------------------------
        for key in ("los", "duration", "cost", "supply", "population"):
            col = self.cols.get(key)
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -------------------------------------------------
        # DATE NORMALIZATION
        # -------------------------------------------------
        for key in ("date", "discharge_date", "fill_date"):
            col = self.cols.get(key)
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # -------------------------------------------------
        # BOOLEAN NORMALIZATION (LIMITED SCOPE)
        # -------------------------------------------------
        BOOL_MAP = {
            "yes": 1, "y": 1, "true": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "0": 0,
        }

        for key in ("readmitted", "flag"):
            col = self.cols.get(key)
            if col in df.columns:
                s = df[col].astype(str).str.lower().str.strip()
                df[col] = pd.to_numeric(
                    s.map(BOOL_MAP).where(s.map(BOOL_MAP).notna(), df[col]),
                    errors="coerce",
                )

        # -------------------------------------------------
        # DERIVE LOS (HOSPITAL ONLY, GUARDED)
        # -------------------------------------------------
        if (
            self.cols.get("los") is None
            and self.cols.get("date") in df.columns
            and self.cols.get("discharge_date") in df.columns
        ):
            delta = (
                df[self.cols["discharge_date"]] -
                df[self.cols["date"]]
            ).dt.days

            delta = delta.where(delta.between(0, 365))
            df["__derived_los"] = pd.to_numeric(delta, errors="coerce")
            self.cols["los"] = "__derived_los"

        # -------------------------------------------------
        # CANONICAL TIME COLUMN (NO PHARMACY PREFERENCE)
        # -------------------------------------------------
        self.time_col = (
            self.cols.get("date")
            or self.cols.get("discharge_date")
            or self.cols.get("fill_date")
        )

        return df
   
    # -------------------------------------------------
    # KPI ENGINE (UNIVERSAL, SUB-DOMAIN HARD-LOCKED)
    # -------------------------------------------------
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        volume = int(len(df))
    
        # -------------------------------------------------
        # STEP 1: INFER SUB-DOMAINS (EVIDENCE-BASED)
        # -------------------------------------------------
        inferred = infer_healthcare_subdomains(df, self.cols)
    
        if not inferred or HealthcareSubDomain.UNKNOWN.value in inferred:
            active_subs = {}
            primary_sub = HealthcareSubDomain.UNKNOWN.value
            is_mixed = False
        else:
            # sort by confidence
            ordered = sorted(
                inferred.items(),
                key=lambda x: x[1],
                reverse=True
            )
    
            # primary always strongest
            primary_sub, primary_conf = ordered[0]
    
            # allow secondary ONLY if close enough
            active_subs = {
                primary_sub: primary_conf
            }
    
            for sub, conf in ordered[1:]:
                if conf >= 0.5 and abs(primary_conf - conf) <= 0.2:
                    active_subs[sub] = conf
    
            is_mixed = len(active_subs) > 1
    
        # -------------------------------------------------
        # STEP 2: BASE KPI CONTEXT
        # -------------------------------------------------
        kpis: Dict[str, Any] = {
            "primary_sub_domain": (
                HealthcareSubDomain.MIXED.value if is_mixed else primary_sub
            ),
            "sub_domains": active_subs,
            "record_count": volume,
            "total_volume": volume,
            "data_completeness": round(1 - df.isna().mean().mean(), 3),
        }
    
        if self.time_col and self.time_col in df.columns and df[self.time_col].notna().any():
            kpis["time_coverage_days"] = int(
                (df[self.time_col].max() - df[self.time_col].min()).days
            )
        else:
            kpis["time_coverage_days"] = None
    
        if volume < MIN_SAMPLE_SIZE:
            kpis["data_warning"] = "Sample size below recommended threshold"
    
        # -------------------------------------------------
        # SAFE KPI HELPERS
        # -------------------------------------------------
        def safe_mean(col):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.mean()) if s.notna().any() else None
    
        def safe_rate(col):
            if not col or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return float((s > 0).mean()) if s.notna().any() else None
    
        # -------------------------------------------------
        # STEP 3: KPI COMPUTATION (STRICT PER SUB-DOMAIN)
        # -------------------------------------------------
        for sub, conf in active_subs.items():
            prefix = f"{sub}_" if is_mixed else ""
    
            # ---------------- HOSPITAL ----------------
            if sub == HealthcareSubDomain.HOSPITAL.value:
                los = self.cols.get("los")
                if los in df.columns:
                    avg_los = safe_mean(los)
                    kpis[f"{prefix}avg_los"] = avg_los
                    kpis[f"{prefix}long_stay_rate"] = (
                        (df[los] > 7).mean() if avg_los is not None else None
                    )
    
                kpis[f"{prefix}readmission_rate"] = safe_rate(self.cols.get("readmitted"))
                kpis[f"{prefix}mortality_rate"] = safe_rate(self.cols.get("flag"))
                kpis[f"{prefix}er_boarding_time"] = safe_mean(self.cols.get("duration"))
    
                bed = self.cols.get("bed_id")
                if bed in df.columns and volume > 0:
                    kpis[f"{prefix}bed_occupancy_rate"] = df[bed].nunique() / volume
    
            # ---------------- CLINIC ----------------
            if sub == HealthcareSubDomain.CLINIC.value:
                visits = volume
                doctor = self.cols.get("doctor")
    
                providers = (
                    df[doctor].nunique()
                    if doctor in df.columns
                    else None
                )
    
                kpis[f"{prefix}no_show_rate"] = safe_rate(self.cols.get("readmitted"))
                kpis[f"{prefix}avg_wait_time"] = safe_mean(self.cols.get("duration"))
                kpis[f"{prefix}visit_cycle_time"] = safe_mean(self.cols.get("duration"))
    
                if providers and providers > 0:
                    kpis[f"{prefix}visits_per_provider"] = visits / providers
    
            # ---------------- DIAGNOSTICS ----------------
            if sub == HealthcareSubDomain.DIAGNOSTICS.value:
                staff = (
                    df[self.cols["doctor"]].nunique()
                    if self.cols.get("doctor") in df.columns
                    else None
                )
    
                kpis[f"{prefix}avg_tat"] = safe_mean(self.cols.get("duration"))
                kpis[f"{prefix}specimen_rejection_rate"] = safe_rate(self.cols.get("flag"))
    
                if staff and staff > 0:
                    kpis[f"{prefix}tests_per_fte"] = volume / staff
    
            # ---------------- PHARMACY (STRICT) ----------------
            if sub == HealthcareSubDomain.PHARMACY.value:
                # HARD REQUIREMENTS
                if not (
                    self.cols.get("fill_date") in df.columns
                    and self.cols.get("supply") in df.columns
                ):
                    continue  # do NOT compute pharmacy KPIs
    
                kpis[f"{prefix}days_supply_on_hand"] = safe_mean(self.cols.get("supply"))
                kpis[f"{prefix}cost_per_rx"] = safe_mean(self.cols.get("cost"))
                kpis[f"{prefix}rx_volume"] = volume
    
            # ---------------- PUBLIC HEALTH ----------------
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                pop = safe_mean(self.cols.get("population"))
                cases = safe_rate(self.cols.get("flag"))
    
                if pop and cases is not None:
                    kpis[f"{prefix}incidence_per_100k"] = cases * 100_000
    
        # -------------------------------------------------
        # STEP 4: CACHE + RETURN
        # -------------------------------------------------
        self._last_kpis = kpis
        return kpis

    # -------------------------------------------------
    # VISUAL ENGINE (STRICT, KPI-LOCKED)
    # -------------------------------------------------
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
    
        output_dir.mkdir(parents=True, exist_ok=True)
        visuals: List[Dict[str, Any]] = []
    
        # -------------------------------------------------
        # SINGLE SOURCE OF TRUTH: KPIs
        # -------------------------------------------------
        kpis = getattr(self, "_last_kpis", None)
        if not isinstance(kpis, dict):
            kpis = self.calculate_kpis(df)
            self._last_kpis = kpis
    
        # -------------------------------------------------
        # ACTIVE SUB-DOMAINS (FROM KPI ENGINE ONLY)
        # -------------------------------------------------
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
        primary = kpis.get("primary_sub_domain")
    
        # UNKNOWN → NO VISUALS
        if not active_subs or primary == HealthcareSubDomain.UNKNOWN.value:
            return []
    
        # Mixed handled explicitly
        if primary == HealthcareSubDomain.MIXED.value:
            visual_subs = list(active_subs.keys())
        else:
            visual_subs = [primary]
    
        # -------------------------------------------------
        # SUB-DOMAIN CONFIDENCE WEIGHTING
        # -------------------------------------------------
        def sub_domain_weight(sub: str) -> float:
            return round(min(1.0, max(0.3, active_subs.get(sub, 0.3))), 2)
    
        # -------------------------------------------------
        # VISUAL REGISTRATION (SAFE)
        # -------------------------------------------------
        def register_visual(
            fig,
            name: str,
            caption: str,
            importance: float,
            base_confidence: float,
            sub_domain: str,
        ):
            fname = name if name.endswith(".png") else f"{name}.png"
            path = output_dir / fname
    
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
    
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": float(importance),
                "confidence": round(
                    min(0.95, base_confidence * sub_domain_weight(sub_domain)), 2
                ),
                "sub_domain": sub_domain,
                "inference_type": (
                    "derived" if "trend" in caption.lower()
                    else "proxy" if "proxy" in caption.lower()
                    else "direct"
                ),
            })
    
        # -------------------------------------------------
        # VISUAL DISPATCH (SUB-DOMAIN GUARDED)
        # -------------------------------------------------
        for sub in visual_subs:
    
            # HARD GUARDS — NEVER RENDER WITHOUT DATA
            if sub == HealthcareSubDomain.PHARMACY.value:
                if not (
                    self.cols.get("fill_date") in df.columns
                    and self.cols.get("supply") in df.columns
                ):
                    continue
    
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                if self.cols.get("population") not in df.columns:
                    continue
    
            visual_keys = HEALTHCARE_VISUAL_MAP.get(sub, [])
            if not visual_keys:
                continue
    
            for visual_key in visual_keys:
                try:
                    self._render_visual_by_key(
                        visual_key=visual_key,
                        df=df.copy(deep=False),
                        output_dir=output_dir,
                        sub_domain=sub,
                        register_visual=register_visual,
                    )
                except Exception:
                    continue
    
        # -------------------------------------------------
        # FINAL FILTER + SORT
        # -------------------------------------------------
        visuals = [
            v for v in visuals
            if Path(v["path"]).exists() and v["confidence"] >= 0.30
        ]
    
        visuals.sort(
            key=lambda v: v["importance"] * v["confidence"],
            reverse=True,
        )
    
        return visuals


    # -------------------------------------------------
    # VISUAL RENDERER DISPATCH (REAL INTELLIGENCE)
    # -------------------------------------------------
    def _render_visual_by_key(
        self,
        visual_key: str,
        df: pd.DataFrame,
        output_dir: Path,
        sub_domain: str,
        register_visual,
    ):
        """
        Concrete visual implementations.
        Raises Exception if data is insufficient.
        """
    
        c = self.cols
        time_col = self.time_col
    
        # -------------------------------------------------
        # SAFETY: time column must exist & be datetime
        # -------------------------------------------------
        if time_col and time_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df = df.copy()
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    
        # =================================================
        # HOSPITAL VISUALS (CORRECTED & HARDENED)
        # =================================================
        if sub_domain == "hospital":
        
            # -------------------------------------------------
            # 1. Average LOS Trend
            # -------------------------------------------------
            if visual_key == "avg_los_trend":
                if not (c.get("los") and time_col):
                    raise ValueError("LOS or time column missing")
        
                series = (
                    df[[time_col, c["los"]]]
                    .dropna()
                    .set_index(time_col)[c["los"]]
                    .resample("M")
                    .mean()
                )
        
                if series.empty:
                    raise ValueError("No LOS data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                series.plot(ax=ax)
                ax.set_title("Average Length of Stay Trend", fontweight="bold")
                ax.set_ylabel("Days")
                ax.grid(alpha=0.3)
        
                register_visual(
                    fig,
                    "hospital_avg_los_trend.png",
                    "Monthly trend of inpatient length of stay.",
                    0.95,
                    0.90,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. Bed Turnover Velocity
            # -------------------------------------------------
            if visual_key == "bed_turnover":
                bed_col = c.get("bed_id")
                if not bed_col:
                    raise ValueError("Bed ID missing")
        
                turnover = (
                    df[bed_col]
                    .dropna()
                    .value_counts()
                    .clip(upper=100)
                )
        
                if turnover.empty:
                    raise ValueError("No bed turnover data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                turnover.plot(kind="hist", bins=15, ax=ax)
                ax.set_title("Bed Turnover Velocity", fontweight="bold")
                ax.set_xlabel("Patients per Bed")
        
                register_visual(
                    fig,
                    "hospital_bed_velocity.png",
                    "Utilization frequency of physical hospital beds.",
                    0.92,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. Readmission Risk
            # -------------------------------------------------
            if visual_key == "readmission_risk":
                col = c.get("readmitted")
                if not col:
                    raise ValueError("Readmission column missing")
        
                rates = df[col].dropna().value_counts(normalize=True)
                if rates.empty:
                    raise ValueError("No readmission data")
        
                rates.index = rates.index.map({0: "No", 1: "Yes"}).fillna(rates.index)
        
                fig, ax = plt.subplots(figsize=(6, 4))
                rates.plot(kind="bar", ax=ax)
                ax.set_title("Readmission Rate Distribution", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    "hospital_readmission.png",
                    "Distribution of 30-day readmissions.",
                    0.93,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. Discharge Hour Distribution (FIXED)
            # -------------------------------------------------
            discharge_col = c.get("discharge_date")
            if visual_key == "discharge_hour":
                if not discharge_col or discharge_col not in df.columns:
                    raise ValueError("Discharge date missing")
        
                hours = pd.to_datetime(df[discharge_col], errors="coerce").dt.hour.dropna()
                if hours.empty:
                    raise ValueError("No discharge time data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                hours.value_counts().sort_index().plot(kind="bar", ax=ax)
                ax.set_title("Discharge Hour Distribution", fontweight="bold")
                ax.set_xlabel("Hour of Day")
        
                register_visual(
                    fig,
                    "hospital_discharge_hour.png",
                    "Inpatient discharge timing pattern.",
                    0.85,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. Acuity vs Staffing (EXPLICIT PROXY)
            # -------------------------------------------------
            if visual_key == "acuity_vs_staffing":
                if not (c.get("los") and c.get("cost")):
                    raise ValueError("LOS or cost missing")
        
                tmp = (
                    df[[c["los"], c["cost"]]]
                    .dropna()
                    .clip(upper={c["los"]: 60, c["cost"]: df[c["cost"]].quantile(0.95)})
                )
        
                if tmp.empty:
                    raise ValueError("No acuity-cost data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(tmp[c["los"]], tmp[c["cost"]], alpha=0.35)
                ax.set_xlabel("Length of Stay (Acuity Proxy)")
                ax.set_ylabel("Cost (Staffing Intensity Proxy)")
                ax.set_title("Acuity vs Staffing Intensity (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "hospital_acuity_staffing.png",
                    "Relationship between patient acuity and staffing intensity (proxy).",
                    0.88,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. ED Boarding Time
            # -------------------------------------------------
            if visual_key == "ed_boarding":
                if not (c.get("duration") and time_col):
                    raise ValueError("Duration or time missing")
        
                series = (
                    df[[time_col, c["duration"]]]
                    .dropna()
                    .set_index(time_col)[c["duration"]]
                    .resample("M")
                    .mean()
                )
        
                if series.empty:
                    raise ValueError("No ED boarding data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                series.plot(ax=ax)
                ax.set_title("ED Boarding Time Trend", fontweight="bold")
                ax.set_ylabel("Hours")
        
                register_visual(
                    fig,
                    "hospital_ed_boarding.png",
                    "Average emergency department boarding time.",
                    0.92,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. Mortality Trend (EXPLICIT PROXY)
            # -------------------------------------------------
            if visual_key == "mortality_trend":
                flag_col = c.get("flag")
                if not (flag_col and time_col):
                    raise ValueError("Mortality proxy or time missing")
        
                rate = (
                    df[[time_col, flag_col]]
                    .dropna()
                    .set_index(time_col)[flag_col]
                    .resample("M")
                    .mean()
                )
        
                if rate.empty:
                    raise ValueError("No mortality data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                rate.plot(ax=ax, marker="o")
                ax.set_title("In-Hospital Mortality Proxy Trend", fontweight="bold")
        
                register_visual(
                    fig,
                    "hospital_mortality_trend.png",
                    "Observed in-hospital mortality proxy trend over time.",
                    0.90,
                    0.80,
                    sub_domain,
                )
                return

    
        # =================================================
        # CLINIC / AMBULATORY VISUALS (CORRECTED & SAFE)
        # =================================================
        if sub_domain == "clinic":
        
            # -------------------------------------------------
            # 1. NO-SHOW RATE BY DAY (PROXY)
            # -------------------------------------------------
            if visual_key == "no_show_by_day":
                if not (c.get("readmitted") and time_col):
                    raise ValueError("No-show proxy or time missing")
        
                tmp = df[[time_col, c["readmitted"]]].dropna()
                if tmp.empty:
                    raise ValueError("No no-show data")
        
                tmp["_dow"] = tmp[time_col].dt.day_name()
                rate = tmp.groupby("_dow")[c["readmitted"]].mean()
        
                day_order = [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ]
                rate = rate.reindex(day_order)
        
                fig, ax = plt.subplots(figsize=(6, 4))
                rate.plot(kind="bar", ax=ax)
                ax.set_title("Appointment No-Show Rate by Day (Proxy)", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    "clinic_no_show_by_day.png",
                    "Appointment no-show rates across the week (proxy indicator).",
                    0.88,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. WAIT TIME TRAJECTORY
            # -------------------------------------------------
            if visual_key == "wait_time_split":
                if not (c.get("duration") and time_col):
                    raise ValueError("Duration or time missing")
        
                series = (
                    df[[time_col, c["duration"]]]
                    .dropna()
                    .assign(_dur=lambda x: x[c["duration"]].clip(upper=240))
                    .set_index(time_col)["_dur"]
                    .resample("D")
                    .mean()
                )
        
                if series.empty:
                    raise ValueError("No wait-time data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                series.plot(ax=ax)
                ax.set_title("Average Patient Wait Time Trajectory", fontweight="bold")
                ax.set_ylabel("Minutes")
        
                register_visual(
                    fig,
                    "clinic_wait_time_trend.png",
                    "Trend of patient wait times from check-in to provider.",
                    0.90,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. APPOINTMENT LAG DISTRIBUTION (PROXY)
            # -------------------------------------------------
            if visual_key == "appointment_lag":
                if not (c.get("date") and c.get("pid")):
                    raise ValueError("Date or patient ID missing")
        
                tmp = df[[c["pid"], c["date"]]].dropna().copy()
                tmp[c["date"]] = pd.to_datetime(tmp[c["date"]], errors="coerce")
        
                lag = (
                    tmp.sort_values(c["date"])
                    .groupby(c["pid"])[c["date"]]
                    .diff()
                    .dt.days
                    .clip(lower=0, upper=60)
                    .dropna()
                )
        
                if lag.empty:
                    raise ValueError("No appointment lag data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                lag.hist(ax=ax, bins=20)
                ax.set_title("Appointment Lag Distribution (Proxy)", fontweight="bold")
                ax.set_xlabel("Days")
        
                register_visual(
                    fig,
                    "clinic_appointment_lag.png",
                    "Days between booking and clinic visit (proxy indicator).",
                    0.85,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. PROVIDER UTILIZATION (WORKLOAD)
            # -------------------------------------------------
            if visual_key == "provider_utilization":
                if not c.get("doctor"):
                    raise ValueError("Doctor column missing")
        
                counts = (
                    df[c["doctor"]]
                    .astype(str)
                    .value_counts()
                    .head(10)
                )
        
                if counts.empty:
                    raise ValueError("No provider data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                counts.plot(kind="bar", ax=ax)
                ax.set_title("Provider Utilization (Top 10)", fontweight="bold")
                ax.set_ylabel("Visits")
        
                register_visual(
                    fig,
                    "clinic_provider_utilization.png",
                    "Comparison of provider workload distribution.",
                    0.88,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PATIENT DEMOGRAPHIC REACH (LOCATION PROXY)
            # -------------------------------------------------
            if visual_key == "demographic_reach":
                if not c.get("facility"):
                    raise ValueError("Facility column missing")
        
                counts = df[c["facility"]].astype(str).value_counts()
                if counts.empty:
                    raise ValueError("No geographic data")
        
                top = counts.head(7)
                if len(counts) > 7:
                    top = top.append(
                        pd.Series({"Other": counts.iloc[7:].sum()})
                    )
        
                fig, ax = plt.subplots(figsize=(6, 6))
                top.plot(kind="pie", ax=ax, autopct="%1.0f%%")
                ax.set_ylabel("")
                ax.set_title("Patient Reach by Location (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "clinic_demographic_reach.png",
                    "Distribution of clinic visits by service location (proxy).",
                    0.82,
                    0.78,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. REFERRAL FUNNEL (ESTIMATED)
            # -------------------------------------------------
            if visual_key == "referral_funnel":
                total = len(df)
        
                stages = {
                    "Referrals": total,
                    "Scheduled (Est.)": int(total * 0.75),
                    "Completed (Est.)": int(total * 0.65),
                }
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(stages.keys(), stages.values())
                ax.set_title("Referral Conversion Funnel (Estimated)", fontweight="bold")
        
                register_visual(
                    fig,
                    "clinic_referral_funnel.png",
                    "Estimated referral flow from intake to completed visits.",
                    0.80,
                    0.70,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. TELEHEALTH MIX (CHANNEL PROXY)
            # -------------------------------------------------
            if visual_key == "telehealth_mix":
                if not c.get("facility"):
                    raise ValueError("Facility column missing")
        
                series = (
                    df[c["facility"]]
                    .astype(str)
                    .str.lower()
                )
        
                mix = series.apply(
                    lambda x: "Telehealth" if "tele" in x else "In-Person"
                ).value_counts()
        
                if mix.empty:
                    raise ValueError("No telehealth data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                mix.plot(kind="bar", ax=ax)
                ax.set_title("Telehealth vs In-Person Visits (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "clinic_telehealth_mix.png",
                    "Service delivery mix across visit types (proxy indicator).",
                    0.82,
                    0.75,
                    sub_domain,
                )
                return

        # =================================================
        # DIAGNOSTICS (LABS / RADIOLOGY) VISUALS (HARDENED)
        # =================================================
        if sub_domain == "diagnostics":
        
            # Ensure datetime safety once
            if time_col:
                df = df.copy()
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
            # -------------------------------------------------
            # 1. TURNAROUND TIME PERCENTILES (SMOOTHED)
            # -------------------------------------------------
            if visual_key == "tat_percentiles":
                if not (c.get("duration") and time_col):
                    raise ValueError("Duration or time column missing")
        
                tmp = (
                    df[[time_col, c["duration"]]]
                    .dropna()
                    .assign(_dur=lambda x: x[c["duration"]].clip(upper=720))
                )
                if tmp.empty:
                    raise ValueError("No TAT data")
        
                grouped = tmp.set_index(time_col)["_dur"].resample("D")
                p50 = grouped.quantile(0.50).rolling(3, min_periods=1).mean()
                p90 = grouped.quantile(0.90).rolling(3, min_periods=1).mean()
                p95 = grouped.quantile(0.95).rolling(3, min_periods=1).mean()
        
                if p50.dropna().empty:
                    raise ValueError("Insufficient TAT distribution")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                p50.plot(ax=ax, label="50th %ile")
                p90.plot(ax=ax, label="90th %ile")
                p95.plot(ax=ax, label="95th %ile")
                ax.legend()
                ax.set_title("Diagnostic Turnaround Time Percentiles", fontweight="bold")
                ax.set_ylabel("Minutes")
        
                register_visual(
                    fig,
                    "diagnostics_tat_percentiles.png",
                    "Smoothed diagnostic turnaround time percentiles over time.",
                    0.92,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. CRITICAL ALERT NOTIFICATION TIME (PROXY)
            # -------------------------------------------------
            if visual_key == "critical_alert_time":
                if not (c.get("duration") and c.get("flag")):
                    raise ValueError("Flag or duration missing")
        
                flag = pd.to_numeric(df[c["flag"]], errors="coerce")
                dur = pd.to_numeric(df[c["duration"]], errors="coerce")
        
                critical = dur[flag == 1].clip(upper=180).dropna()
                if critical.empty:
                    raise ValueError("No critical alert data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                critical.hist(ax=ax, bins=20)
                ax.set_title("Critical Result Notification Time (Proxy)", fontweight="bold")
                ax.set_xlabel("Minutes")
        
                register_visual(
                    fig,
                    "diagnostics_critical_alert_time.png",
                    "Speed of notifying critical diagnostic results (proxy).",
                    0.90,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. SPECIMEN REJECTION SIGNALS (PROXY)
            # -------------------------------------------------
            if visual_key == "specimen_rejection":
                if not c.get("flag"):
                    raise ValueError("Flag column missing")
        
                reasons = (
                    pd.to_numeric(df[c["flag"]], errors="coerce")
                    .value_counts()
                    .head(5)
                )
                if reasons.empty:
                    raise ValueError("No specimen rejection signals")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                reasons.plot(kind="bar", ax=ax)
                ax.set_title("Specimen Rejection Signals (Proxy)", fontweight="bold")
                ax.set_ylabel("Count")
        
                register_visual(
                    fig,
                    "diagnostics_specimen_rejection.png",
                    "Observed specimen rejection indicators (proxy signals).",
                    0.88,
                    0.82,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. RELATIVE DEVICE UTILIZATION (PROXY)
            # -------------------------------------------------
            if visual_key == "device_downtime":
                if not (c.get("facility") and time_col):
                    raise ValueError("Facility or time missing")
        
                tmp = df[[c["facility"], time_col]].dropna()
                if tmp.empty:
                    raise ValueError("No device utilization data")
        
                usage = (
                    tmp[c["facility"]]
                    .astype(str)
                    .value_counts()
                    .head(10)
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                usage.plot(kind="bar", ax=ax)
                ax.set_title("Relative Device Utilization (Proxy)", fontweight="bold")
                ax.set_ylabel("Observation Count")
        
                register_visual(
                    fig,
                    "diagnostics_device_utilization.png",
                    "Relative diagnostic equipment utilization by site (proxy).",
                    0.82,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PEAK ORDER LOAD HEATMAP (CAPPED)
            # -------------------------------------------------
            if visual_key == "order_heatmap":
                if not time_col:
                    raise ValueError("Time column missing")
        
                tmp = df[[time_col]].dropna()
                if tmp.empty:
                    raise ValueError("No order timestamps")
        
                tmp["_hour"] = tmp[time_col].dt.hour.clip(0, 23)
                tmp["_day"] = tmp[time_col].dt.day_name()
        
                heat = pd.crosstab(tmp["_day"], tmp["_hour"])
                if heat.empty:
                    raise ValueError("Empty order heatmap")
        
                day_order = [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ]
                heat = heat.reindex(day_order).fillna(0)
        
                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.imshow(heat, aspect="auto", cmap="Blues")
                ax.set_title("Peak Diagnostic Order Load", fontweight="bold")
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Day of Week")
                plt.colorbar(im, ax=ax)
        
                register_visual(
                    fig,
                    "diagnostics_order_heatmap.png",
                    "Hourly diagnostic order intensity by day.",
                    0.90,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. REPEAT SCAN INCIDENCE (VOLUME GUARDED)
            # -------------------------------------------------
            if visual_key == "repeat_scan":
                if not c.get("encounter"):
                    raise ValueError("Encounter column missing")
        
                counts = df[c["encounter"]].value_counts()
                if counts.empty or len(counts) < 30:
                    raise ValueError("Insufficient scan volume")
        
                repeat_rate = (counts > 1).mean()
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Repeat Scan Rate"], [repeat_rate])
                ax.set_ylim(0, 1)
                ax.set_title("Repeat Diagnostic Incidence", fontweight="bold")
        
                register_visual(
                    fig,
                    "diagnostics_repeat_scan.png",
                    "Rate of repeated diagnostic tests indicating potential waste.",
                    0.86,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. PROVIDER ORDERING VARIANCE
            # -------------------------------------------------
            if visual_key == "ordering_variance":
                if not c.get("doctor"):
                    raise ValueError("Doctor column missing")
        
                orders = (
                    df[c["doctor"]]
                    .astype(str)
                    .value_counts()
                    .head(10)
                )
        
                if orders.empty:
                    raise ValueError("No provider ordering data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                orders.plot(kind="bar", ax=ax)
                ax.set_title("Provider Ordering Variance", fontweight="bold")
                ax.set_ylabel("Orders")
        
                register_visual(
                    fig,
                    "diagnostics_ordering_variance.png",
                    "Variation in diagnostic ordering behavior across providers.",
                    0.87,
                    0.82,
                    sub_domain,
                )
                return
 
        # =================================================
        # PHARMACY VISUALS (STRICT & HARDENED)
        # =================================================
        if sub_domain == "pharmacy":
        
            # -------------------------------------------------
            # HARD PHARMACY DATA GATE
            # -------------------------------------------------
            fill_col = c.get("fill_date")
            if not fill_col or fill_col not in df.columns:
                raise ValueError("Pharmacy data missing fill date")
        
            df = df.copy()
            df[fill_col] = pd.to_datetime(df[fill_col], errors="coerce")
        
            # -------------------------------------------------
            # 1. MEDICATION SPEND VELOCITY (CUMULATIVE)
            # -------------------------------------------------
            if visual_key == "spend_velocity":
                if not c.get("cost"):
                    raise ValueError("Cost column missing")
        
                tmp = df[[fill_col, c["cost"]]].dropna()
                if tmp.empty:
                    raise ValueError("No spend data")
        
                spend = (
                    tmp.set_index(fill_col)[c["cost"]]
                    .resample("M")
                    .sum()
                    .cumsum()
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                spend.plot(ax=ax)
                ax.set_title("Medication Spend Velocity", fontweight="bold")
                ax.set_ylabel("Cumulative Spend")
        
                ax.yaxis.set_major_formatter(
                    lambda x, _: f"{x/1_000_000:.1f}M"
                    if x >= 1_000_000 else f"{x/1_000:.0f}K"
                )
        
                register_visual(
                    fig,
                    "pharmacy_spend_velocity.png",
                    "Cumulative medication expenditure over time.",
                    0.90,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. REFILL ADHERENCE GAP (TRUE GAP, PROXY)
            # -------------------------------------------------
            if visual_key == "refill_gap":
                if not (c.get("supply")):
                    raise ValueError("Supply column missing")
        
                fill = df[fill_col]
                supply = pd.to_numeric(df[c["supply"]], errors="coerce")
        
                tmp = pd.DataFrame({"fill": fill, "supply": supply}).dropna()
                if len(tmp) < 2:
                    raise ValueError("Insufficient refill data")
        
                tmp = tmp.sort_values("fill")
                expected = tmp["fill"] + pd.to_timedelta(tmp["supply"], unit="D")
                actual_next = tmp["fill"].shift(-1)
        
                gap = (actual_next - expected).dt.days.dropna()
                gap = gap.clip(lower=-30, upper=60)
        
                if gap.empty:
                    raise ValueError("No refill adherence gap data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                gap.hist(ax=ax, bins=20)
                ax.set_title("Refill Adherence Gap (Proxy)", fontweight="bold")
                ax.set_xlabel("Days Late / Early")
        
                register_visual(
                    fig,
                    "pharmacy_refill_gap.png",
                    "Delay between expected and actual prescription refills (proxy).",
                    0.88,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. THERAPEUTIC SPEND (FACILITY PROXY)
            # -------------------------------------------------
            if visual_key == "therapeutic_spend":
                if not (c.get("facility") and c.get("cost")):
                    raise ValueError("Facility or cost missing")
        
                spend = (
                    df[[c["facility"], c["cost"]]]
                    .dropna()
                    .assign(_grp=lambda x: x[c["facility"]].astype(str))
                    .groupby("_grp")[c["cost"]]
                    .sum()
                    .nlargest(6)
                )
        
                if spend.empty:
                    raise ValueError("No spend data")
        
                fig, ax = plt.subplots(figsize=(6, 6))
                spend.plot(kind="pie", autopct="%1.0f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Medication Spend by Group (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "pharmacy_therapeutic_spend.png",
                    "Medication spend distribution by proxy grouping (facility-based).",
                    0.82,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. GENERIC SUBSTITUTION RATE (PROXY)
            # -------------------------------------------------
            if visual_key == "generic_rate":
                if not c.get("facility"):
                    raise ValueError("Facility column missing")
        
                series = (
                    df[c["facility"]]
                    .astype(str)
                    .str.lower()
                    .str.contains("generic", na=False)
                )
        
                rate = series.mean()
                if pd.isna(rate):
                    raise ValueError("Invalid generic proxy")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Generic Substitution Rate"], [rate])
                ax.set_ylim(0, 1)
                ax.set_title("Generic Substitution Rate (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "pharmacy_generic_rate.png",
                    "Share of prescriptions filled with generic alternatives (proxy).",
                    0.80,
                    0.72,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PRESCRIBING COST VARIANCE
            # -------------------------------------------------
            if visual_key == "prescribing_variance":
                if not (c.get("doctor") and c.get("cost")):
                    raise ValueError("Doctor or cost missing")
        
                variance = (
                    df[[c["doctor"], c["cost"]]]
                    .dropna()
                    .assign(_doc=lambda x: x[c["doctor"]].astype(str))
                    .groupby("_doc")[c["cost"]]
                    .mean()
                    .nlargest(10)
                )
        
                if variance.empty:
                    raise ValueError("No prescribing variance data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                variance.plot(kind="bar", ax=ax)
                ax.set_title("Prescribing Cost Variance", fontweight="bold")
                ax.set_ylabel("Average Cost")
        
                register_visual(
                    fig,
                    "pharmacy_prescribing_variance.png",
                    "Variation in average prescribing cost across providers.",
                    0.86,
                    0.82,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. INVENTORY TURN RATIO (PROXY)
            # -------------------------------------------------
            if visual_key == "inventory_turn":
                if not (c.get("supply") and c.get("cost")):
                    raise ValueError("Supply or cost missing")
        
                supply = pd.to_numeric(df[c["supply"]], errors="coerce")
                cost = pd.to_numeric(df[c["cost"]], errors="coerce")
        
                avg_supply = supply.mean()
                total_cost = cost.sum()
        
                if not avg_supply or avg_supply <= 0:
                    raise ValueError("Invalid inventory supply")
        
                turn = total_cost / avg_supply
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Inventory Turn Ratio"], [turn])
                ax.set_title("Inventory Turn Ratio (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "pharmacy_inventory_turn.png",
                    "Efficiency of medication inventory turnover (proxy).",
                    0.78,
                    0.70,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. DRUG SAFETY INTERVENTIONS (PROXY)
            # -------------------------------------------------
            if visual_key == "drug_alerts":
                if not c.get("flag"):
                    raise ValueError("Flag column missing")
        
                alerts = (
                    pd.to_numeric(df[c["flag"]], errors="coerce")
                    .value_counts(normalize=True)
                    .sort_index()
                )
        
                if alerts.empty:
                    raise ValueError("No safety alert data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                alerts.plot(kind="bar", ax=ax)
                ax.set_title("Pharmacist Safety Interventions (Proxy)", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    "pharmacy_drug_alerts.png",
                    "Frequency of pharmacist safety interventions (proxy indicator).",
                    0.82,
                    0.75,
                    sub_domain,
                )
                return

        # =================================================
        # PUBLIC HEALTH / POPULATION HEALTH VISUALS (HARDENED)
        # =================================================
        if sub_domain == "public_health":
        
            # -------------------------------------------------
            # HARD PUBLIC HEALTH DATA GATE
            # -------------------------------------------------
            if not c.get("population"):
                raise ValueError("Population data required for public health analysis")
        
            # Ensure datetime safety once
            if time_col:
                df = df.copy()
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
            # -------------------------------------------------
            # 1. DISEASE INCIDENCE RATE (PER 100K, PROXY)
            # -------------------------------------------------
            if visual_key == "incidence_geo":
                if not c.get("flag"):
                    raise ValueError("Outcome flag missing")
        
                pop = pd.to_numeric(df[c["population"]], errors="coerce").dropna()
                cases = pd.to_numeric(df[c["flag"]], errors="coerce").fillna(0)
        
                if pop.empty or cases.sum() == 0:
                    raise ValueError("Insufficient incidence data")
        
                pop_denom = pop.median()
                if pop_denom <= 0:
                    raise ValueError("Invalid population denominator")
        
                incidence = (cases.sum() / pop_denom) * 100_000
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Incidence per 100k"], [incidence])
                ax.set_title("Disease Incidence Rate (Proxy)", fontweight="bold")
                ax.set_ylabel("Cases per 100,000")
        
                register_visual(
                    fig,
                    "public_health_incidence_rate.png",
                    "Observed disease incidence per 100,000 population (proxy denominator).",
                    0.90,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. COHORT GROWTH TRAJECTORY (SMOOTHED)
            # -------------------------------------------------
            if visual_key == "cohort_growth":
                if not (time_col and c.get("flag")):
                    raise ValueError("Time or outcome flag missing")
        
                tmp = df[[time_col, c["flag"]]].dropna()
                if tmp.empty:
                    raise ValueError("No cohort data")
        
                cohort = (
                    (tmp[c["flag"]] == 1)
                    .astype(int)
                    .groupby(tmp[time_col].dt.to_period("M"))
                    .sum()
                    .cumsum()
                    .rolling(2, min_periods=1)
                    .mean()
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                cohort.plot(ax=ax)
                ax.set_title("Cohort Growth Trajectory (Smoothed)", fontweight="bold")
                ax.set_ylabel("Cumulative Cases")
        
                register_visual(
                    fig,
                    "public_health_cohort_growth.png",
                    "Smoothed cumulative growth of observed population health cohort.",
                    0.88,
                    0.82,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. DEMOGRAPHIC SEGMENT PREVALENCE (PROXY)
            # -------------------------------------------------
            if visual_key == "prevalence_age":
                if not (c.get("pid") and c.get("flag")):
                    raise ValueError("Patient ID or outcome flag missing")
        
                pid_len = df[c["pid"]].astype(str).str.len()
                flag = pd.to_numeric(df[c["flag"]], errors="coerce")
        
                buckets = pd.cut(
                    pid_len,
                    bins=[0, 6, 8, 10, 99],
                    labels=["Segment A", "Segment B", "Segment C", "Segment D"],
                )
        
                prevalence = (
                    flag.groupby(buckets)
                    .mean()
                    .dropna()
                )
        
                if prevalence.empty:
                    raise ValueError("Prevalence calculation failed")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                prevalence.plot(kind="bar", ax=ax)
                ax.set_title("Outcome Prevalence by Demographic Segment (Proxy)", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    "public_health_demographic_prevalence.png",
                    "Outcome prevalence by inferred demographic segments (proxy).",
                    0.82,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. SERVICE ACCESS INDICATOR (PROXY)
            # -------------------------------------------------
            if visual_key == "access_gap":
                if not c.get("facility"):
                    raise ValueError("Facility column missing")
        
                pop = pd.to_numeric(df[c["population"]], errors="coerce").dropna()
                facilities = df[c["facility"]].astype(str).dropna()
        
                if pop.empty or facilities.empty:
                    raise ValueError("No access data")
        
                service_points = facilities.nunique()
                pop_denom = pop.median()
                if pop_denom <= 0:
                    raise ValueError("Invalid population denominator")
        
                ratio = (service_points / pop_denom) * 1_000
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Service Points per 1k"], [ratio])
                ax.set_title("Healthcare Access Indicator (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "public_health_access_gap.png",
                    "Availability of healthcare service points per 1,000 residents (proxy).",
                    0.85,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PROGRAM EFFICACY TREND (SMOOTHED, PROXY)
            # -------------------------------------------------
            if visual_key == "program_effect":
                if not (time_col and c.get("flag")):
                    raise ValueError("Time or outcome flag missing")
        
                tmp = df[[time_col, c["flag"]]].dropna()
                if tmp.empty:
                    raise ValueError("No program outcome data")
        
                trend = (
                    (tmp[c["flag"]] == 1)
                    .astype(int)
                    .groupby(tmp[time_col].dt.to_period("M"))
                    .mean()
                    .rolling(3, min_periods=1)
                    .mean()
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                trend.plot(ax=ax)
                ax.set_title("Program Outcome Trend (Proxy)", fontweight="bold")
                ax.set_ylabel("Outcome Rate")
        
                register_visual(
                    fig,
                    "public_health_program_effect.png",
                    "Smoothed population outcome trends following interventions (proxy).",
                    0.86,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. SOCIAL DETERMINANTS RISK OVERLAY (PROXY)
            # -------------------------------------------------
            if visual_key == "sdoh_overlay":
                if not (c.get("facility") and c.get("flag")):
                    raise ValueError("Facility or outcome flag missing")
        
                sdoh = (
                    df[[c["facility"], c["flag"]]]
                    .dropna()
                    .assign(_area=lambda x: x[c["facility"]].astype(str))
                    .groupby("_area")[c["flag"]]
                    .mean()
                    .nlargest(8)
                )
        
                if sdoh.empty:
                    raise ValueError("No SDOH data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                sdoh.plot(kind="bar", ax=ax)
                ax.set_title("Social Determinants Risk Overlay (Proxy)", fontweight="bold")
                ax.set_ylabel("Outcome Rate")
        
                register_visual(
                    fig,
                    "public_health_sdoh_overlay.png",
                    "Health outcome variation across socioeconomic regions (proxy).",
                    0.80,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. IMMUNIZATION / SCREENING COVERAGE (PROXY)
            # -------------------------------------------------
            if visual_key == "immunization_rate":
                if not c.get("flag"):
                    raise ValueError("Outcome flag missing")
        
                rate = pd.to_numeric(df[c["flag"]], errors="coerce").mean()
                if pd.isna(rate):
                    raise ValueError("Invalid coverage rate")
        
                rate = min(max(rate, 0.0), 1.0)
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Coverage Rate"], [rate])
                ax.set_ylim(0, 1)
                ax.set_title("Immunization / Screening Coverage (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    "public_health_immunization_rate.png",
                    "Population coverage of immunization or screening programs (proxy).",
                    0.84,
                    0.78,
                    sub_domain,
                )
                return

    # -------------------------------------------------
    # INSIGHTS ENGINE (EVIDENCE-LOCKED, EXECUTIVE SAFE)
    # -------------------------------------------------
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *_,
    ) -> List[Dict[str, Any]]:
    
        insights: List[Dict[str, Any]] = []
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # CONFIDENCE CALCULATION (HONEST)
        # -------------------------------------------------
        def insight_conf(kpi_conf: float, sub_score: float) -> float:
            base = min(kpi_conf, 0.85)
            return round(min(0.92, base * (0.6 + 0.4 * sub_score)), 2)
    
        # -------------------------------------------------
        # CROSS-DOMAIN INSIGHT (STRICTLY GUARDED)
        # -------------------------------------------------
        if (
            active_subs.get("hospital", 0) >= 0.5
            and active_subs.get("diagnostics", 0) >= 0.5
        ):
            los = self.get_kpi(kpis, "hospital", "avg_los")
            tat = self.get_kpi(kpis, "diagnostics", "avg_tat")
    
            if (
                isinstance(los, (int, float))
                and isinstance(tat, (int, float))
                and tat > 120
            ):
                conf_val = min(
                    self.get_kpi_confidence(kpis, "hospital", "avg_los"),
                    self.get_kpi_confidence(kpis, "diagnostics", "avg_tat"),
                )
    
                insights.append({
                    "sub_domain": "cross_domain",
                    "level": "RISK",
                    "title": "Diagnostic Turnaround Delays Linked to Inpatient Stay",
                    "so_what": (
                        f"Long diagnostic turnaround times "
                        f"({tat:.0f} minutes) are likely contributing to "
                        f"extended inpatient stays (avg LOS {los:.1f} days). "
                        "This association indicates a cross-functional throughput constraint."
                    ),
                    "confidence": insight_conf(conf_val, min(active_subs["hospital"], active_subs["diagnostics"])),
                })
    
        # -------------------------------------------------
        # SUB-DOMAIN INSIGHTS (MAX 5 EACH, NO FILLERS)
        # -------------------------------------------------
        for sub, score in active_subs.items():
            sub_insights: List[Dict[str, Any]] = []
    
            # ===============================
            # STRENGTHS
            # ===============================
            if sub == HealthcareSubDomain.HOSPITAL.value:
                avg_los = self.get_kpi(kpis, sub, "avg_los")
                if isinstance(avg_los, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Inpatient Throughput Visibility",
                        "so_what": (
                            f"Length of stay is consistently observable "
                            f"(average {avg_los:.1f} days), enabling governance "
                            "of inpatient throughput and discharge planning."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_los"),
                            score,
                        ),
                    })
    
            if sub == HealthcareSubDomain.CLINIC.value:
                wait = self.get_kpi(kpis, sub, "avg_wait_time")
                if isinstance(wait, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Clinic Access Transparency",
                        "so_what": (
                            f"Patient wait times are measurable "
                            f"(average {wait:.0f} minutes), supporting "
                            "access and scheduling optimization."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_wait_time"),
                            score,
                        ),
                    })
    
            if sub == HealthcareSubDomain.DIAGNOSTICS.value:
                tat = self.get_kpi(kpis, sub, "avg_tat")
                if isinstance(tat, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Diagnostic Turnaround Observability",
                        "so_what": (
                            f"Turnaround times are tracked "
                            f"(average {tat:.0f} minutes), enabling SLA monitoring "
                            "and operational accountability."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_tat"),
                            score,
                        ),
                    })
    
            if sub == HealthcareSubDomain.PHARMACY.value:
                cost = self.get_kpi(kpis, sub, "cost_per_rx")
                if isinstance(cost, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Prescription Cost Visibility",
                        "so_what": (
                            f"Average prescription cost is observable "
                            f"(₹{cost:.0f}), supporting pharmacy spend oversight."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "cost_per_rx"),
                            score,
                        ),
                    })
    
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                inc = self.get_kpi(kpis, sub, "incidence_per_100k")
                if isinstance(inc, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Population Health Signal Coverage",
                        "so_what": (
                            "Population-level indicators enable monitoring "
                            "of disease burden and prevention effectiveness."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "incidence_per_100k"),
                            score,
                        ),
                    })
    
            # ===============================
            # WARNINGS / RISKS
            # ===============================
            if sub == HealthcareSubDomain.CLINIC.value:
                no_show = self.get_kpi(kpis, sub, "no_show_rate")
                if isinstance(no_show, (int, float)) and no_show >= 0.10:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Elevated Appointment No-Show Risk",
                        "so_what": (
                            f"A no-show rate of {no_show:.1%} may reduce "
                            "clinic throughput and revenue efficiency."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "no_show_rate"),
                            score,
                        ),
                    })
    
            if sub == HealthcareSubDomain.HOSPITAL.value:
                long_stay = self.get_kpi(kpis, sub, "long_stay_rate")
                if isinstance(long_stay, (int, float)) and long_stay >= 0.25:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Discharge Throughput Constraint",
                        "so_what": (
                            f"{long_stay:.1%} of patients exceed expected "
                            "length-of-stay thresholds, indicating discharge bottlenecks."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "long_stay_rate"),
                            score,
                        ),
                    })
    
            if sub == HealthcareSubDomain.DIAGNOSTICS.value:
                tat = self.get_kpi(kpis, sub, "avg_tat")
                if isinstance(tat, (int, float)) and tat > 120:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Diagnostic Turnaround Pressure",
                        "so_what": (
                            f"Average turnaround time of {tat:.0f} minutes "
                            "may delay clinical decision-making."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "avg_tat"),
                            score,
                        ),
                    })
    
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                inc = self.get_kpi(kpis, sub, "incidence_per_100k")
                if isinstance(inc, (int, float)) and inc > 300:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Elevated Disease Incidence",
                        "so_what": (
                            f"Incidence rate of {inc:.0f} per 100k exceeds "
                            "expected thresholds, indicating prevention gaps."
                        ),
                        "confidence": insight_conf(
                            self.get_kpi_confidence(kpis, sub, "incidence_per_100k"),
                            score,
                        ),
                    })
    
            # -------------------------------------------------
            # CAP INSIGHTS PER SUB-DOMAIN (EXECUTIVE SAFE)
            # -------------------------------------------------
            insights.extend(sub_insights[:5])
    
        return insights

    # --------------------------------
    # RECOMMENDATIONS ENGINE (INSIGHT-BOUND, EXECUTIVE SAFE)
    # --------------------------------
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *_,
    ) -> List[Dict[str, Any]]:
    
        recommendations: List[Dict[str, Any]] = []
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # INDEX INSIGHTS BY SUB-DOMAIN & LEVEL
        # -------------------------------------------------
        insights_by_sub: Dict[str, List[Dict[str, Any]]] = {}
        for ins in insights:
            if isinstance(ins, dict):
                insights_by_sub.setdefault(ins.get("sub_domain"), []).append(ins)
    
        # -------------------------------------------------
        # CONFIDENCE BINDING (HONEST)
        # -------------------------------------------------
        def rec_conf(insight_conf: float, sub_score: float) -> float:
            return round(min(0.90, insight_conf * (0.7 + 0.3 * sub_score)), 2)
    
        # -------------------------------------------------
        # SUB-DOMAIN RECOMMENDATIONS
        # -------------------------------------------------
        for sub, score in active_subs.items():
    
            sub_insights = insights_by_sub.get(sub, [])
            if not sub_insights:
                continue  # 🔒 no insight → no recommendation
    
            for ins in sub_insights:
    
                level = ins.get("level")
                ins_conf = float(ins.get("confidence", 0.6))
    
                # ===============================
                # HOSPITAL
                # ===============================
                if sub == HealthcareSubDomain.HOSPITAL.value and level == "RISK":
                    los = self.get_kpi(kpis, sub, "avg_los")
    
                    recommendations.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Establish centralized discharge coordination and daily LOS review",
                        "owner": "Hospital Operations",
                        "timeline": "30–60 days",
                        "goal": "Reduce prolonged inpatient stays and free bed capacity",
                        "current_value": round(los, 2) if isinstance(los, (int, float)) else None,
                        "expected_impact": "10–15% improvement in bed availability",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
                # ===============================
                # CLINIC
                # ===============================
                if sub == HealthcareSubDomain.CLINIC.value and level in {"WARNING", "RISK"}:
                    recommendations.append({
                        "sub_domain": sub,
                        "priority": "MEDIUM",
                        "action": "Deploy predictive no-show mitigation and automated reminders",
                        "owner": "Ambulatory Operations",
                        "timeline": "30–90 days",
                        "goal": "Stabilize provider utilization and improve access",
                        "expected_impact": "5–10% increase in completed visits",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
                # ===============================
                # DIAGNOSTICS
                # ===============================
                if sub == HealthcareSubDomain.DIAGNOSTICS.value and level == "RISK":
                    recommendations.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Implement STAT workflow escalation and turnaround time governance",
                        "owner": "Diagnostics Leadership",
                        "timeline": "30–60 days",
                        "goal": "Reduce diagnostic turnaround delays",
                        "expected_impact": "Faster clinical decision-making",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
                # ===============================
                # PHARMACY
                # ===============================
                if sub == HealthcareSubDomain.PHARMACY.value and level in {"WARNING", "RISK"}:
                    recommendations.append({
                        "sub_domain": sub,
                        "priority": "MEDIUM",
                        "action": "Strengthen pharmacist safety review and alert resolution workflows",
                        "owner": "Pharmacy Operations",
                        "timeline": "30–60 days",
                        "goal": "Reduce medication safety risk",
                        "expected_impact": "Lower intervention and error rates",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
                # ===============================
                # PUBLIC HEALTH
                # ===============================
                if sub == HealthcareSubDomain.PUBLIC_HEALTH.value and level == "RISK":
                    recommendations.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Target high-incidence regions with focused prevention programs",
                        "owner": "Public Health Authority",
                        "timeline": "90–180 days",
                        "goal": "Reduce disease incidence and population risk",
                        "expected_impact": "Lower future healthcare burden",
                        "confidence": rec_conf(ins_conf, score),
                    })
    
            # -------------------------------
            # DEDUPLICATE PER SUB-DOMAIN
            # -------------------------------
            seen = set()
            deduped = []
            for r in recommendations:
                key = (r.get("sub_domain"), r.get("action"))
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
    
            recommendations = deduped
    
        # -------------------------------------------------
        # EXECUTIVE SORTING & HARD CAP
        # -------------------------------------------------
        priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(
            key=lambda r: (
                priority_rank.get(r.get("priority"), 3),
                -r.get("confidence", 0),
            )
        )
    
        # HARD CAP: MAX 8 TOTAL (BOARD SAFE)
        return recommendations[:8]

# =====================================================
# REGISTRATION
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    def detect(self, df: pd.DataFrame):

        # -------------------------------------------------
        # STRICT COLUMN PRESENCE CHECKS (DETECTOR ≠ DOMAIN)
        # -------------------------------------------------
        def has_any(candidates):
            return any(c in df.columns for c in candidates)

        # -------------------------------------------------
        # SEMANTIC SIGNALS (STRICT, NON-FUZZY)
        # -------------------------------------------------
        signals = {
            # Identity (clinical anchor)
            "pid": has_any([
                "patient_id", "patientid", "mrn", "medical_record_number"
            ]),

            # Time / Lifecycle
            "date": has_any([
                "admission_date", "visit_date", "encounter_date"
            ]),
            "discharge_date": has_any([
                "discharge_date"
            ]),
            "los": has_any([
                "length_of_stay", "los"
            ]),

            # Clinical
            "diagnosis": has_any([
                "diagnosis", "icd_code", "primary_diagnosis"
            ]),

            # Operational (weak alone)
            "facility": has_any([
                "facility", "hospital", "clinic"
            ]),

            # Pharmacy / Population (supporting only)
            "supply": has_any([
                "days_supply", "supply"
            ]),
            "cost": has_any([
                "cost", "billing_amount", "total_charges"
            ]),
            "population": has_any([
                "population", "catchment_population"
            ]),
        }

        # -------------------------------------------------
        # HARD GUARDRAIL: ≥2 TOTAL SIGNALS
        # -------------------------------------------------
        signal_count = sum(bool(v) for v in signals.values())
        if signal_count < 2:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals=signals,
            )

        # -------------------------------------------------
        # HARD CLINICAL ANCHOR REQUIREMENT (CRITICAL)
        # -------------------------------------------------
        clinical_anchor = (
            signals["pid"]
            or signals["diagnosis"]
            or signals["discharge_date"]
        )

        if not clinical_anchor:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals=signals,
            )

        # -------------------------------------------------
        # WEIGHTED CONFIDENCE (ROUTING-SAFE)
        # -------------------------------------------------
        weights = {
            "pid": 0.18,
            "diagnosis": 0.16,
            "date": 0.12,
            "discharge_date": 0.10,
            "facility": 0.14,
            "los": 0.06,          # weak alone (logistics-safe)
            "cost": 0.12,         # supporting only
            "population": 0.12,   # public health support
            "supply": 0.05,       # never decisive
        }

        confidence = round(
            min(
                0.95,
                sum(weights[k] for k, v in signals.items() if v)
            ),
            2,
        )

        # -------------------------------------------------
        # CONFIDENCE FLOOR (FALSE-POSITIVE SAFE)
        # -------------------------------------------------
        if confidence < 0.45:
            return DomainDetectionResult(
                domain=None,
                confidence=confidence,
                signals=signals,
            )

        # -------------------------------------------------
        # SAFE SUB-DOMAIN HINTING (NON-BINDING)
        # -------------------------------------------------
        sub_domains = infer_healthcare_subdomains(df, signals)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=confidence,
            signals={
                **signals,
                "likely_subdomains": sub_domains,
            },
        )


# -----------------------------------------------------
# REGISTRY HOOK
# -----------------------------------------------------
def register(registry):
    registry.register(
        "healthcare",
        HealthcareDomain,
        HealthcareDomainDetector,
    )
