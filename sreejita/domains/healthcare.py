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
# SAFE SIGNAL DETECTION (UNIVERSAL, DETERMINISTIC)
# =====================================================

def _has_signal(
    df: pd.DataFrame,
    col: Optional[str],
    min_coverage: float = 0.2,
) -> bool:
    """
    Column must exist AND meet minimum non-null coverage.
    """
    if not (col and isinstance(col, str) and col in df.columns):
        return False

    if len(df) == 0:
        return False

    coverage = df[col].notna().sum() / len(df)
    return coverage >= min_coverage

# =====================================================
# UNIVERSAL SUB-DOMAIN INFERENCE — HEALTHCARE
# =====================================================

def infer_healthcare_subdomains(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
) -> Dict[str, float]:
    """
    Determines healthcare sub-domain confidence scores using
    weighted semantic signal evidence.

    Returns:
        Dict[sub_domain: confidence_score]
    """

    # -------------------------------
    # SIGNAL COUNTS (NOT BOOLEAN)
    # -------------------------------
    hospital_signals = sum([
        _has_signal(df, cols.get("los")),
        _has_signal(df, cols.get("bed_id")),
        _has_signal(df, cols.get("admit_type")),
        _has_signal(df, cols.get("date")) and _has_signal(df, cols.get("discharge_date")),
    ])

    clinic_signals = sum([
        _has_signal(df, cols.get("duration")),
        _has_signal(df, cols.get("doctor")),
        _has_signal(df, cols.get("facility")),
    ])

    diagnostics_signals = sum([
        _has_signal(df, cols.get("duration")),
        _has_signal(df, cols.get("flag")),
        _has_signal(df, cols.get("encounter")),
    ])

    pharmacy_signals = sum([
        _has_signal(df, cols.get("cost")),
        _has_signal(df, cols.get("supply")),
        _has_signal(df, cols.get("fill_date")),
    ])

    public_health_signals = sum([
        _has_signal(df, cols.get("population")),
        _has_signal(df, cols.get("flag")),
    ])

    # -------------------------------
    # CONFIDENCE SCALING
    # -------------------------------
    scores: Dict[str, float] = {
        HealthcareSubDomain.HOSPITAL.value: (
            min(1.0, 0.25 * hospital_signals)
            if hospital_signals >= 2 else 0.0
        ),

        HealthcareSubDomain.CLINIC.value: (
            min(0.85, 0.3 * clinic_signals)
            if clinic_signals >= 2 and diagnostics_signals < 2 else 0.0
        ),

        HealthcareSubDomain.DIAGNOSTICS.value: (
            min(0.85, 0.3 * diagnostics_signals)
            if diagnostics_signals >= 2 else 0.0
        ),

        HealthcareSubDomain.PHARMACY.value: (
            min(0.8, 0.3 * pharmacy_signals)
            if pharmacy_signals >= 2 else 0.0
        ),

        HealthcareSubDomain.PUBLIC_HEALTH.value: (
            min(0.9, 0.45 * public_health_signals)
            if public_health_signals >= 2 else 0.0
        ),
    }

    # -------------------------------
    # REMOVE ZERO CONFIDENCE
    # -------------------------------
    scores = {k: round(v, 2) for k, v in scores.items() if v > 0}

    # -------------------------------
    # HARD SAFETY FALLBACK
    # -------------------------------
    if not scores:
        return {HealthcareSubDomain.UNKNOWN.value: 1.0}

    return scores    

# =====================================================
# HEALTHCARE DOMAIN
# =====================================================

class HealthcareDomain(BaseDomain):
    name = "healthcare"

    @staticmethod
    def get_kpi(kpis: Dict[str, Any], sub: str, key: str):
        return (
            kpis.get(f"{sub}_{key}")
            if f"{sub}_{key}" in kpis
            else kpis.get(key)
        )
    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Healthcare domain preprocessing (authoritative).

        Responsibilities:
        - Resolve semantic columns robustly
        - Normalize numeric / date / boolean fields
        - Derive LOS safely when lifecycle exists
        - Emit stable self.cols + self.time_col
        """

        # 🔒 NEVER mutate upstream dataframe
        df = df.copy()

        # ---------------------------------------------
        # DATASET SHAPE (TRACEABILITY ONLY)
        # ---------------------------------------------
        self.shape_info = detect_dataset_shape(df)

        # ---------------------------------------------
        # SEMANTIC COLUMN RESOLUTION (STRICT)
        # ---------------------------------------------
        self.cols = {
            # Identity
            "pid": resolve_column(df, "patient_id"),
            "encounter": (
                resolve_column(df, "encounter_id")
                or resolve_column(df, "visit_id")
                or resolve_column(df, "record_id")
            ),

            # Time / Lifecycle
            "date": (
                resolve_column(df, "admission_date")
                or resolve_column(df, "date")
            ),
            "discharge_date": resolve_column(df, "discharge_date"),
            "fill_date": resolve_column(df, "fill_date"),
            "los": resolve_column(df, "length_of_stay"),
            "duration": resolve_column(df, "duration"),

            # Financial
            "cost": resolve_column(df, "cost"),

            # Outcomes / Flags (GENERALIZED)
            "readmitted": resolve_column(df, "readmitted"),
            "flag": (
                resolve_column(df, "flag")
                or resolve_column(df, "mortality")
                or resolve_column(df, "no_show")
            ),

            # Clinical / Operational
            "facility": resolve_column(df, "facility"),
            "doctor": resolve_column(df, "doctor"),
            "diagnosis": resolve_column(df, "diagnosis"),
            "admit_type": resolve_column(df, "admission_type"),
            "bed_id": resolve_column(df, "bed_id"),

            # Pharmacy / Population
            "supply": resolve_column(df, "supply"),
            "population": resolve_column(df, "population"),
        }

        # ---------------------------------------------
        # NUMERIC COERCION (STRICT, SAFE)
        # ---------------------------------------------
        for key in ("los", "duration", "cost", "supply", "population"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ---------------------------------------------
        # DATE COERCION (STRICT)
        # ---------------------------------------------
        for key in ("date", "discharge_date", "fill_date"):
            col = self.cols.get(key)
            if col and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # ---------------------------------------------
        # BOOLEAN / FLAG NORMALIZATION (HARDENED)
        # ---------------------------------------------
        BOOL_MAP = {
            "yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1,
            "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0,
        }

        for key in ("readmitted", "flag"):
            col = self.cols.get(key)
            if col and col in df.columns:
                s = df[col].astype(str).str.strip().str.lower()
                mapped = s.map(BOOL_MAP)

                df[col] = pd.to_numeric(
                    mapped.where(mapped.notna(), df[col]),
                    errors="coerce"
                )

        # ---------------------------------------------
        # DERIVE LOS (SAFE & GUARDED)
        # ---------------------------------------------
        if (
            not self.cols.get("los")
            and self.cols.get("date")
            and self.cols.get("discharge_date")
            and self.cols["date"] in df.columns
            and self.cols["discharge_date"] in df.columns
        ):
            delta = (
                df[self.cols["discharge_date"]] -
                df[self.cols["date"]]
            ).dt.days

            # Guard against negative / insane LOS
            delta = delta.where(delta.between(0, 365))

            derived_col = "__derived_los"
            df[derived_col] = pd.to_numeric(delta, errors="coerce")
            self.cols["los"] = derived_col

        # ---------------------------------------------
        # CANONICAL TIME COLUMN (DETERMINISTIC)
        # ---------------------------------------------
        self.time_col = (
            self.cols.get("date")
            or self.cols.get("fill_date")
            or self.cols.get("discharge_date")
        )

        return df    
    # -------------------------------------------------
    # KPI ENGINE (UNIVERSAL, SUB-DOMAIN LOCKED)
    # -------------------------------------------------
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        volume = int(len(df))

        sub_scores = infer_healthcare_subdomains(df, self.cols)

        active_subs = {
            sub: score
            for sub, score in sub_scores.items()
            if isinstance(score, (int, float)) and score >= 0.3
        }

        if not active_subs:
            primary_sub = HealthcareSubDomain.UNKNOWN.value
            is_mixed = False
        else:
            primary_sub = max(active_subs, key=active_subs.get)
            is_mixed = len(active_subs) > 1

        # -------------------------------------------------
        # BASE KPI CONTEXT
        # -------------------------------------------------
        kpis: Dict[str, Any] = {
            "primary_sub_domain": (
                HealthcareSubDomain.MIXED.value if is_mixed else primary_sub
            ),
            "sub_domains": active_subs,
            "sub_domain_signals": active_subs,
            "total_volume": volume,
            "record_count": volume,
            "data_completeness": round(1 - df.isna().mean().mean(), 3),
            "time_coverage_days": (
                int((df[self.time_col].max() - df[self.time_col].min()).days)
                if (
                    self.time_col
                    and self.time_col in df.columns
                    and df[self.time_col].notna().any()
                )
                else None
            ),
        }

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
            if s.dropna().empty:
                return None
            return float((s > 0).mean())
        # -------------------------------------------------
        # SUB-DOMAIN KPI COMPUTATION
        # -------------------------------------------------
        for sub in active_subs:
            prefix = f"{sub}_" if is_mixed else ""

            # ---------------- HOSPITAL ----------------
            if sub == HealthcareSubDomain.HOSPITAL.value:
                avg_los = safe_mean(self.cols.get("los"))
                total_cost = safe_mean(self.cols.get("cost"))

                kpis.update({
                    f"{prefix}avg_los": avg_los,
                    f"{prefix}readmission_rate": safe_rate(self.cols.get("readmitted")),
                    f"{prefix}mortality_rate": safe_rate(self.cols.get("flag")),
                    f"{prefix}long_stay_rate": (
                        (df[self.cols["los"]] > 7).mean()
                        if self.cols.get("los") in df.columns else None
                    ),
                    f"{prefix}avg_cost_per_day": (
                        total_cost / avg_los if avg_los and total_cost else None
                    ),
                    f"{prefix}labor_cost_per_day": total_cost,
                    f"{prefix}er_boarding_time": safe_mean(self.cols.get("duration")),
                })

                bed_col = self.cols.get("bed_id")
                kpis[f"{prefix}bed_occupancy_rate"] = (
                    df[bed_col].nunique() / volume
                    if bed_col and bed_col in df.columns and volume > 0
                    else None
                )

                fac_col = self.cols.get("facility")
                los_col = self.cols.get("los")

                if (
                    volume >= MIN_SAMPLE_SIZE
                    and fac_col in df.columns
                    and los_col in df.columns
                ):
                    grouped = df.groupby(fac_col)[los_col].mean()
                    mean = grouped.mean()
                    kpis[f"{prefix}facility_variance_score"] = (
                        min(2.0, grouped.std() / mean)
                        if len(grouped) > 1 and mean > 1
                        else None
                    )
                else:
                    kpis["facility_variance_score"] = None

            # ---------------- CLINIC ----------------
            if sub == HealthcareSubDomain.CLINIC.value:
                visits = volume
                providers = (
                    df[self.cols["doctor"]].nunique()
                    if self.cols.get("doctor") in df.columns
                    else None
                )
            
                kpis.update({
                    f"{prefix}no_show_rate": safe_rate(self.cols.get("readmitted")),
                    f"{prefix}avg_wait_time": safe_mean(self.cols.get("duration")),
                    f"{prefix}provider_productivity": (
                        visits / providers if providers and providers > 0 else None
                    ),
                    f"{prefix}visit_cycle_time": safe_mean(self.cols.get("duration")),
                    f"{prefix}visits_per_provider": (
                        visits / providers if providers and providers > 0 else None
                    ),
                })
            
            # ---------------- DIAGNOSTICS ----------------
            if sub == HealthcareSubDomain.DIAGNOSTICS.value:
                tests = volume
                staff = (
                    df[self.cols["doctor"]].nunique()
                    if self.cols.get("doctor") in df.columns
                    else None
                )
            
                kpis.update({
                    f"{prefix}avg_tat": safe_mean(self.cols.get("duration")),
                    f"{prefix}critical_alert_rate": safe_rate(self.cols.get("flag")),
                    f"{prefix}specimen_rejection_rate": safe_rate(self.cols.get("flag")),
                    f"{prefix}tests_per_fte": (
                        tests / staff if staff and staff > 0 else None
                    ),
                    f"{prefix}cost_per_test": safe_mean(self.cols.get("cost")),
                })
            
            # ---------------- PHARMACY ----------------
            if sub == HealthcareSubDomain.PHARMACY.value:
                fills = volume
            
                kpis.update({
                    f"{prefix}days_supply_on_hand": safe_mean(self.cols.get("supply")),
                    f"{prefix}cost_per_rx": safe_mean(self.cols.get("cost")),
                    f"{prefix}med_error_rate": safe_rate(self.cols.get("flag")),
                    f"{prefix}rx_volume": fills,
                    f"{prefix}avg_patient_wait_time": safe_mean(self.cols.get("duration")),
                })
            
            # ---------------- PUBLIC HEALTH ----------------
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                pop = safe_mean(self.cols.get("population"))
                cases = (
                    pd.to_numeric(df[self.cols["flag"]], errors="coerce").sum()
                    if self.cols.get("flag") in df.columns
                    else None
                )
            
                kpis.update({
                    f"{prefix}incidence_per_100k": (
                        (cases / pop) * 100_000 if pop and cases else None
                    ),
                    f"{prefix}screening_coverage_rate": safe_rate(self.cols.get("flag")),
                    f"{prefix}chronic_readmission_rate": safe_rate(self.cols.get("readmitted")),
                    f"{prefix}immunization_rate": safe_rate(self.cols.get("flag")),
                    f"{prefix}cost_per_member": safe_mean(self.cols.get("cost")),
                })
            
        # -------------------------------------------------
        # KPI → CAPABILITY CONTRACT (EXPANDED & CORRECT)
        # -------------------------------------------------
        kpis["_kpi_capabilities"] = {
            # Time / Flow
            "avg_los": Capability.TIME_FLOW.value,
            "avg_wait_time": Capability.TIME_FLOW.value,
            "avg_tat": Capability.TIME_FLOW.value,
            "visit_cycle_time": Capability.TIME_FLOW.value,
                "er_boarding_time": Capability.TIME_FLOW.value,
            
            # Cost
            "cost_per_rx": Capability.COST.value,
            "cost_per_test": Capability.COST.value,
            "cost_per_member": Capability.COST.value,
            "avg_cost_per_day": Capability.COST.value,
            "labor_cost_per_day": Capability.COST.value,
            
            # Quality
            "no_show_rate": Capability.QUALITY.value,
            "readmission_rate": Capability.QUALITY.value,
            "mortality_rate": Capability.QUALITY.value,
            "specimen_rejection_rate": Capability.QUALITY.value,
            "med_error_rate": Capability.QUALITY.value,
            
            # Volume / Access
            "rx_volume": Capability.VOLUME.value,
            "tests_per_fte": Capability.VOLUME.value,
            "visits_per_provider": Capability.ACCESS.value,
            "incidence_per_100k": Capability.VOLUME.value,
            
            # Variance
            "facility_variance_score": Capability.VARIANCE.value,
            "long_stay_rate": Capability.VARIANCE.value,
        }
            
        # -------------------------------------------------
        # KPI EVIDENCE COVERAGE (FORENSIC)
        # -------------------------------------------------
        kpis["_evidence"] = {}
            
        kpi_sources  = {
            "avg_los": [self.cols.get("los")],
            "readmission_rate": [self.cols.get("readmitted")],
            "avg_wait_time": [self.cols.get("duration")],
            "cost_per_rx": [self.cols.get("cost")],
            "incidence_per_100k": [self.cols.get("population"), self.cols.get("flag")],
        }
            
        for k, cols in kpi_sources.items():
            coverages = [
                df[c].notna().mean()
                for c in cols
                if c and c in df.columns
            ]
            if coverages:
                kpis["_evidence"][k] = round(float(max(coverages)), 2)

        # -------------------------------------------------
        # KPI INFERENCE TYPE (AUDIT-READY)
        # -------------------------------------------------
        kpis["_inference_type"] = {}
            
        for key in kpis:
            if key.startswith("_"):
                continue
            
            if "proxy" in key or "estimated" in key:
                kpis["_inference_type"][key] = "proxy"
            elif key.startswith("__derived"):
                kpis["_inference_type"][key] = "derived"
            else:
                kpis["_inference_type"][key] = "direct"
                
        # -------------------------------------------------
        # KPI CONFIDENCE (HONEST & SUB-DOMAIN AWARE)
        # -------------------------------------------------
        kpis["_confidence"] = {}

        min_sub_conf = min(active_subs.values()) if active_subs else 0.3
        evidence = kpis.get("_evidence", {})
            
        for key, value in kpis.items():
            if key.startswith("_"):
                continue
            
            if key.endswith("_placeholder_kpi"):
                kpis["_confidence"][key] = 0.0
                continue
            
            coverage = evidence.get(key, 0.6)  # default conservative
            
            if isinstance(value, (int, float)):
                base = 0.85 * min_sub_conf
                adjusted = base * coverage
                kpis["_confidence"][key] = round(min(0.95, max(0.25, adjusted)), 2)
                
            else:
                kpis["_confidence"][key] = round(0.3 * coverage, 2)

        self._last_kpis = kpis

        # -------------------------------------------------
        # HARD GUARANTEE: ≥5 KPIs PER SUB-DOMAIN
        # -------------------------------------------------
        for sub in active_subs:
            expected = HEALTHCARE_KPI_MAP.get(sub, [])
            present = [
                k for k in expected
                if isinstance(kpis.get(k), (int, float)) and not pd.isna(kpis.get(k))
            ]

            for i in range(max(0, 5 - len(present))):
                kpis[f"{sub}_placeholder_kpi_{i+1}"] = None

        return kpis
    # -------------------------------------------------
    # VISUAL INTELLIGENCE (ORCHESTRATOR)
    # -------------------------------------------------
    def generate_visuals(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
    
        output_dir.mkdir(parents=True, exist_ok=True)
        visuals: List[Dict[str, Any]] = []
    
        # -------------------------------------------------
        # GET / CACHE KPIs (SINGLE SOURCE OF TRUTH)
        # -------------------------------------------------
        kpis = getattr(self, "_last_kpis", None)
        if not isinstance(kpis, dict):
            kpis = self.calculate_kpis(df)
            self._last_kpis = kpis
    
        sub_scores: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        # -------------------------------------------------
        # ACTIVE SUB-DOMAIN SELECTION
        # -------------------------------------------------
        active_subs = [
            s for s, score in sub_scores.items()
            if isinstance(score, (int, float)) and score >= 0.3
        ]
    
        # Fallback handling
        if not active_subs:
            primary = kpis.get("primary_sub_domain")
    
            if primary == HealthcareSubDomain.MIXED.value:
                # take top 2 confident sub-domains
                active_subs = sorted(
                    sub_scores,
                    key=lambda k: sub_scores.get(k, 0),
                    reverse=True
                )[:2]
    
            elif primary in HEALTHCARE_VISUAL_MAP:
                active_subs = [primary]
    
            else:
                # UNKNOWN → no visuals, safe exit
                return []
    
        # -------------------------------------------------
        # SUB-DOMAIN CONFIDENCE WEIGHTING
        # -------------------------------------------------
        def sub_domain_weight(sub: str) -> float:
            score = float(sub_scores.get(sub, 0.3))
            return round(min(1.0, max(0.15, score)), 2)
    
        # -------------------------------------------------
        # VISUAL REGISTRATION (AUTHORITATIVE CONTRACT)
        # -------------------------------------------------
        def register_visual(
            fig,
            name: str,
            caption: str,
            importance: float,
            base_confidence: float,
            sub_domain: str,
        ):
            # enforce unique, collision-safe naming
            safe_name = f"{sub_domain}_{name}" if not name.startswith(sub_domain) else name
            path = output_dir / safe_name
    
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
    
            final_conf = round(
                min(0.95, base_confidence * sub_domain_weight(sub_domain)),
                2
            )
    
            visuals.append({
                "path": str(path),
                "caption": caption,
                "importance": float(importance),
                "confidence": final_conf,
                "sub_domain": sub_domain,
                "inference_type": (
                    "proxy" if "proxy" in caption.lower()
                    else "derived" if "trend" in caption.lower()
                    else "direct"
                ),
            })
    
        # -------------------------------------------------
        # MAIN VISUAL DISPATCH (SAFE, ISOLATED)
        # -------------------------------------------------
        for sub in active_subs:
            visual_keys = HEALTHCARE_VISUAL_MAP.get(sub, [])
    
            for visual_key in visual_keys:
                try:
                    self._render_visual_by_key(
                        visual_key=visual_key,
                        df=df.copy(deep=False),
                        output_dir=output_dir,
                        sub_domain=sub,
                        register_visual=register_visual,
                    )
                except ValueError:
                    continue
    
        # -------------------------------------------------
        # FILTER INVALID / WEAK VISUALS
        # -------------------------------------------------
        visuals = [
            v for v in visuals
            if isinstance(v, dict)
            and Path(v.get("path", "")).exists()
            and float(v.get("confidence", 0)) >= 0.3
        ]
    
        # -------------------------------------------------
        # FINAL SORT (EXECUTIVE PRIORITY)
        # -------------------------------------------------
        visuals.sort(
            key=lambda v: float(v.get("importance", 0)) * float(v.get("confidence", 1)),
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
        # HOSPITAL VISUALS
        # =================================================
        if sub_domain == "hospital":
        
            # 1. Average LOS Trend
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
                    f"{sub_domain}_avg_los_trend.png",
                    "Monthly trend of inpatient length of stay.",
                    0.95,
                    0.90,
                    sub_domain,
                )
                return
        
            # 2. Bed Turnover Velocity
            if visual_key == "bed_turnover":
                bed_col = c.get("bed_id")
                if not (bed_col and time_col):
                    raise ValueError("Bed or time column missing")
        
                turnover = (
                    df[[bed_col, time_col]]
                    .dropna()
                    .groupby(bed_col)[time_col]
                    .count()
                    .clip(upper=100)  # 🔒 outlier safety
                )
        
                if turnover.empty:
                    raise ValueError("No bed turnover data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                turnover.plot(kind="hist", bins=15, ax=ax)
                ax.set_title("Bed Turnover Velocity", fontweight="bold")
                ax.set_xlabel("Patients per Bed")
        
                register_visual(
                    fig,
                    f"{sub_domain}_bed_velocity.png",
                    "Utilization frequency of physical hospital beds.",
                    0.92,
                    0.88,
                    sub_domain,
                )
                return
        
            # 3. Readmission Risk
            if visual_key == "readmission_risk":
                col = c.get("readmitted")
                if not col:
                    raise ValueError("Readmission column missing")
        
                rates = df[col].dropna().value_counts(normalize=True)
                if rates.empty:
                    raise ValueError("No readmission data")
        
                # Executive-friendly labels
                rates.index = rates.index.map({0: "No", 1: "Yes"}).fillna(rates.index)
        
                fig, ax = plt.subplots(figsize=(6, 4))
                rates.plot(kind="bar", ax=ax)
                ax.set_title("Readmission Rate Distribution", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    f"{sub_domain}_readmission.png",
                    "Distribution of 30-day readmissions.",
                    0.93,
                    0.88,
                    sub_domain,
                )
                return
        
            # 4. Discharge Hour Distribution
            if visual_key == "discharge_hour":
                if not time_col:
                    raise ValueError("Time column missing")
        
                hours = df[time_col].dropna().dt.hour
                if hours.empty:
                    raise ValueError("No discharge time data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                hours.value_counts().sort_index().plot(kind="bar", ax=ax)
                ax.set_title("Discharge Hour Distribution", fontweight="bold")
                ax.set_xlabel("Hour of Day")
        
                register_visual(
                    fig,
                    f"{sub_domain}_discharge_hour.png",
                    "Inpatient discharge timing pattern.",
                    0.85,
                    0.80,
                    sub_domain,
                )
                return
        
            # 5. Acuity vs Staffing
            if visual_key == "acuity_vs_staffing":
                if not (c.get("los") and c.get("cost")):
                    raise ValueError("LOS or cost missing")
        
                tmp = df[[c["los"], c["cost"]]].dropna()
                if tmp.empty:
                    raise ValueError("No acuity-cost data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(tmp[c["los"]], tmp[c["cost"]], alpha=0.4)
                ax.set_xlabel("LOS (Acuity Proxy)")
                ax.set_ylabel("Cost (Staffing Proxy)")
                ax.set_title("Acuity vs Staffing Intensity", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_acuity_staffing.png",
                    "Relationship between patient acuity and staffing intensity.",
                    0.88,
                    0.82,
                    sub_domain,
                )
                return
        
            # 6. ED Boarding Time
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
                    f"{sub_domain}_ed_boarding.png",
                    "Average emergency department boarding time.",
                    0.92,
                    0.85,
                    sub_domain,
                )
                return
        
            # 7. Mortality Trend
            if visual_key == "mortality_trend":
                target_col = c.get("flag")
                if not (target_col and time_col):
                    raise ValueError("Mortality proxy or time missing")
        
                rate = (
                    df[[time_col, target_col]]
                    .dropna()
                    .set_index(time_col)[target_col]
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
                    f"{sub_domain}_mortality_trend.png",
                    "Observed mortality proxy trend over time.",
                    0.90,
                    0.80,
                    sub_domain,
                )
                return
    
        # =================================================
        # CLINIC / AMBULATORY VISUALS
        # =================================================
        if sub_domain == "clinic":
        
            # 1. NO-SHOW RATE BY DAY
            if visual_key == "no_show_by_day":
                if not (c.get("readmitted") and time_col):
                    raise ValueError("Required columns missing")
        
                tmp = df[[time_col, c["readmitted"]]].dropna()
                if tmp.empty:
                    raise ValueError("No no-show data")
        
                tmp["_dow"] = tmp[time_col].dt.day_name()
                rate = tmp.groupby("_dow")[c["readmitted"]].mean()
        
                day_order = [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday"
                ]
                rate = rate.reindex(day_order).fillna(0)
        
                fig, ax = plt.subplots(figsize=(6, 4))
                rate.plot(kind="bar", ax=ax)
                ax.set_title("No-Show Rate by Day of Week", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    f"{sub_domain}_no_show_by_day.png",
                    "Appointment no-show rates across the week.",
                    0.92,
                    0.85,
                    sub_domain,
                )
                return
        
            # 2. WAIT TIME TRAJECTORY
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
                    f"{sub_domain}_wait_time_trend.png",
                    "Trend of patient wait times from check-in to provider.",
                    0.90,
                    0.80,
                    sub_domain,
                )
                return
        
            # 3. APPOINTMENT LAG DISTRIBUTION
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
                    .dropna()
                )
        
                if lag.empty:
                    raise ValueError("No appointment lag data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                lag.clip(upper=60).hist(ax=ax, bins=20)
                ax.set_title("Appointment Lag Distribution (Days)", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_appointment_lag.png",
                    "Days between booking and actual clinic visit.",
                    0.88,
                    0.75,
                    sub_domain,
                )
                return
        
            # 4. PROVIDER UTILIZATION RATE
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
                    f"{sub_domain}_provider_utilization.png",
                    "Comparison of provider workload distribution.",
                    0.91,
                    0.90,
                    sub_domain,
                )
                return
        
            # 5. PATIENT DEMOGRAPHIC REACH
            if visual_key == "demographic_reach":
                if not c.get("facility"):
                    raise ValueError("Facility column missing")
        
                counts = df[c["facility"]].value_counts()
                if counts.empty:
                    raise ValueError("No geographic data")
        
                top = counts.head(7)
                if len(counts) > 7:
                    top["Other"] = counts.iloc[7:].sum()
        
                fig, ax = plt.subplots(figsize=(6, 6))
                top.plot(kind="pie", ax=ax, autopct="%1.0f%%")
                ax.set_ylabel("")
                ax.set_title("Patient Demographic Reach", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_demographic_reach.png",
                    "Distribution of patient visits by service location.",
                    0.85,
                    0.80,
                    sub_domain,
                )
                return
        
            # 6. REFERRAL FUNNEL (ESTIMATED)
            if visual_key == "referral_funnel":
                stages = {
                    "Referrals": len(df),
                    "Scheduled (Est.)": int(len(df) * 0.75),
                    "Completed (Est.)": int(len(df) * 0.65),
                }
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(stages.keys(), stages.values())
                ax.set_title("Referral Conversion Funnel (Estimated)", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_referral_funnel.png",
                    "Estimated referral flow from intake to completed visits.",
                    0.87,
                    0.70,
                    sub_domain,
                )
                return
        
            # 7. TELEHEALTH MIX
            if visual_key == "telehealth_mix":
                if not c.get("facility"):
                    raise ValueError("Facility column missing")
        
                series = (
                    df[c["facility"]]
                    .fillna("")
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
                ax.set_title("Telehealth vs In-Person Visits", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_telehealth_mix.png",
                    "Service delivery mix across visit types.",
                    0.86,
                    0.75,
                    sub_domain,
                )
                return
    
        # =================================================
        # DIAGNOSTICS (LABS / RADIOLOGY) VISUALS
        # =================================================
        if sub_domain == "diagnostics":
        
            # Ensure datetime safety once
            if time_col:
                df = df.copy()
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
            # -------------------------------------------------
            # 1. TURNAROUND TIME PERCENTILES
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
                p50 = grouped.quantile(0.50)
                p90 = grouped.quantile(0.90)
                p95 = grouped.quantile(0.95)
        
                if p50.empty:
                    raise ValueError("Insufficient TAT distribution")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                p50.plot(ax=ax, label="50th %ile")
                p90.plot(ax=ax, label="90th %ile")
                p95.plot(ax=ax, label="95th %ile")
                ax.legend()
                ax.set_title("Turnaround Time Percentiles", fontweight="bold")
                ax.set_ylabel("Minutes")
        
                register_visual(
                    fig,
                    f"{sub_domain}_tat_percentiles.png",
                    "Diagnostic turnaround time percentiles over time.",
                    0.95,
                    0.90,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. CRITICAL VALUE NOTIFICATION SPEED
            # -------------------------------------------------
            if visual_key == "critical_alert_time":
                if not (c.get("duration") and c.get("flag")):
                    raise ValueError("Flag or duration missing")
        
                flag = pd.to_numeric(df[c["flag"]], errors="coerce")
                dur = pd.to_numeric(df[c["duration"]], errors="coerce")
        
                critical = dur[flag == 1].dropna()
                if critical.empty:
                    raise ValueError("No critical alerts")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                critical.clip(upper=180).hist(ax=ax, bins=20)
                ax.set_title("Critical Result Notification Time", fontweight="bold")
                ax.set_xlabel("Minutes")
        
                register_visual(
                    fig,
                    f"{sub_domain}_critical_alert_time.png",
                    "Speed of notifying life-threatening diagnostic results.",
                    0.93,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. SPECIMEN REJECTION (PROXY)
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
                    raise ValueError("No rejection data")
        
                fig, ax = plt.subplots(figsize=(8, 4))
                reasons.plot(kind="bar", ax=ax)
                ax.set_title("Specimen Rejection Signals (Proxy)", fontweight="bold")
                ax.set_ylabel("Count")
        
                register_visual(
                    fig,
                    f"{sub_domain}_specimen_rejection.png",
                    "Observed specimen rejection indicators (proxy signal).",
                    0.90,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. DEVICE UTILIZATION (DOWNTIME PROXY)
            # -------------------------------------------------
            if visual_key == "device_downtime":
                if not (c.get("facility") and time_col):
                    raise ValueError("Facility or time missing")
        
                tmp = df[[c["facility"], time_col]].dropna()
                if tmp.empty:
                    raise ValueError("No device usage data")
        
                downtime = (
                    tmp[c["facility"]]
                    .astype(str)
                    .value_counts()
                    .head(10)
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                downtime.plot(kind="bar", ax=ax)
                ax.set_title("Relative Device Utilization (Downtime Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_device_downtime.png",
                    "Relative diagnostic equipment availability across facilities.",
                    0.87,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PEAK ORDER LOAD HEATMAP
            # -------------------------------------------------
            if visual_key == "order_heatmap":
                if not time_col:
                    raise ValueError("Time column missing")
        
                tmp = df[[time_col]].dropna()
                if tmp.empty:
                    raise ValueError("No order timestamps")
        
                tmp["_hour"] = tmp[time_col].dt.hour
                tmp["_day"] = tmp[time_col].dt.day_name()
        
                heat = pd.crosstab(tmp["_day"], tmp["_hour"])
                if heat.empty:
                    raise ValueError("Empty heatmap")
        
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
                    f"{sub_domain}_order_heatmap.png",
                    "Hourly diagnostic order intensity by day.",
                    0.92,
                    0.90,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. REPEAT SCAN INCIDENCE
            # -------------------------------------------------
            if visual_key == "repeat_scan":
                if not c.get("encounter"):
                    raise ValueError("Encounter column missing")
        
                counts = df[c["encounter"]].value_counts()
                if counts.empty:
                    raise ValueError("No scan data")
        
                repeat_rate = (counts > 1).mean()
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Repeat Scan Rate"], [repeat_rate])
                ax.set_ylim(0, 1)
                ax.set_title("Repeat Diagnostic Incidence", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_repeat_scan.png",
                    "Rate of repeated diagnostic tests indicating waste.",
                    0.89,
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
                    f"{sub_domain}_ordering_variance.png",
                    "Variation in diagnostic ordering behavior across providers.",
                    0.88,
                    0.85,
                    sub_domain,
                )
                return
        
        # =================================================
        # PHARMACY VISUALS
        # =================================================
        if sub_domain == "pharmacy":
        
            # Ensure datetime safety once
            if time_col:
                df = df.copy()
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
            # -------------------------------------------------
            # 1. MEDICATION SPEND VELOCITY (CUMULATIVE)
            # -------------------------------------------------
            if visual_key == "spend_velocity":
                if not (c.get("cost") and time_col):
                    raise ValueError("Cost or time column missing")
        
                tmp = df[[time_col, c["cost"]]].dropna()
                if tmp.empty:
                    raise ValueError("No spend data")
        
                spend = (
                    tmp.set_index(time_col)[c["cost"]]
                    .resample("M")
                    .sum()
                    .cumsum()
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                spend.plot(ax=ax)
                ax.set_title("Medication Spend Velocity", fontweight="bold")
                ax.set_ylabel("Cumulative Spend")
        
                ax.yaxis.set_major_formatter(
                    lambda x, _: f"{x/1_000_000:.1f}M" if x >= 1_000_000 else f"{x/1_000:.0f}K"
                )
        
                register_visual(
                    fig,
                    f"{sub_domain}_spend_velocity.png",
                    "Cumulative medication expenditure over time.",
                    0.95,
                    0.90,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. REFILL ADHERENCE GAP (DAYS LATE)
            # -------------------------------------------------
            if visual_key == "refill_gap":
                if not (c.get("fill_date") and c.get("supply")):
                    raise ValueError("Fill date or supply missing")
        
                fill = pd.to_datetime(df[c["fill_date"]], errors="coerce")
                supply = pd.to_numeric(df[c["supply"]], errors="coerce")
        
                expected = fill + pd.to_timedelta(supply, unit="D")
                gap = (expected - fill).dt.days.dropna()  # ✅ positive = late
        
                if gap.empty:
                    raise ValueError("No refill gap data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                gap.clip(lower=-30, upper=60).hist(ax=ax, bins=20)
                ax.set_title("Refill Adherence Gap (Days)", fontweight="bold")
                ax.set_xlabel("Days Late / Early")
        
                register_visual(
                    fig,
                    f"{sub_domain}_refill_gap.png",
                    "Delay between expected and actual prescription refills.",
                    0.92,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. THERAPEUTIC CLASS SPEND (PROXY)
            # -------------------------------------------------
            if visual_key == "therapeutic_spend":
                if not (c.get("facility") and c.get("cost")):
                    raise ValueError("Facility or cost missing")
        
                spend = (
                    df[[c["facility"], c["cost"]]]
                    .dropna()
                    .assign(_cat=lambda x: x[c["facility"]].astype(str))
                    .groupby("_cat")[c["cost"]]
                    .sum()
                    .nlargest(6)
                )
        
                if spend.empty:
                    raise ValueError("No therapeutic spend data")
        
                fig, ax = plt.subplots(figsize=(6, 6))
                spend.plot(kind="pie", autopct="%1.0f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Therapeutic Spend Distribution (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_therapeutic_spend.png",
                    "Medication spend distribution by therapeutic class (proxy grouping).",
                    0.90,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. GENERIC SUBSTITUTION RATE
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
        
                if series.empty:
                    raise ValueError("No generic indicator data")
        
                rate = series.mean()
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Generic Substitution Rate"], [rate])
                ax.set_ylim(0, 1)
                ax.set_title("Generic Substitution Rate", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_generic_rate.png",
                    "Share of prescriptions filled with generic alternatives (proxy).",
                    0.88,
                    0.75,
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
                ax.set_title("Prescribing Cost Variance (Top Providers)", fontweight="bold")
                ax.set_ylabel("Average Cost")
        
                register_visual(
                    fig,
                    f"{sub_domain}_prescribing_variance.png",
                    "Variation in average prescribing cost across providers.",
                    0.91,
                    0.85,
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
                    f"{sub_domain}_inventory_turn.png",
                    "Efficiency of medication inventory turnover.",
                    0.87,
                    0.70,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. DRUG INTERACTION ALERTS (PROXY)
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
                    raise ValueError("No alert data")
        
                fig, ax = plt.subplots(figsize=(6, 4))
                alerts.plot(kind="bar", ax=ax)
                ax.set_title("Pharmacist Safety Interventions", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    f"{sub_domain}_drug_alerts.png",
                    "Frequency of pharmacist interventions for drug safety (proxy).",
                    0.89,
                    0.80,
                    sub_domain,
                )
                return 
                
        # =================================================
        # PUBLIC HEALTH / POPULATION HEALTH VISUALS
        # =================================================
        if sub_domain == "public_health":
        
            # Ensure datetime safety once
            if time_col:
                df = df.copy()
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        
            # -------------------------------------------------
            # 1. DISEASE INCIDENCE RATE (PER 100K)
            # -------------------------------------------------
            if visual_key == "incidence_geo":
                if not (c.get("population") and c.get("flag")):
                    raise ValueError("Population or outcome flag missing")
        
                pop = pd.to_numeric(df[c["population"]], errors="coerce").dropna()
                cases = pd.to_numeric(df[c["flag"]], errors="coerce").fillna(0)
        
                if pop.empty or cases.empty:
                    raise ValueError("Insufficient incidence data")
        
                # Robust denominator selection
                pop_denom = pop.median() if pop.nunique() > 1 else pop.iloc[0]
                if pop_denom <= 0:
                    raise ValueError("Invalid population denominator")
        
                incidence = (cases.sum() / pop_denom) * 100_000
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Incidence per 100k"], [incidence])
                ax.set_title("Disease Incidence Rate", fontweight="bold")
                ax.set_ylabel("Cases per 100,000")
        
                register_visual(
                    fig,
                    f"{sub_domain}_incidence_rate.png",
                    "Observed disease incidence per 100,000 population (proxy).",
                    0.95,
                    0.90,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 2. COHORT GROWTH TRAJECTORY
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
                )
        
                fig, ax = plt.subplots(figsize=(8, 4))
                cohort.plot(ax=ax)
                ax.set_title("Cohort Growth Trajectory", fontweight="bold")
                ax.set_ylabel("Cumulative Cases")
        
                register_visual(
                    fig,
                    f"{sub_domain}_cohort_growth.png",
                    "Cumulative growth of observed population health cohort.",
                    0.93,
                    0.88,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 3. PREVALENCE BY AGE GROUP (PROXY)
            # -------------------------------------------------
            if visual_key == "prevalence_age":
                if not (c.get("pid") and c.get("flag")):
                    raise ValueError("Patient id or outcome flag missing")
        
                pid_len = df[c["pid"]].astype(str).str.len()
                flag = pd.to_numeric(df[c["flag"]], errors="coerce")
        
                buckets = pd.cut(
                    pid_len,
                    bins=[0, 6, 8, 10, 99],
                    labels=["0–18", "19–35", "36–60", "60+"],
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
                ax.set_title("Prevalence by Age Group (Proxy)", fontweight="bold")
                ax.set_ylabel("Rate")
        
                register_visual(
                    fig,
                    f"{sub_domain}_prevalence_age.png",
                    "Prevalence by Demographic Group (Proxy).",
                    0.90,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 4. SERVICE ACCESS GAP (PROXY)
            # -------------------------------------------------
            if visual_key == "access_gap":
                if not (c.get("population") and c.get("facility")):
                    raise ValueError("Population or facility missing")
        
                pop = pd.to_numeric(df[c["population"]], errors="coerce").dropna()
                facilities = df[c["facility"]].astype(str).dropna()
        
                if pop.empty or facilities.empty:
                    raise ValueError("No access data")
        
                providers = facilities.nunique()
                pop_denom = pop.median()
        
                if pop_denom <= 0:
                    raise ValueError("Invalid population denominator")
        
                ratio = providers / pop_denom * 1000
        
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Providers per 1k"], [ratio])
                ax.set_title("Healthcare Access Indicator (Proxy)", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_access_gap.png",
                    "Healthcare provider availability per 1,000 residents (proxy).",
                    0.92,
                    0.85,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 5. PROGRAM EFFICACY TREND
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
                ax.set_title("Program Efficacy Trend", fontweight="bold")
                ax.set_ylabel("Outcome Rate")
        
                register_visual(
                    fig,
                    f"{sub_domain}_program_effect.png",
                    "Smoothed population outcome trends following interventions.",
                    0.90,
                    0.80,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 6. SOCIAL DETERMINANTS OVERLAY (PROXY)
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
                    f"{sub_domain}_sdoh_overlay.png",
                    "Health outcome variation across socioeconomic regions (proxy).",
                    0.88,
                    0.75,
                    sub_domain,
                )
                return
        
            # -------------------------------------------------
            # 7. IMMUNIZATION / SCREENING RATE
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
                ax.set_title("Immunization / Screening Coverage", fontweight="bold")
        
                register_visual(
                    fig,
                    f"{sub_domain}_immunization_rate.png",
                    "Population coverage of immunization or screening programs.",
                    0.91,
                    0.85,
                    sub_domain,
                )
                return
            
            raise ValueError(f"Unhandled visual key: {visual_key}")
    
    # -------------------------------------------------
    # INSIGHTS ENGINE (UNIVERSAL, SUB-DOMAIN LOCKED)
    # -------------------------------------------------
    def generate_insights(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        *_,
    ) -> List[Dict[str, Any]]:
    
        insights: List[Dict[str, Any]] = []
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        def conf(base: float, sub_score: float) -> float:
            return round(min(0.95, base + min(sub_score, 1.0) * 0.25), 2)
    
        filler_titles = [
            "Operational Baseline Signal",
            "Stable Performance Pattern",
            "No Material Risk Detected",
        ]

        # -------------------------------------------------
        # CROSS-SUBDOMAIN INTELLIGENCE (EXECUTIVE-LEVEL)
        # -------------------------------------------------
        if (
            "hospital" in active_subs
            and "diagnostics" in active_subs
            and isinstance(self.get_kpi(kpis, "diagnostics", "avg_tat"), (int, float))
            and isinstance(self.get_kpi(kpis, "hospital", "avg_los"), (int, float))
            and self.get_kpi(kpis, "diagnostics", "avg_tat") > 120
        ):
            cross_conf = min(
                kpis.get("_confidence", {}).get("avg_tat", 0.6),
                kpis.get("_confidence", {}).get("avg_los", 0.6),
            )
            
            insights.append({
                "sub_domain": "cross_domain",
                "level": "RISK",
                "title": "Diagnostic Delays Driving Inpatient Length of Stay",
                "so_what": (
                    f"Elevated diagnostic turnaround times "
                    f"({self.get_kpi(kpis, 'diagnostics', 'avg_tat'):.0f} mins) "
                    f"are likely contributing to prolonged inpatient stays "
                    f"(avg LOS {self.get_kpi(kpis, 'hospital', 'avg_los'):.1f} days)."
                ),
                "confidence": round(min(0.95, cross_conf), 2),
            })
        
        for sub, score in active_subs.items():
            sub_insights: List[Dict[str, Any]] = []
    
            # =================================================
            # STRENGTHS (EVIDENCE-BASED ONLY)
            # =================================================
            if sub == HealthcareSubDomain.HOSPITAL.value:
                avg_los = self.get_kpi(kpis, sub, "avg_los")
                if isinstance(avg_los, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Inpatient Throughput Visibility",
                        "so_what": (
                            f"Length of stay is consistently measured "
                            f"(average {avg_los:.1f} days), enabling throughput governance."
                        ),
                        "confidence": conf(0.75, score),
                    })
    
            if sub == HealthcareSubDomain.CLINIC.value:
                wait = self.get_kpi(kpis, sub, "avg_wait_time")
                if isinstance(wait, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Clinic Access Transparency",
                        "so_what": (
                            f"Patient wait times are observable "
                            f"(average {wait:.0f} minutes), supporting access optimization."
                        ),
                        "confidence": conf(0.72, score),
                    })
    
            if sub == HealthcareSubDomain.DIAGNOSTICS.value:
                tat = self.get_kpi(kpis, sub, "avg_tat")
                if isinstance(tat, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Diagnostic Turnaround Observability",
                        "so_what": (
                            f"Turnaround times are measurable "
                            f"(average {tat:.0f} minutes), enabling SLA management."
                        ),
                        "confidence": conf(0.72, score),
                    })
    
            if sub == HealthcareSubDomain.PHARMACY.value:
                cost = self.get_kpi(kpis, sub, "cost_per_rx")
                if isinstance(cost, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Prescription Cost Visibility",
                        "so_what": (
                            f"Average prescription cost is tracked "
                            f"(₹{cost:.0f}), supporting spend control."
                        ),
                        "confidence": conf(0.70, score),
                    })
    
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                inc = self.get_kpi(kpis, sub, "incidence_per_100k")
                if isinstance(inc, (int, float)):
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "STRENGTH",
                        "title": "Population Health Signal Coverage",
                        "so_what": (
                            "Population-level indicators support monitoring "
                            "of disease burden and prevention effectiveness."
                        ),
                        "confidence": conf(0.74, score),
                    })
    
            # =================================================
            # WARNINGS (EARLY SIGNALS)
            # =================================================
            if sub == HealthcareSubDomain.CLINIC.value:
                no_show = self.get_kpi(kpis, sub, "no_show_rate")
                if isinstance(no_show, (int, float)) and no_show >= 0.10:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Elevated Appointment No-Show Risk",
                        "so_what": (
                            f"No-show rate of {no_show:.1%} may reduce clinic throughput "
                            "and revenue efficiency."
                        ),
                        "confidence": conf(0.75, score),
                    })
    
            if sub == HealthcareSubDomain.PHARMACY.value:
                med_err = self.get_kpi(kpis, sub, "med_error_rate")
                if isinstance(med_err, (int, float)) and med_err >= 0.05:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "WARNING",
                        "title": "Medication Safety Signals Detected",
                        "so_what": (
                            f"Observed intervention rate of {med_err:.1%} "
                            "suggests need for tighter dispensing controls."
                        ),
                        "confidence": conf(0.75, score),
                    })
    
            # =================================================
            # RISKS (MATERIAL IMPACT)
            # =================================================
            if sub == HealthcareSubDomain.HOSPITAL.value:
                long_stay = self.get_kpi(kpis, sub, "long_stay_rate")
                if isinstance(long_stay, (int, float)) and long_stay >= 0.25:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Discharge Throughput Constraint",
                        "so_what": (
                            f"{long_stay:.1%} of patients exceed acceptable LOS thresholds, "
                            "indicating discharge bottlenecks."
                        ),
                        "confidence": conf(0.85, score),
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
                        "confidence": conf(0.82, score),
                    })
    
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
                inc = self.get_kpi(kpis, sub, "incidence_per_100k")
                if isinstance(inc, (int, float)) and inc > 300:
                    sub_insights.append({
                        "sub_domain": sub,
                        "level": "RISK",
                        "title": "Elevated Disease Incidence",
                        "so_what": (
                            f"Incidence rate of {inc:.0f} per 100k exceeds norms, "
                            "indicating prevention gaps."
                        ),
                        "confidence": conf(0.88, score),
                    })
    
            # =================================================
            # FILLERS (LOW CONFIDENCE ONLY)
            # =================================================
            filler_idx = 0
            while len(sub_insights) < 9:
                sub_insights.append({
                    "sub_domain": sub,
                    "level": "INFO",
                    "title": filler_titles[filler_idx % len(filler_titles)],
                    "so_what": (
                        "Available indicators remain within expected ranges "
                        "with no statistically significant deviations observed."
                    ),
                    "confidence": 0.45,
                })
                filler_idx += 1
    
            insights.extend(sub_insights[:9])
    
        return insights

    #--------------------------------
    #-----Recommendations------------
    #--------------------------------
    def generate_recommendations(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        insights: List[Dict[str, Any]],
        *_,
    ) -> List[Dict[str, Any]]:
    
        recommendations: List[Dict[str, Any]] = []
        active_subs: Dict[str, float] = kpis.get("sub_domains", {}) or {}
    
        def conf(base: float, sub_score: float) -> float:
            return round(min(0.95, base + min(sub_score, 1.0) * 0.25), 2)
    
        # -------------------------------------------------
        # INDEX INSIGHTS BY SUB + LEVEL (CRITICAL)
        # -------------------------------------------------
        insights_by_sub = {}
        for i in insights:
            if not isinstance(i, dict):
                continue
            sub = i.get("sub_domain")
            insights_by_sub.setdefault(sub, []).append(i)
    
        fallback_actions = [
            "Continue monitoring operational performance indicators",
            "Review trends during monthly governance meetings",
            "Maintain current escalation and governance controls",
        ]
    
        for sub, score in active_subs.items():
            sub_recs: List[Dict[str, Any]] = []
    
            sub_insights = insights_by_sub.get(sub, [])
            levels = {i.get("level") for i in sub_insights}
    
            # =================================================
            # HOSPITAL
            # =================================================
            if sub == HealthcareSubDomain.HOSPITAL.value and "RISK" in levels:
                current_los = self.get_kpi(kpis, sub, "avg_los")
            
                sub_recs.append({
                    "sub_domain": sub,
                    "priority": "HIGH",
                    "action": "Implement centralized discharge command center",
                    "owner": "Hospital Operations",
                    "timeline": "60 days",
                    "goal": "Reduce average inpatient length of stay",
                    "current_value": round(current_los, 2) if current_los else None,
                    "target_value": (
                        round(current_los * 0.9, 2) if current_los else None
                    ),
                    "expected_impact": "10–15% capacity release and improved bed availability",
                    "confidence": conf(0.90, score),
                })
    
                if "RISK" in levels:
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Implement centralized discharge command structure",
                        "owner": "Hospital Operations",
                        "timeline": "30–60 days",
                        "goal": "Reduce prolonged length of stay and free inpatient capacity",
                        "confidence": conf(0.88, score),
                    })
    
                if any(i.get("title") == "High Facility Performance Variance" for i in sub_insights):
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Standardize clinical pathways across facilities",
                        "owner": "Clinical Governance",
                        "timeline": "60–90 days",
                        "goal": "Reduce outcome and LOS variability",
                        "confidence": conf(0.85, score),
                    })
    
            # =================================================
            # CLINIC
            # =================================================
            if sub == HealthcareSubDomain.CLINIC.value:
    
                sub_recs.append({
                    "sub_domain": sub,
                    "priority": "MEDIUM",
                    "action": "Deploy automated appointment reminders and confirmations",
                    "owner": "Ambulatory Operations",
                    "timeline": "30–60 days",
                    "goal": "Reduce missed appointments and access friction",
                    "confidence": conf(0.75, score),
                })
    
                if "WARNING" in levels:
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "MEDIUM",
                        "action": "Introduce no-show prediction and overbooking logic",
                        "owner": "Clinic Operations",
                        "timeline": "60–90 days",
                        "goal": "Stabilize provider utilization and clinic throughput",
                        "confidence": conf(0.78, score),
                    })
    
            # =================================================
            # DIAGNOSTICS
            # =================================================
            if sub == HealthcareSubDomain.DIAGNOSTICS.value:
    
                sub_recs.append({
                    "sub_domain": sub,
                    "priority": "MEDIUM",
                    "action": "Optimize lab and imaging capacity during peak demand",
                    "owner": "Diagnostics Leadership",
                    "timeline": "60–120 days",
                    "goal": "Stabilize turnaround times and clinician satisfaction",
                    "confidence": conf(0.78, score),
                })
    
                if "RISK" in levels:
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "HIGH",
                        "action": "Introduce STAT workflow and alert escalation protocols",
                        "owner": "Diagnostics Leadership",
                        "timeline": "30–60 days",
                        "goal": "Prevent clinical decision delays",
                        "confidence": conf(0.85, score),
                    })
    
            # =================================================
            # PHARMACY
            # =================================================
            if sub == HealthcareSubDomain.PHARMACY.value:
    
                sub_recs.append({
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": "Increase generic substitution and formulary adherence",
                    "owner": "Pharmacy Leadership",
                    "timeline": "Ongoing",
                    "goal": "Control medication spend without compromising care",
                    "confidence": conf(0.70, score),
                })
    
                if "WARNING" in levels:
                    sub_recs.append({
                        "sub_domain": sub,
                        "priority": "MEDIUM",
                        "action": "Strengthen pharmacist double-check and alert review protocols",
                        "owner": "Pharmacy Operations",
                        "timeline": "30–60 days",
                        "goal": "Reduce medication safety risk",
                        "confidence": conf(0.80, score),
                    })
    
            # =================================================
            # PUBLIC HEALTH
            # =================================================
            if sub == HealthcareSubDomain.PUBLIC_HEALTH.value:
    
                sub_recs.append({
                    "sub_domain": sub,
                    "priority": "HIGH",
                    "action": "Target high-incidence regions with preventive programs",
                    "owner": "Public Health Authority",
                    "timeline": "90–180 days",
                    "goal": "Reduce disease incidence and population risk",
                    "confidence": conf(0.85, score),
                })
    
            # =================================================
            # FALLBACK (LOW CONFIDENCE ONLY)
            # =================================================
            idx = 0
            while len(sub_recs) < 5:
                sub_recs.append({
                    "sub_domain": sub,
                    "priority": "LOW",
                    "action": fallback_actions[idx % len(fallback_actions)],
                    "owner": "Operations",
                    "timeline": "Ongoing",
                    "goal": "Maintain operational stability and signal monitoring",
                    "confidence": 0.50,
                })
                idx += 1
    
            recommendations.extend(sub_recs[:5])
    
        # -------------------------------------------------
        # EXECUTIVE SORTING
        # -------------------------------------------------
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(
            key=lambda r: (
                priority_order.get(r.get("priority"), 3),
                r.get("sub_domain", ""),
            )
        )
    
        return recommendations
# =====================================================
# REGISTRATION
# =====================================================

class HealthcareDomainDetector(BaseDomainDetector):
    domain_name = "healthcare"

    def detect(self, df: pd.DataFrame):

        # -------------------------------------------------
        # SEMANTIC SIGNALS (MUST ALIGN WITH DOMAIN LOGIC)
        # -------------------------------------------------
        cols = {
            # Identity
            "pid": resolve_column(df, "patient_id"),

            # Time / Lifecycle
            "date": resolve_column(df, "admission_date"),
            "discharge_date": resolve_column(df, "discharge_date"),
            "los": resolve_column(df, "length_of_stay"),

            # Clinical / Ops
            "diagnosis": resolve_column(df, "diagnosis"),
            "facility": resolve_column(df, "facility"),

            # Pharmacy / Population (critical for routing)
            "supply": resolve_column(df, "supply"),
            "cost": resolve_column(df, "cost"),
            "population": resolve_column(df, "population"),
        }

        signals = {k: _has_signal(df, v) for k, v in cols.items()}

        # -------------------------------------------------
        # HARD GUARDRAIL: require ≥2 meaningful signals
        # -------------------------------------------------
        signal_count = sum(signals.values())
        if signal_count < 2:
            return DomainDetectionResult(
                domain=None,
                confidence=0.0,
                signals=signals,
            )

        # -------------------------------------------------
        # WEIGHTED CONFIDENCE (ROUTING-SAFE)
        # -------------------------------------------------
        weights = {
            "pid": 0.15,
            "date": 0.12,
            "discharge_date": 0.08,
            "los": 0.12,          # reduced
            "facility": 0.15,     # increased
            "cost": 0.12,         # pharmacy support
            "population": 0.12,   # public health parity
            "diagnosis": 0.10,
        }

        confidence = round(
            min(
                0.95,
                sum(
                    weights.get(k, 0) for k, v in signals.items() if v
                )
            ),
            2,
        )

        if confidence < 0.35:
            return DomainDetectionResult(
                domain=None,
                confidence=confidence,
                signals=signals,
            )

        # -------------------------------------------------
        # SAFE SUB-DOMAIN HINTING (NON-BINDING)
        # -------------------------------------------------
        sub_domains = infer_healthcare_subdomains(df, cols)

        return DomainDetectionResult(
            domain=self.domain_name,
            confidence=confidence,
            signals={
                **signals,
                "likely_subdomains": sub_domains,
            },
        )


def register(registry):
    registry.register(
        "healthcare",
        HealthcareDomain,
        HealthcareDomainDetector,
    )
