# =====================================================
# DOMAIN METADATA & BUSINESS CONTEXT CONFIGURATION
# =====================================================
"""
Domain-specific metadata and validation rules.
Allows users to provide business context for better domain detection.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class HealthcareSubdomainHint(str, Enum):
    """Healthcare sub-domain hints provided by user."""
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    DIAGNOSTICS = "diagnostics"
    PHARMACY = "pharmacy"
    PUBLIC_HEALTH = "public_health"
    UNKNOWN = "unknown"


@dataclass
class DomainMetadata:
    """
    Metadata about the dataset to assist domain detection.
    
    User can provide this to short-circuit/guide detection.
    """
    business_domain: Optional[str] = None  # e.g., "healthcare", "retail"
    domain_hint: Optional[str] = None      # User-provided domain
    healthcare_subdomain: Optional[str] = None  # If healthcare, which sub-domain?
    dataset_purpose: Optional[str] = None  # e.g., "hospital operations", "pharmacy spend"
    start_date: Optional[str] = None       # Data period start
    end_date: Optional[str] = None         # Data period end
    expected_volume: Optional[int] = None  # Approx record count
    description: Optional[str] = None      # Free-form context


class DomainContextValidator:
    """
    Validates and normalizes domain metadata.
    Provides intelligent fallback handling.
    """
    
    DOMAIN_KEYWORDS = {
        "healthcare": [
            "hospital", "clinic", "patient", "medical", "clinical",
            "diagnosis", "admission", "discharge", "pharmacy", "health",
            "disease", "treatment", "provider", "care", "los", "readmit"
        ],
        "retail": [
            "sale", "store", "customer", "product", "inventory",
            "transaction", "purchase", "vendor", "sku", "basket"
        ],
        "finance": [
            "account", "transaction", "payment", "invoice", "credit",
            "debit", "balance", "loan", "interest", "cash", "bank"
        ],
    }
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> DomainMetadata:
        """
        Convert dict to validated DomainMetadata.
        """
        if isinstance(metadata, DomainMetadata):
            return metadata
        
        if not isinstance(metadata, dict):
            return DomainMetadata()
        
        # Normalize domain hints
        domain_hint = metadata.get("domain_hint") or metadata.get("business_domain")
        if domain_hint:
            domain_hint = domain_hint.strip().lower()
        
        return DomainMetadata(
            business_domain=metadata.get("business_domain"),
            domain_hint=domain_hint,
            healthcare_subdomain=metadata.get("healthcare_subdomain"),
            dataset_purpose=metadata.get("dataset_purpose"),
            start_date=metadata.get("start_date"),
            end_date=metadata.get("end_date"),
            expected_volume=metadata.get("expected_volume"),
            description=metadata.get("description"),
        )
    
    @staticmethod
    def detect_domain_from_purpose(purpose: str) -> Optional[str]:
        """
        Detect likely domain from dataset_purpose string.
        """
        if not purpose:
            return None
        
        purpose_lower = purpose.strip().lower()
        
        for domain, keywords in DomainContextValidator.DOMAIN_KEYWORDS.items():
            if any(kw in purpose_lower for kw in keywords):
                return domain
        
        return None


# =====================================================
# EXAMPLE USAGE FOR USERS
# =====================================================
HEALTHCARE_METADATA_EXAMPLES = {
    "hospital": {
        "business_domain": "healthcare",
        "healthcare_subdomain": "hospital",
        "dataset_purpose": "Hospital operations and inpatient length of stay analysis",
        "description": "Daily hospital admission/discharge records with patient demographics",
    },
    "clinic": {
        "business_domain": "healthcare",
        "healthcare_subdomain": "clinic",
        "dataset_purpose": "Ambulatory clinic visits and provider utilization",
        "description": "Clinic appointment and no-show data",
    },
    "pharmacy": {
        "business_domain": "healthcare",
        "healthcare_subdomain": "pharmacy",
        "dataset_purpose": "Pharmaceutical spend and prescription analysis",
        "description": "Prescription fill records and medication costs",
    },
    "diagnostics": {
        "business_domain": "healthcare",
        "healthcare_subdomain": "diagnostics",
        "dataset_purpose": "Lab and diagnostic test turnaround time analysis",
        "description": "Test order and result delivery records",
    },
}
