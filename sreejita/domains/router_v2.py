"""
Router v2.0

Registry-based routing.
Does NOT import domain modules directly.
"""

from typing import Optional
from sreejita.domains.registry import registry
from sreejita.domains.contracts import DomainDetectionResult


# Define healthcare keywords to identify healthcare datasets
HEALTHCARE_KEYWORDS = {
    'patient_id', 'pid', 'patient', 'admission', 'discharge', 
    'length_of_stay', 'los', 'diagnosis', 'diagnosis_code', 
    'icd_code', 'procedure', 'cpt_code', 'facility', 'hospital',
    'clinic', 'provider', 'physician', 'doctor', 'nurse',
    'treatment', 'medication', 'drug', 'cost', 'charge',
    'billing', 'claim', 'insurance', 'medical', 'health'
}


def detect_domain(df, domain_hint=None, strict=False):
    """
    Detect domain with optional hints and non-strict fallback.
    
    Args:
        df: Input DataFrame
        domain_hint: Explicit domain name (e.g., "healthcare")
        strict: If True, only accept detections with high confidence
        
    Returns:
        DomainDetectionResult or None
    """
    
    # If user provides domain hint, use that directly
    if domain_hint and domain_hint.lower() == "healthcare":
        return DomainDetectionResult(
            domain="healthcare",
            confidence=1.0,
            signals={}
        )
    
    # Try detection with all registered domain detectors
    best_result = None
    best_confidence = 0.0
    
    for domain_name in registry.list_domains():
        try:
            detector = registry.get_detector(domain_name)
            if detector is None:
                continue
            
            result = detector.detect(df)
            if result and result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result
        except Exception:
            # Skip detectors that fail
            continue
    
    # If detection fails and we have minimal healthcare signals, use healthcare
    if best_result is None or best_result.confidence < 0.40:
        healthcare_signals = {c.lower() for c in df.columns} & HEALTHCARE_KEYWORDS
        
        if len(healthcare_signals) >= 2:
            return DomainDetectionResult(
                domain="healthcare",
                confidence=0.35,  # Safe fallback confidence
                signals={"healthcare_signal_count": len(healthcare_signals)}
            )
    
    # Return best detection result (or None)
    return best_result


def apply_domain(df):
    """
    Apply domain preprocessing using detected domain.
    """
    result = detect_domain(df)
    
    # v1.x safety: no detection → return df unchanged
    if result is None or result.domain is None:
        return df
    
    # Get the domain instance from registry
    domain = registry.get_domain(result.domain)
    
    if domain is None:
        # Domain not registered (safe fallback)
        return df
    
    # Apply domain-specific preprocessing
    try:
        return domain.preprocess(df)
    except Exception:
        # Domain logic must never crash pipeline
        return df
