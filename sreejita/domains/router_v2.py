"""
Router v2.0 — Healthcare Domain-Hint Aware
Registry-based routing with explicit domain hint support.
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
    Detect domain with optional explicit hints and non-strict fallback.
    
    CRITICAL: If user explicitly selects a domain via UI (domain_hint),
    we TRUST that hint and BYPASS column detection.
    
    Args:
        df: Input DataFrame
        domain_hint: Explicit domain name (e.g., "healthcare") from UI
        strict: If True, only accept high-confidence detections
    
    Returns:
        DomainDetectionResult object
    """
    
    # ============================================================
    # STEP 1: EXPLICIT DOMAIN HINT (USER OVERRIDE)
    # ============================================================
    # If user explicitly selected a domain in UI, honor that!
    # Do NOT fall through to auto-detection.
    if domain_hint:
        hint_lower = domain_hint.lower().strip()
        
        # User selected healthcare explicitly
        if hint_lower in ["healthcare", "hospital", "clinic", "diagnostics", "pharmacy", "public_health"]:
            return DomainDetectionResult(
                domain="healthcare",
                confidence=0.95,  # High confidence because user selected it explicitly
                signals={"user_hint": domain_hint}
            )
        
        # User selected retail
        if hint_lower in ["retail", "ecommerce"]:
            return DomainDetectionResult(
                domain=hint_lower,
                confidence=0.95,
                signals={"user_hint": domain_hint}
            )
        
        # User selected finance
        if hint_lower == "finance":
            return DomainDetectionResult(
                domain="finance",
                confidence=0.95,
                signals={"user_hint": domain_hint}
            )
    
    # ============================================================
    # STEP 2: AUTO-DETECTION (FALLBACK)
    # ============================================================
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
    
    # ============================================================
    # STEP 3: HEALTHCARE FALLBACK (MINIMAL SIGNAL CHECK)
    # ============================================================
    # If detection failed or confidence is too low,
    # check for minimal healthcare signals as last resort
    if best_result is None or best_result.confidence < 0.40:
        healthcare_signals = {c.lower() for c in df.columns} & HEALTHCARE_KEYWORDS
        
        if len(healthcare_signals) >= 2:
            return DomainDetectionResult(
                domain="healthcare",
                confidence=0.35,  # Safe fallback confidence
                signals={"healthcare_signal_count": len(healthcare_signals)}
            )
    
    # ============================================================
    # STEP 4: RETURN BEST DETECTION
    # ============================================================
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
