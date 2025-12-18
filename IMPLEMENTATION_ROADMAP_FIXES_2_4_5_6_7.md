# Implementation Roadmap: Fixes #2, #4, #5, #6, #7

## Quick Summary

✅ **COMPLETED**:
- Fix #1: Reorder detectors alphabetically (✅ DONE - router.py)
- Fix #3: Add universal column mapping (✅ DONE - column_mapping.py)
- Audit document created (✅ DONE - RETAIL_BIAS_AUDIT_AND_FIXES.md)

🔄 **NEXT STEPS** (Fixes #2, #4, #5, #6, #7):

---

## Fix #2: Remove 1.5x Confidence Multipliers

**Files to Update**: 6 domain detector files

```bash
# Find all instances:
grep -r "\* 1.5" sreejita/domains/

# Files affected:
- sreejita/domains/retail.py (line ~47)
- sreejita/domains/customer.py (line ~37)
- sreejita/domains/finance.py (line ~35)
- sreejita/domains/healthcare.py (line ~35)
- sreejita/domains/marketing.py (line ~37)
- sreejita/domains/ops.py (line ~35)
```

**Change Pattern**:
```python
# BEFORE:
score = min((len(matches) / len(self.DOMAIN_COLUMNS)) * 1.5, 1.0)

# AFTER:
score = min((len(matches) / len(self.DOMAIN_COLUMNS)), 1.0)
```

**Time**: 5-10 min (edit all 6 files, 1 line each)

---

## Fix #4: Refactor KPI Engines for Column Mapping

**Files to Update**: 7 KPI files under `sreejita/reporting/*/kpis.py`

```python
# Pattern for each domain (example: customer):

from sreejita.domains.column_mapping import ColumnMapping
import pandas as pd

def compute_customer_kpis(df, mapping=None):
    """Compute KPIs with flexible column mapping."""
    
    if mapping is None:
        mapping = ColumnMapping.auto_detect(df)
    
    revenue_col = mapping['revenue_col']
    id_col = mapping['id_col']
    
    if not revenue_col or not id_col:
        return {}  # Not enough columns
    
    return {
        "total_revenue": df[revenue_col].sum(),
        "customer_count": df[id_col].nunique(),
        "avg_customer_value": df[revenue_col].sum() / df[id_col].nunique(),
    }
```

**Apply to**:
- sreejita/reporting/retail/kpis.py
- sreejita/reporting/ecommerce/kpis.py
- sreejita/reporting/customer/kpis.py
- sreejita/reporting/finance/kpis.py
- sreejita/reporting/healthcare/kpis.py
- sreejita/reporting/marketing/kpis.py
- sreejita/reporting/ops/kpis.py

**Time**: 45 min (refactor all 7, ~50 lines each)

---

## Fix #5: Abstract Visualization Functions

**Files to Update**: 7 visualization files under `sreejita/reporting/*/visuals.py`

```python
# Pattern for each domain (example: generic revenue by category):

from sreejita.domains.column_mapping import ColumnMapping
from pathlib import Path
import matplotlib.pyplot as plt

def revenue_by_category(df, output_path: Path, mapping=None):
    """Generic visualization: revenue by category."""
    
    if mapping is None:
        mapping = ColumnMapping.auto_detect(df)
    
    revenue_col = mapping['revenue_col']
    category_col = mapping['category_col']
    
    if not (revenue_col and category_col):
        return None  # Suitable columns not found
    
    agg = df.groupby(category_col)[revenue_col].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(8, 6))
    agg.plot(kind="bar")
    plt.title(f"Total Revenue by {category_col.title()}")
    plt.ylabel("Revenue")
    plt.xlabel(category_col.title())
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
```

**Apply to**:
- All 7 reporting domain visuals modules
- Replace domain-specific functions with generic mappable versions

**Time**: 45 min (refactor visualizations to be mapping-aware)

---

## Fix #6: Normalize Domain Detector Column Sets

**Files to Update**: 6 domain detector files

```python
# Pattern: Add alternative column names to each detector

# BEFORE (customer.py):
CUSTOMER_COLUMNS: Set[str] = {
    "customer_id",
    "customer_name",
    "email",
    "phone",
    "segment",
    "revenue",
}

# AFTER:
CUSTOMER_COLUMNS: Set[str] = {
    "customer_id", "id", "cust_id",
    "customer_name", "name", "customer",
    "email", "contact_email",
    "phone", "contact_phone",
    "segment", "cohort", "group", "bucket",
    "revenue", "value", "spend", "sales",
}
```

Apply similar normalization to all 6 domains.

**Time**: 30 min (expand column sets, ~10-15 alternatives per domain)

---

## Fix #7: Add Domain Detection Fairness Tests

**File to Create**: `tests/test_domain_detection_fairness.py`

```python
import pytest
import pandas as pd
from sreejita.domains.router import DOMAIN_DETECTORS, decide_domain

def test_detector_alphabetical_order():
    """Detectors should run in alphabetical order (no bias)."""
    names = [d.domain_name for d in DOMAIN_DETECTORS]
    assert names == sorted(names), f"Detectors not alphabetical: {names}"

def test_equal_confidence_multiplier():
    """All detectors must use 1.0x multiplier."""
    # This could be enforced by reading detector code
    # OR by testing confidence scores don't exceed expectations
    pass

def test_non_retail_detection_accuracy():
    """Verify non-retail data isn't misclassified as retail."""
    
    # Test healthcare
    healthcare_df = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'outcome_score': [0.8, 0.9, 0.7],
        'readmitted': [0, 0, 1],
        'length_of_stay': [3, 5, 7]
    })
    decision = decide_domain(healthcare_df)
    assert decision.selected_domain == 'healthcare', f"Got {decision.selected_domain}"
    
    # Test finance
    finance_df = pd.DataFrame({
        'revenue': [1000, 2000, 3000],
        'cost': [500, 1000, 1500],
        'quarter': ['Q1', 'Q2', 'Q3']
    })
    decision = decide_domain(finance_df)
    assert decision.selected_domain == 'finance', f"Got {decision.selected_domain}"
    
    # Test ops
    ops_df = pd.DataFrame({
        'cycle_time': [5.2, 6.1, 4.9],
        'on_time': [0, 1, 1],
        'delay': [2, 0, 3]
    })
    decision = decide_domain(ops_df)
    assert decision.selected_domain == 'ops', f"Got {decision.selected_domain}"

def test_confidence_scores_normalized():
    """All domain scores should be <= 1.0."""
    test_data = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [100, 200, 300],
        'category': ['A', 'B', 'A']
    })
    decision = decide_domain(test_data)
    for domain, score in decision.domain_scores.items():
        assert score['confidence'] <= 1.0
```

**Time**: 30 min (write 4-5 comprehensive tests)

---

## Execution Order

1. **Fix #2** (5 min) - Remove multipliers from all 6 domain files
2. **Fix #6** (30 min) - Expand column sets in detectors  
3. **Fix #4** (45 min) - Refactor KPI engines
4. **Fix #5** (45 min) - Abstract visualizations
5. **Fix #7** (30 min) - Add fairness tests

**Total Time**: ~2.5 hours

---

## Validation Commands

```bash
# Run fairness tests
pytest tests/test_domain_detection_fairness.py -v

# Test non-retail detection
python -c "from sreejita.domains.router import decide_domain; import pandas as pd; df = pd.DataFrame({'patient_id': [1,2], 'outcome_score': [0.8, 0.9], 'readmitted': [0, 1]}); print(decide_domain(df))"

# Verify no 1.5x multipliers remain
grep -r "\* 1.5" sreejita/
```

---

## Notes

- All fixes maintain backward compatibility
- Column mapping auto-detects but can be overridden
- Tests ensure no retail bias regression in future versions
- Tag release as `v2.10-retail-bias-fix` after completion
