# EMERGENCY KPI Fix Checklist - Production Errors

## Status: CRITICAL - All Domains Failing

**Error Pattern**: KeyError on domain-specific hardcoded column names
- retail: KeyError 'sales'
- healthcare: KeyError 'outcome_score'
- operations: RuntimeError (cascading from KPI failure)

**Root Cause**: Fix #4 incomplete - only retail.kpis.py updated, others still have hardcoded columns

---

## Quick Fix Instructions

### Use This for ALL Domains:

```python
from sreejita.reporting.kpi_base_flexible import FlexibleKPIEngine, compute_kpis_flexible
from sreejita.reporting.utils import safe_mean, safe_sum, safe_ratio
```

### Domain-Specific Configs:

#### 1. Healthcare KPIs
**File**: `sreejita/reporting/healthcare/kpis.py`

```python
from sreejita.reporting.kpi_base_flexible import compute_kpis_flexible

def compute_healthcare_kpis(df):
    config = {
        'outcome_score': ['outcome_score', 'score', 'health_score'],
        'readmitted': ['readmitted', 'readmission', 'rehospitalized'],
        'length_of_stay': ['length_of_stay', 'los', 'days_admitted'],
    }
    return compute_kpis_flexible(df, config)
```

#### 2. Operations KPIs
**File**: `sreejita/reporting/ops/kpis.py`

```python
from sreejita.reporting.kpi_base_flexible import compute_kpis_flexible

def compute_ops_kpis(df):
    config = {
        'cycle_time': ['cycle_time', 'processing_time', 'duration'],
        'on_time': ['on_time', 'on_schedule', 'on_time_delivery'],
        'delay': ['delay', 'delay_days', 'late'],
    }
    return compute_kpis_flexible(df, config)
```

#### 3. Finance KPIs
**File**: `sreejita/reporting/finance/kpis.py`

```python
from sreejita.reporting.kpi_base_flexible import compute_kpis_flexible

def compute_finance_kpis(df):
    config = {
        'revenue': ['revenue', 'sales', 'income', 'total_revenue'],
        'cost': ['cost', 'expense', 'total_cost'],
        'profit': ['profit', 'net_income', 'earnings'],
    }
    return compute_kpis_flexible(df, config)
```

#### 4. Customer KPIs
**File**: `sreejita/reporting/customer/kpis.py`

```python
from sreejita.reporting.kpi_base_flexible import compute_kpis_flexible

def compute_customer_kpis(df):
    config = {
        'customer_id': ['customer_id', 'id', 'cust_id'],
        'revenue': ['revenue', 'customer_revenue', 'lifetime_value'],
        'purchase_count': ['purchase_count', 'num_purchases', 'orders'],
    }
    return compute_kpis_flexible(df, config)
```

#### 5. Marketing KPIs
**File**: `sreejita/reporting/marketing/kpis.py`

```python
from sreejita.reporting.kpi_base_flexible import compute_kpis_flexible

def compute_marketing_kpis(df):
    config = {
        'campaign': ['campaign', 'campaign_id', 'campaign_name'],
        'converted': ['converted', 'conversion', 'converted_flag'],
        'cost': ['cost', 'spend', 'campaign_cost'],
        'clicks': ['clicks', 'click_count'],
        'impressions': ['impressions', 'impression_count'],
    }
    return compute_kpis_flexible(df, config)
```

#### 6. E-commerce KPIs
**File**: `sreejita/reporting/ecommerce/kpis.py`

```python
from sreejita.reporting.kpi_base_flexible import compute_kpis_flexible

def compute_ecommerce_kpis(df):
    config = {
        'revenue': ['revenue', 'order_value', 'total_spend'],
        'quantity': ['quantity', 'units_sold', 'items'],
        'conversion': ['conversion', 'converted', 'conversion_flag'],
    }
    return compute_kpis_flexible(df, config)
```

---

## Priority Order

1. ✅ Retail - DONE
2. 🔴 Healthcare - CRITICAL (currently failing)
3. 🔴 Operations - CRITICAL (currently failing)
4. 🟡 Finance - MEDIUM
5. 🟡 Customer - MEDIUM
6. 🟡 Marketing - MEDIUM
7. 🟡 Ecommerce - MEDIUM

---

## Verification

After fixing each, test with:

```bash
# Test healthcare
python -c "from sreejita.reporting.healthcare.kpis import compute_healthcare_kpis; import pandas as pd; df = pd.DataFrame({'outcome_score': [0.8, 0.9], 'readmitted': [0, 1]}); print(compute_healthcare_kpis(df))"

# Test ops
python -c "from sreejita.reporting.ops.kpis import compute_ops_kpis; import pandas as pd; df = pd.DataFrame({'cycle_time': [5.2, 6.1], 'on_time': [0, 1], 'delay': [2, 0]}); print(compute_ops_kpis(df))"
```

---

## Key Changes

- Use `FlexibleKPIEngine.find_column()` for case-insensitive matching
- Pass config dict with alternative column names
- All 7 domains now support multiple naming conventions
- No more KeyError on missing columns
- 
## RETAIL BIAS AUDIT & FIXES - COMPREHENSIVE CHECKLIST

This section outlines the systematic audit process for detecting and fixing retail bias across all KPI domains in v2.9.

### What is Retail Bias?
Retail bias occurs when KPI calculations systematically favor certain product categories, customer segments, or regions due to:
- Hardcoded column assumptions (e.g., 'sales' column may not exist)
- Inconsistent metric definitions across domains
- Missing null/zero handling in aggregations
- Segment-specific filtering that excludes valid data

### Audit Scope - All 7 Domains
1. **Retail** - Product sales, revenue, margins by category
2. **Healthcare** - Patient outcomes, readmission rates, length of stay
3. **Operations** - Cycle time, on-time delivery, processing delays
4. **Finance** - Revenue, costs, profit margins by business unit
5. **Customer** - Customer lifetime value, purchase patterns, segmentation
6. **Marketing** - Campaign ROI, conversion rates, click-through rates
7. **Ecommerce** - Order value, conversion rates, product performance

---

## AUDIT CHECKLIST - Phase 1: Column Discovery

### Phase 1A: Identify All Possible Column Names

For each KPI metric, document all possible column name variations:

```python
# RETAIL DOMAIN
RETAIL_KPI_VARIANTS = {
    'sales': ['sales', 'revenue', 'total_sales', 'order_value'],
    'quantity': ['quantity', 'units_sold', 'qty', 'units'],
    'category': ['category', 'product_category', 'cat', 'dept'],
}

# HEALTHCARE DOMAIN
HEALTHCARE_KPI_VARIANTS = {
    'outcome_score': ['outcome_score', 'score', 'health_score'],
    'readmitted': ['readmitted', 'readmission', 'rehospitalized'],
    'length_of_stay': ['length_of_stay', 'los', 'days_admitted'],
}
```

---

## AUDIT CHECKLIST - Phase 2: Testing All Domains

### Phase 2A: Automated Column Matching Test

Test that FlexibleKPIEngine correctly identifies columns:

```bash
# Test Retail
python -c "from sreejita.reporting.retail.kpis import compute_retail_kpis; import pandas as pd; df = pd.DataFrame({'revenue': [100, 200], 'units': [5, 10]}); print(compute_retail_kpis(df))"

# Test Healthcare  
python -c "from sreejita.reporting.healthcare.kpis import compute_healthcare_kpis; import pandas as pd; df = pd.DataFrame({'score': [0.8, 0.9], 'readmission': [0, 1]}); print(compute_healthcare_kpis(df))"

# Test Operations
python -c "from sreejita.reporting.ops.kpis import compute_ops_kpis; import pandas as pd; df = pd.DataFrame({'processing_time': [5.2, 6.1], 'on_schedule': [0, 1], 'late': [2, 0]}); print(compute_ops_kpis(df))"
```

### Phase 2B: Domain Coverage Checklist

- [ ] **Retail**: retail.kpis.py updated with FlexibleKPIEngine
- [ ] **Healthcare**: healthcare.kpis.py updated with FlexibleKPIEngine
- [ ] **Operations**: ops.kpis.py updated with FlexibleKPIEngine  
- [ ] **Finance**: finance.kpis.py updated with FlexibleKPIEngine
- [ ] **Customer**: customer.kpis.py updated with FlexibleKPIEngine
- [ ] **Marketing**: marketing.kpis.py updated with FlexibleKPIEngine
- [ ] **Ecommerce**: ecommerce.kpis.py updated with FlexibleKPIEngine

---

## AUDIT CHECKLIST - Phase 3: Bias Detection & Fixes

### Phase 3A: Statistical Bias Analysis

Bias occurs when aggregations systematically exclude or underweight certain segments:

```python
# Check for segment-wise bias
def check_bias(df, metric_col, segment_col):
    """Check if metric calculation varies significantly by segment"""
    grouped = df.groupby(segment_col)[metric_col].agg(['mean', 'std', 'count'])
    print(grouped)
    # Flag if any segment has < 10% of data but > 50% different average
    
# Example: Check if high-value customers are counted differently
df_retail = pd.read_csv('retail_data.csv')
check_bias(df_retail, 'revenue', 'customer_segment')
```

### Phase 3B: Common Bias Patterns

1. **Column Naming Bias**: Hardcoded 'sales' column fails on 'revenue' dataset
2. **Null Handling Bias**: Some domains drop nulls, others impute to zero
3. **Segment Filtering Bias**: Excluding low-value segments from calculations
4. **Time Period Bias**: Using different date columns leads to different results
5. **Data Type Bias**: String vs numeric representations of same metric

### Phase 3C: Fix Implementation Checklist

```bash
# Step 1: Apply flexible column matching
cd sreejita/reporting
for domain in retail healthcare operations finance customer marketing ecommerce; do
    echo "Updating $domain/kpis.py..."
    # Replace hardcoded column names with FlexibleKPIEngine
done

# Step 2: Run comprehensive tests
python -m pytest tests/reporting/test_kpis_flexible.py -v

# Step 3: Validate against sample data
python scripts/validate_kpis_all_domains.py
```

---

## FINAL VERIFICATION CHECKLIST

- [ ] All 7 domains use FlexibleKPIEngine
- [ ] No hardcoded column names remain
- [ ] All column variants documented
- [ ] Tests pass for all domains
- [ ] Bias analysis shows no systematic exclusions
- [ ] Documentation updated with variant column names
- [ ] Performance benchmarked (no degradation)
- [ ] Ready for production deployment

**Version**: v2.9 - Emergency KPI Fix
**Status**: Audit and fix process initiated
**Last Updated**: 2025-01-15
