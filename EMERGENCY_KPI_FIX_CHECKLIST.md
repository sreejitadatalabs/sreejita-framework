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
