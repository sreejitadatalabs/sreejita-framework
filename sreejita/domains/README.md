# Sreejita Framework - Domain Modules

## Overview

Domain modules extend the **Sreejita Core Engine** with specialized KPIs, preprocessing, and insights for specific business contexts.

**Architecture**: Core Engine + Pluggable Domain Modules

Instead of hard-coding domain logic, we use a plugin architecture where each domain implements a standard interface (`BaseDomain`) and plugs into the core.

## Available Domains

### 1. **Retail** (`retail.py`)
For retail analytics, sales, and inventory data.

**Supports:**
- Sales & revenue metrics
- Product performance
- Customer transaction analysis
- Inventory insights
- Seasonal trends

**Example:**
```python
from sreejita.domains import RetailDomain

domain = RetailDomain()
results = domain.run(df)  # DataFrame with sales data
```

---

### 2. **E-commerce** (`ecommerce.py`)
For online retail, orders, and conversion analysis.

**Supports:**
- Conversion rate analysis
- Cart metrics
- Customer lifetime value
- Product performance
- Channel attribution

**Example:**
```python
from sreejita.domains import EcommerceDomain

domain = EcommerceDomain()
results = domain.run(df)  # DataFrame with order data
```

---

### 3. **Customer** (`customer.py`)
For customer segmentation, profiling, and behavior analysis.

**Supports:**
- RFM analysis (Recency, Frequency, Monetary)
- Customer segmentation
- Churn prediction
- Lifetime value
- Engagement scoring

**Example:**
```python
from sreejita.domains import CustomerDomain

domain = CustomerDomain()
results = domain.run(df)  # DataFrame with customer data
```

---

### 4. **Text/NLP** (`text.py`)
Adapter for analyzing preprocessed text features.

**Important:** This domain expects **preprocessed text features**, not raw text.

**Input Features:**
- Sentiment scores
- Topic distributions
- Polarity & subjectivity
- Word frequencies
- Text embeddings

**Workflow:**
```
Raw Text → NLP Pipeline (spaCy/Transformers) → Features → TextDomain → Sreejita Core
```

**Example:**
```python
from sreejita.domains import TextDomain

# First: preprocess text with external NLP tool
# Then: pass features to TextDomain
domain = TextDomain()
results = domain.run(df)  # DataFrame with text features
```

---

### 5. **Finance** (`finance.py`)
For financial analysis, P&L, and risk metrics.

**Supports:**
- Revenue & expense analysis
- Profit margins
- Cash flow metrics
- Financial ratios
- Risk & volatility
- Forecasting signals

**Example:**
```python
from sreejita.domains import FinanceDomain

domain = FinanceDomain()
results = domain.run(df)  # DataFrame with financial data
```

---

## Domain Architecture

### Base Class: `BaseDomain`

All domains inherit from `BaseDomain` and implement:

```python
class BaseDomain(ABC):
    @abstractmethod
    def validate_data(self, df) -> bool:
        """Check if data is compatible with domain."""
        pass
    
    @abstractmethod
    def preprocess(self, df) -> DataFrame:
        """Domain-specific cleaning & preparation."""
        pass
    
    @abstractmethod
    def calculate_kpis(self, df) -> Dict[str, Any]:
        """Calculate domain-specific KPIs."""
        pass
    
    @abstractmethod
    def generate_insights(self, df, kpis) -> List[str]:
        """Generate domain-specific insights."""
        pass
    
    def run(self, df) -> Dict[str, Any]:
        """Execute full pipeline."""
        # Validates → Preprocesses → Calculates KPIs → Generates Insights
        pass
```

### Return Format

Each domain's `run()` method returns:

```python
{
    "domain": "retail",  # Domain name
    "description": "Retail Analytics Domain",
    "data": processed_df,  # Cleaned DataFrame
    "kpis": {...},  # Calculated metrics
    "insights": [...]  # Generated insights
}
```

---

## Using Domains with Sreejita Core

Domains integrate seamlessly with the core engine:

```python
from sreejita.domains import get_domain
from sreejita.core import ProfileData, GenerateInsights

# Load domain dynamically
domain = get_domain("retail")

# Run domain analysis
domain_results = domain.run(df)

# Pass to core for further analysis
profile = ProfileData(domain_results["data"])
insights = GenerateInsights(profile, domain_results["kpis"])

# Combined output
final_report = {
    **domain_results,
    "core_insights": insights
}
```

---

## Design Philosophy

✅ **Separation of Concerns**: Core handles tabular data, domains add context

✅ **Pluggable Architecture**: Add new domains without modifying core

✅ **Consistent Interface**: All domains follow the same pattern

✅ **Type Safety**: Full type hints for IDE support

✅ **Extensible**: Easy to add domain-specific preprocessing and KPIs

---

## Adding a New Domain

1. Create `new_domain.py` in this folder
2. Inherit from `BaseDomain`
3. Implement all abstract methods
4. Register in `__init__.py`

```python
# new_domain.py
from .base import BaseDomain

class MyDomain(BaseDomain):
    name = "mydomain"
    description = "My Domain Analytics"
    
    def validate_data(self, df):
        return len(df) > 0
    
    def preprocess(self, df):
        return df.copy()
    
    def calculate_kpis(self, df):
        return {"metric_1": 0, "metric_2": 0}
    
    def generate_insights(self, df, kpis):
        return [f"Insight 1", f"Insight 2"]
```

Then register in `__init__.py`:

```python
from .new_domain import MyDomain

DOMAIN_REGISTRY = {
    ...existing...,
    "mydomain": MyDomain,
}
```

---

## Status

✅ **v1.1 Complete**: 5 domain modules ready
- Retail
- E-commerce
- Customer
- Text/NLP
- Finance

🚀 **v1.2 Roadmap**: 
- Domain CLI integration
- Cross-domain analysis
- Domain templates

---

## Questions?

Refer to main README for framework overview.
