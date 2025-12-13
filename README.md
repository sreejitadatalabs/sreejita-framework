# 🌿 Sreejita Framework

**Universal Data Analytics & Reporting Engine with Pluggable Domain Modules**

## 🎯 Current Version: v1.6 (Quality Assurance & Observability Release)
The **Sreejita Framework** is a production-ready framework that transforms raw data into clean, analyzed insights with a standard, repeatable workflow.

### What's in v1.6?
Core Engine + **5 Domain Modules** + **Quality Assurance Suite** (Data Validation, Profiling, Observability)
- ✅ **Core Engine**: Data loading, cleaning, profiling, insights
- ✅ **Retail Domain**: Sales, inventory, customer behavior
- ✅ **E-commerce Domain**: Conversions, cart metrics, CLV
- ✅ **Customer Domain**: Segmentation, RFM, churn analysis
- ✅ **Text Domain**: NLP feature analysis, sentiment
- ✅ **Finance Domain**: P&L, ratios, volatility, forecasting
- - ✅ **Data Quality Validator**: 6 comprehensive validation checks
- ✅ **Data Profiler**: Statistical analysis with outlier detection
- ✅ **Dry-Run Mode**: Preview transformations without writing
- ✅ **Metrics Collector**: Execution time & memory tracking
- ✅ **Run History Database**: Audit trail & run comparisons

---

## 📦 Installation

```bash
git clone https://github.com/sreejitadatalabs/sreejita-framework.git
cd sreejita-framework
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Using Domain Modules

```python
from sreejita import get_domain
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Load domain (retail, ecommerce, customer, text, finance)
domain = get_domain('retail')
results = domain.run(df)

print(f"Domain: {results['domain']}")
print(f"KPIs: {results['kpis']}")
print(f"Insights: {results['insights']}")
```

### Using Core Engine

```python
from sreejita import DataCleaner, InsightGenerator

cleaner = DataCleaner()
df_clean = cleaner.clean(df)

insight_gen = InsightGenerator()
insights = insight_gen.generate(df_clean)
```

---

## 📁 Folder Structure

```
sreejita-framework/
├── sreejita/
│   ├── core/              # Core engine (cleaner, insights, KPIs)
│   ├── domains/           # Pluggable domain modules ✨ NEW IN v1.2
│   │   ├── base.py
│   │   ├── retail.py
│   │   ├── ecommerce.py
│   │   ├── customer.py
│   │   ├── text.py
│   │   ├── finance.py
│   │   └── README.md
│   ├── config/            # Configuration loader
│   ├── utils/             # Utility functions
│   ├── visuals/           # Visualization helpers
│   ├── reports/           # Report generation
│   └── __init__.py        # Main API (updated for v1.2)
├── examples/              # Example notebooks
├── requirements.txt
└── README.md
```

---

## 📊 What Data Can This Framework Handle?

Sreejita Framework supports **any structured dataset**
(rows × columns), including:

- **Retail & ecommerce transactions** - Sales, inventory, customer behavior
- **Sales & revenue data** - Revenue metrics, sales forecasts, trends
- **Customer metrics & segmentation outputs** - Customer profiles, RFM analysis, cohorts
- **Marketing campaign data** - Campaign performance, engagement metrics, conversions
- **Text analytics outputs** - Sentiment scores, topic classifications, text features

### What It CANNOT Handle:

⚠️ **Raw unstructured data** (text, images, audio) must be converted into **structured features** before use.

- Raw text → Extract sentiment scores, embeddings, topics
- Images → Extract features, classifications, metadata
- Audio → Extract transcripts, emotions, speech features

This design protects you from misuse and ensures data quality.

## 🏗️ Architecture: Core + Domains

Instead of a monolithic framework, Sreejita uses a **plugin architecture**:

```
┌─────────────────────────────────────────┐
│      Sreejita Core Engine               │
│   (Tabular data processing)             │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┬──────────┬──────────┬────────┐
    ▼                 ▼          ▼          ▼        ▼
┌────────┐      ┌──────────┐ ┌────────┐ ┌──────┐ ┌────────┐
│Retail  │      │E-commerce│ │Customer│ │Text  │ │Finance │
│Domain  │      │Domain    │ │Domain  │ │Domain│ │Domain  │
└────────┘      └──────────┘ └────────┘ └──────┘ └────────┘
```

✅ Each domain is **independent**  
✅ Core engine stays **stable**  
✅ Add new domains **without modifying core**  
✅ Domains **share consistent interface**  

---

## 💡 Why This Architecture?

| Feature | Benefit |
|---------|----------|
| **Plugin Pattern** | Add domains without touching core |
| **Separation of Concerns** | Core handles data, domains add context |
| **Type Safe** | Full type hints for IDE support |
| **Extensible** | Create custom domains in minutes |
| **Production-Grade** | Used in Spark, Airflow, sklearn |

---

## 📚 Available Domains

### Retail Domain
- Sales & revenue metrics
- Product performance
- Inventory insights
- Seasonal trends

### E-commerce Domain
- Conversion rates
- Cart metrics
- Customer lifetime value
- Channel attribution

### Customer Domain
- RFM analysis
- Customer segmentation
- Churn prediction
- Engagement scoring

### Text Domain
- Sentiment analysis
- Topic extraction
- Word frequencies
- **Note**: Expects preprocessed features, not raw text

### Finance Domain
- P&L analysis
- Cash flow metrics
- Financial ratios
- Volatility & risk

👉 See [sreejita/domains/README.md](sreejita/domains/README.md) for detailed domain documentation.

---

## 🛠️ CLI Usage

```bash
python -m sreejita.cli -i data.csv -o report.json -p retail
python -m sreejita.cli -i orders.csv -o report.json -p ecommerce
python -m sreejita.cli -i customers.csv -o report.json -p customer --ml
```

---
## Supported Data Types

Sreejita Framework supports any **structured or semi-structured data**
that can be represented as a table (rows × columns), including:

- Retail & ecommerce transactions
- Sales & revenue data
- Customer profiles & metrics
- Marketing campaign data
- Text analytics outputs (sentiment, topics, scores)

❌ Not supported directly:
- Raw text (must be converted to features)
- Images, audio, video
- Streaming data
-----------------

## 📈 Version Roadmap

| Version | Status | Features |
|---------|--------|----------|
| **v1.0** | ✅ Complete | Core engine, configs, utils |
| **v1.1** | ✅ Complete | CLI, enhanced validation |
| **v1.2** | ✅ **CURRENT** | **Domain modules (5 domains)** |
| v1.5 | 🔜 Planned | Automation, scheduling |
| v2.0 | 🔜 Planned | Streamlit UI, dashboards |
| v3.0 | 🔜 Planned | AI-powered insights |
| v4.0 | 🔜 Planned | SaaS platform |

---

## 🎓 Learn More

- **Domains Guide**: [sreejita/domains/README.md](sreejita/domains/README.md)
- **Core API**: [sreejita/core/](sreejita/core/)
- **Examples**: [examples/](examples/)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

Yeswanth Arasavalli  
🔗 [Portfolio](https://yeswantharasavalli.me) | 🔗 [GitHub](https://github.com/sreejitadatalabs)  
📧 Contact: [LinkedIn](https://linkedin.com)
