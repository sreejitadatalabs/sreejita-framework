![CI](https://github.com/sreejitadatalabs/sreejita-framework/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

# 🌿 Sreejita Framework

**Universal Data Analytics & Reporting Engine with Pluggable Domain Modules**

**Version:** 1.9.5
**Status:** Deterministic Decision Reports
**Maintained by:** Sreejita Data Labs

Sreejita Framework is a production-grade, domain-agnostic **data analytics automation framework** designed for freelancers, consultants, and small teams. It automates data cleaning, profiling, reporting, and batch workflows with a clean CLI and robust CI/CD.

---

## 🚀 Key Features

### Core Analytics
- Data cleaning & normalization
- Data quality validation
- Profiling & metrics
- Domain-ready architecture (retail, ecommerce, text, etc.)

### Reports
- Hybrid consulting-style PDF report
- Executive summary report
- Dynamic report generation

### Automation (v1.5+)
- Batch processing of folders
- File watcher (real-time ingestion)
- Scheduler (time-based automation)
- Retry & failure handling
- Deterministic run folders

### CLI (v1.6+)
- Lightweight CLI entry point
- Supports single file, batch, watch, and schedule modes
- Config-driven execution

### Engineering Quality (v1.7)
- Full CI/CD pipeline (GitHub Actions)
- Test suite for CLI, domains, automation
- Python version compatibility (3.9 – 3.12)
- Structured logging

---


## 🎯 Version: v1.6 (Quality Assurance & Observability Release)
The **Sreejita Framework** is a production-ready framework that transforms raw data into clean, analyzed insights with a standard, repeatable workflow.

### What's in v1.6?
Core Engine + **5 Domain Modules** + **Quality Assurance Suite** (Data Validation, Profiling, Observability)
- ✅ **Core Engine**: Data loading, cleaning, profiling, insights
- ✅ **Retail Domain**: Sales, inventory, customer behavior
- ✅ **E-commerce Domain**: Conversions, cart metrics, CLV
- ✅ **Customer Domain**: Segmentation, RFM, churn analysis
- ✅ **Text Domain**: NLP feature analysis, sentiment
- ✅ **Finance Domain**: P&L, ratios, volatility, forecasting
- ✅ **Data Quality Validator**: 6 comprehensive validation checks
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
│
├── sreejita/                      # Core framework package
│   ├── __init__.py
│   ├── __version__.py             # Version control
│   ├── cli.py                     # Command Line Interface (CLI)
│
│   ├── config/                    # Configuration system
│   │   ├── __init__.py
│   │   ├── defaults.py            # Default settings
│   │   └── loader.py              # Load & validate config.yaml
│
│   ├── core/                      # Core analytics logic
│   │   ├── __init__.py
│   │   ├── cleaner.py             # Data cleaning & preprocessing
│   │   ├── kpis.py                # KPI calculations
│   │   ├── insights.py            # Insight generation (rule-based)
│   │   ├── recommendations.py     # Business recommendations
│   │   └── schema.py              # Schema & column-type detection
│
│   ├── visuals/                   # Visualization engines
│   │   ├── __init__.py
│   │   ├── time_series.py         # Time-based trends
│   │   ├── distributions.py       # Numeric distributions
│   │   ├── categorical.py         # Categorical analysis
│   │   └── correlation.py         # Correlation heatmaps
│
│   ├── reports/                   # Report generators
│   │   ├── __init__.py
│   │   ├── hybrid.py              # Main automated PDF report
│   │   ├── executive.py           # Executive-level summaries
│   │   └── dynamic.py             # Config-driven reports
│
│   ├── automation/                # Automation & orchestration
│   │   ├── __init__.py
│   │   ├── batch_runner.py        # Batch processing
│   │   ├── file_watcher.py        # Folder monitoring
│   │   ├── scheduler.py           # Scheduled execution
│   │   ├── retry.py               # Retry & failure handling
│   │   └── run_metadata.py        # Run logs & metadata
│
│   ├── domains/                   # Domain routing (v2.x ready)
│   │   ├── __init__.py
│   │   ├── router.py              # Domain detection & routing
│   │   └── retail.py              # Retail-specific logic (example)
│
│   └── utils/                     # Utilities
│       ├── __init__.py
│       └── logger.py              # Centralized logging
│
├── tests/                         # Automated tests (CI)
│   ├── test_cli_smoke.py
│   ├── test_domains_import.py
│   ├── test_automation_import.py
│   ├── test_batch_runner.py
│   ├── test_file_watcher.py
│   └── test_scheduler.py
│
├── reports/                       # Generated output (runtime)
│   └── hybrid_report_YYYYMMDD.pdf
│
├── hybrid_images/                 # Generated charts (runtime)
│
├── examples/
│   └── config.yaml                # Example configuration
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI testing
│       └── package.yml            # Build & package verification
│
├── pyproject.toml                 # Packaging & metadata
├── requirements.txt               # Dependencies (if used)
├── README.md                      # Project overview
├── CHANGELOG.md                   # Version history
└── LICENSE

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
| **v1.2** | ✅ Complete | Domain modules (5 domains) |
| **v1.5** | ✅ Complete | Automation, scheduling |
| **v1.6** | ✅ Complete | Quality Assurance & Observability |
| **v1.7** | ✅ Complete |  Professional Quality & Developer Experience |
| **v1.8** | ✅ Complete | Packaging & Distribution Foundation |
| **v1.9.0** | ✅ Complete | Streamlit UI, dashboards |
| **v1.9.5** | ✅ Complete| Deterministic Decision Reports |
| **v1.9.6** | ✅ Complete | Narrative & Executive Safety |
| v1.9.7 | ✅ Complete | Evidence Snapshot (visual policy) |
| v1.9.8 | ✅ Complete | Executive Snapshot contract (full) |
| v1.9.9 | ✅ Current | **Prescriptive archetypes** |
| v2.0 | 🔜 Planned | Domain Intelligence |
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
