# Changelog
All notable changes to the **Sreejita Framework** will be documented in this file.

The project follows a practical, capability-based versioning approach.
Not every internal milestone is released, only meaningful, stable versions.

---
[v1.6.0] – Quality Assurance & Observability Release
=====================================================

### Added

- Data Quality Validation Framework (validator.py)
  - 6 comprehensive validation checks
  - Schema validation and anomaly detection
  - Detailed validation reports
  
- Data Profiling Engine (profiler.py)
  - Statistical analysis per column
  - Outlier detection (IQR method)
  - Missing value analysis
  
- Metrics Collection (metrics.py)
  - Execution time tracking
  - Memory profiling
  - Throughput measurement (rows/sec)
  
- Run History Database (run_history.py)
  - SQLite-backed run persistence
  - Run comparison capabilities
  - Audit trail logging
  
- Dry-Run Mode (--dry-run CLI flag)
  - Preview transformations without write
  - Validation-only execution
  
- Data Profiling CLI (--profile flag)
  - Generate statistical profiles
  - Detect outliers and anomalies

### Improved

- CLI with new validation and profiling options
- Observability for production deployments
- Run history for debugging and comparison
- Enterprise-grade data quality checks

### Notes

- Fully backward compatible with v1.5
- New modules are independent and optional
- Production-ready quality assurance layer
- Zero-dependency profiling (uses NumPy/Pandas only)

## [v1.5.0] – Automation & File-Watcher Release
### Added
- Batch processing via CLI (`--batch`)
- Folder-based automation for multiple datasets
- Real-time file-watcher (`--watch`) for auto-processing new files
- Deterministic run structure with timestamped run folders
- Centralized logging for automation workflows
- Config-driven automation support

### Improved
- Operational stability for repeated and scheduled runs
- Clear separation between analytics logic and automation logic
- Reusable automation layer without affecting core framework

### Notes
- No analytics or domain logic changed
- Fully backward compatible with v1.2 and v1.1
- Designed for freelancer and consulting workflows

---

## [v1.2.0] – Domain-Agnostic Analytics Foundation
### Added
- Domain adapter architecture (retail, ecommerce, customer, text)
- Structured-data–first design (rows × columns)
- Clean separation between core logic and domain logic
- Schema-aware analytics foundation

### Improved
- Framework extensibility across multiple business domains
- Reusability for different datasets without rewriting code

### Notes
- Raw unstructured data (text, images, audio) must be converted to features
- Acts as the architectural base for all future versions

---

## [v1.1.0] – Modular CLI Release
### Added
- Command-line interface (`sreejita`)
- Modular project structure (core, domains, reports, visuals)
- YAML-based configuration system
- Multiple report modes (hybrid, dynamic, executive)

### Improved
- Code maintainability and readability
- Clear separation of responsibilities across modules

---

## [v1.0.0] – Initial Stable Release
### Added
- Core data cleaning pipeline
- Exploratory Data Analysis (EDA)
- Automated visualizations
- PDF report generation
- Reusable analytics scripts

### Notes
- First stable version used as the baseline for all future releases

---

## Versioning Philosophy
- Minor versions represent **capability milestones**, not just code changes
- Internal experimentation may occur between releases
- Backward compatibility is prioritized whenever possible
