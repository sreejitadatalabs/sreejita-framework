import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# --- v3.0 IMPORTS ---
from sreejita.reporting.hybrid import run as run_hybrid  # Fixed import path
from sreejita.domains.router import decide_domain
from sreejita.domains import registry

# Optional: If you have these utils, keep them. If not, standard print/logging is used below.
try:
    from sreejita.utils.logger import get_logger
    log = get_logger("batch-runner")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("batch-runner")

SUPPORTED_EXT = [".csv", ".xlsx", ".xls"]

def run_batch(
    input_folder: str,
    output_root: str = "runs",
    recursive: bool = False
) -> Dict[str, Any]:
    """
    v3.0 Batch Orchestrator:
    1. Ingests Files
    2. Detects Domain
    3. Runs Domain Engine (KPIs + Insights)
    4. Generates Hybrid Report
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / timestamp
    
    # Create directory structure
    vis_dir = run_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_folder)
    files = []
    
    # 1. Collect Files
    log.info(f"Scanning {input_path} for data files...")
    pattern = "**/*" if recursive else "*"
    for ext in SUPPORTED_EXT:
        files.extend(input_path.glob(f"{pattern}{ext}"))
        
    results: Dict[str, Dict[str, Any]] = {}
    processed_count = 0
    failed_count = 0

    # 2. Orchestrate Analysis
    for file_path in files:
        try:
            log.info(f"Processing: {file_path.name}")
            
            # Load Data
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Smart Detect
            decision = decide_domain(df)
            domain_name = decision.domain
            
            if domain_name == "unknown":
                log.warning(f"⚠️  Skipping {file_path.name} (Unknown Domain)")
                continue
                
            log.info(f"✅  Detected [{domain_name.upper()}] for {file_path.name}")
            
            # Get Engine
            domain_cls = registry.get_domain(domain_name)
            if not domain_cls:
                log.error(f"No engine registered for {domain_name}")
                continue
                
            engine = domain_cls()
            
            # Run Intelligence Pipeline
            df_clean = engine.preprocess(df)
            kpis = engine.calculate_kpis(df_clean)
            insights = engine.generate_insights(df_clean, kpis)
            recs = engine.generate_recommendations(df_clean, kpis)
            
            # Generate Visuals (Sandboxed per file)
            file_vis_dir = vis_dir / file_path.stem
            visuals = engine.generate_visuals(df_clean, file_vis_dir)
            
            # Aggregate Results
            if domain_name not in results:
                results[domain_name] = {
                    "kpis": kpis,
                    "insights": insights,
                    "recommendations": recs,
                    "visuals": visuals
                }
            else:
                # Merge logic for batch (Simple append for v3.0)
                results[domain_name]["insights"].extend(insights)
                results[domain_name]["recommendations"].extend(recs)
                results[domain_name]["visuals"].extend(visuals)
                
            processed_count += 1

        except Exception as e:
            log.error(f"❌ Failed to process {file_path.name}: {e}")
            failed_count += 1

    # 3. Generate Executive Report
    if results:
        log.info("📝 Compiling Executive Hybrid Report...")
        report_path = run_hybrid(
            domain_results=results,
            output_dir=run_dir,
            metadata={
                "Input Directory": str(input_path),
                "Files Analyzed": processed_count,
                "Failed Files": failed_count,
                "Batch ID": timestamp
            }
        )
        log.info(f"🚀 SUCCESS: Report saved to {report_path}")
    else:
        log.warning("⚠️  No valid data processed. No report generated.")

    return {
        "run_dir": str(run_dir),
        "processed": processed_count,
        "failed": failed_count
    }
