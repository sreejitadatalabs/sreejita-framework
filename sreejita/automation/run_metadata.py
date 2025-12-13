import json
from datetime import datetime

def create_run_metadata(input_files, config, output_dir):
    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": input_files,
        "config": config,
        "status": "started"
    }

    path = output_dir / "run.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta

