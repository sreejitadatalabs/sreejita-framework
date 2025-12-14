import sqlite3
import json
from pathlib import Path
from datetime import datetime

class RunHistoryDB:
    def __init__(self, db_path="runs/run_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    input_file TEXT,
                    status TEXT,
                    metadata TEXT
                )
            """)

    def log_run(self, input_file, status, metadata=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO runs VALUES (NULL, ?, ?, ?, ?)",
                (
                    datetime.utcnow().isoformat(),
                    input_file,
                    status,
                    json.dumps(metadata or {})
                )
            )
