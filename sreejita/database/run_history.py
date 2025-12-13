"""Run History Database for v1.6"""

import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

class RunHistoryDB:
    """SQLite-backed run history database."""
    
    def __init__(self, db_path: str = "runs/history.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                input_file TEXT NOT NULL,
                status TEXT NOT NULL,
                rows_processed INTEGER,
                duration_seconds REAL,
                errors INTEGER DEFAULT 0,
                metrics TEXT,
                validation_result TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id_1 TEXT NOT NULL,
                run_id_2 TEXT NOT NULL,
                comparison_data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id_1) REFERENCES runs(run_id),
                FOREIGN KEY(run_id_2) REFERENCES runs(run_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_run(self, run_data: Dict[str, Any]) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO runs 
            (run_id, timestamp, input_file, status, rows_processed, 
             duration_seconds, errors, metrics, validation_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_data.get("run_id"),
            run_data.get("timestamp", datetime.utcnow().isoformat()),
            run_data.get("input_file"),
            run_data.get("status"),
            run_data.get("rows_processed", 0),
            run_data.get("duration_seconds", 0),
            run_data.get("errors", 0),
            json.dumps(run_data.get("metrics", {})),
            json.dumps(run_data.get("validation", {}))
        ))
        
        conn.commit()
        conn.close()
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "run_id": row[1],
                "timestamp": row[2],
                "input_file": row[3],
                "status": row[4],
                "rows_processed": row[5],
                "duration_seconds": row[6],
                "errors": row[7],
                "metrics": json.loads(row[8]) if row[8] else {},
                "validation": json.loads(row[9]) if row[9] else {}
            }
        return None
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "run_id": row[1],
                "timestamp": row[2],
                "status": row[4],
                "rows_processed": row[5],
                "duration_seconds": row[6],
                "errors": row[7]
            })
        
        return results
