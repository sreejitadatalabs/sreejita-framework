"""Metrics Collection and Performance Tracking for v1.6"""

import time
import psutil
import os
from typing import Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class Metrics:
    """Runtime metrics container."""
    start_time: float
    end_time: float = None
    rows_processed: int = 0
    memory_start_mb: float = 0
    memory_end_mb: float = 0
    errors: int = 0
    
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0
    
    def rows_per_second(self) -> float:
        duration = self.duration_seconds()
        return self.rows_processed / duration if duration > 0 else 0
    
    def memory_delta_mb(self) -> float:
        return self.memory_end_mb - self.memory_start_mb
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds(), 2),
            "rows_processed": self.rows_processed,
            "rows_per_second": round(self.rows_per_second(), 2),
            "memory_delta_mb": round(self.memory_delta_mb(), 2),
            "total_errors": self.errors,
            "timestamp": datetime.utcnow().isoformat()
        }

class MetricsCollector:
    """Collect and track execution metrics."""
    
    def __init__(self):
        self.metrics = None
    
    def start(self, rows_count: int = 0) -> None:
        """Start metrics collection."""
        process = psutil.Process(os.getpid())
        self.metrics = Metrics(
            start_time=time.time(),
            rows_processed=rows_count,
            memory_start_mb=process.memory_info().rss / 1024**2
        )
    
    def end(self) -> Dict[str, Any]:
        """End metrics collection and return results."""
        if self.metrics:
            process = psutil.Process(os.getpid())
            self.metrics.end_time = time.time()
            self.metrics.memory_end_mb = process.memory_info().rss / 1024**2
            return self.metrics.to_dict()
        return {}
    
    def record_error(self) -> None:
        """Record an error."""
        if self.metrics:
            self.metrics.errors += 1
