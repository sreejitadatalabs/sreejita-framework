import time
import os

try:
    import psutil
except ImportError:
    psutil = None


class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()

    def collect(self):
        metrics = {
            "duration_sec": round(time.time() - self.start_time, 2)
        }

        if psutil:
            process = psutil.Process(os.getpid())
            metrics["memory_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)

        return metrics
