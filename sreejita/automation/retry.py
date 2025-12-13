import time
from functools import wraps
from sreejita.utils.logger import get_logger

log = get_logger("retry")

def retry(times: int = 3, delay: int = 3):
    """
    Retry decorator for automation tasks.

    Args:
        times: number of attempts
        delay: delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, times + 1):
                try:
                    log.info("Attempt %s/%s for %s", attempt, times, func.__name__)
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    log.error(
                        "Attempt %s failed for %s: %s",
                        attempt,
                        func.__name__,
                        str(e)
                    )
                    if attempt < times:
                        time.sleep(delay)
            log.critical("All retries failed for %s", func.__name__)
            raise last_exception
        return wrapper
    return decorator
