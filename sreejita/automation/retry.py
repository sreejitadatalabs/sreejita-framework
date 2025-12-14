import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def retry(times: int = 3, delay: int = 3):
    """
    Retry decorator for automation tasks.

    Args:
        times (int): Number of retry attempts
        delay (int): Delay in seconds between attempts
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, times + 1):
                try:
                    logger.info(
                        "Attempt %s/%s for %s",
                        attempt,
                        times,
                        func.__name__,
                    )
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    logger.error(
                        "Error on attempt %s for %s: %s",
                        attempt,
                        func.__name__,
                        exc,
                    )
                    if attempt < times:
                        time.sleep(delay)

            logger.critical(
                "All %s attempts failed for %s",
                times,
                func.__name__,
            )
            raise last_exception

        return wrapper

    return decorator
