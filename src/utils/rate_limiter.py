"""
Rate limiter utility for API calls.
"""

import time
from functools import wraps
import logging
from typing import Callable, Any
import random

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_requests: int = 5, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits."""
        current_time = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        
        return len(self.requests) < self.max_requests
    
    def add_request(self):
        """Record a new request."""
        self.requests.append(time.time())

def rate_limited(_func=None, *, max_retries=3, retry_delay=5):
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries in seconds
    """
    def decorator(func):
        rate_limiter = RateLimiter()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                if rate_limiter.can_make_request():
                    try:
                        rate_limiter.add_request()
                        return func(*args, **kwargs)
                    except Exception as e:
                        if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            jitter = random.uniform(0, 0.1 * wait_time)  # Add jitter
                            total_wait = wait_time + jitter
                            logger.info(f"Rate limit reached. Waiting {total_wait:.2f} seconds...")
                            time.sleep(total_wait)
                            continue
                        raise e
                else:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        jitter = random.uniform(0, 0.1 * wait_time)
                        total_wait = wait_time + jitter
                        logger.info(f"Rate limit reached. Waiting {total_wait:.2f} seconds...")
                        time.sleep(total_wait)
                    else:
                        raise Exception("Rate limit exceeded after all retry attempts")
            
            return None  # In case all retries are exhausted
        
        return wrapper
    
    if _func is None:
        return decorator
    return decorator(_func)
