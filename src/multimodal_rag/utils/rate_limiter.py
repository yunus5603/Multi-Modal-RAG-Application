import time
from typing import Callable, TypeVar, Any
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class RateLimiter:
    """Rate limiter for API calls with adaptive backoff."""
    
    def __init__(self, max_retries: int = 5, initial_delay: float = 1.0):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def with_retry(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for retrying functions with exponential backoff."""
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = self.initial_delay
            last_exception = None
            
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if "rate_limit" in str(e).lower():
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}/{self.max_retries}")
                        time.sleep(wait_time)
                    else:
                        # If it's not a rate limit error, re-raise immediately
                        raise e
            
            # If we've exhausted all retries
            logger.error(f"Max retries ({self.max_retries}) exceeded")
            raise last_exception

        return wrapper
    
    async def process_batch_async(self, items: list, processor: Callable, batch_size: int = 2, delay: float = 1.0) -> list:
        """Process items in batches asynchronously with rate limiting."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            try:
                # Process batch in thread pool to avoid blocking
                batch_results = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.with_retry(processor),
                    batch
                )
                results.extend(batch_results)
                
                if i + batch_size < len(items):
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                # Return processed results if partial failure
                return results
        
        return results

    def __del__(self):
        """Cleanup executor on deletion."""
        self.executor.shutdown(wait=False) 