import asyncio
from typing import Optional

class DynamicSemaphore:
    """A class that implements a dynamically adjustable semaphore for async control flow.

    Attributes:
        batch_size (int): The number of tasks that can be handled in a batch.
        max_tokens (Optional[int]): The maximum limit of permits that can be issued.
        initial_permits (int): The initial number of permits.
    """
    
    def __init__(self, initial_permits: int, batch_size: int, max_tokens: Optional[int] = None):
        """Initialize the DynamicSemaphore with a specific number of initial permits, batch size, and an optional maximum tokens limit."""
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.initial_permits = initial_permits
        self._permits = initial_permits
        self._semaphore = asyncio.Semaphore(initial_permits)
        self.requests_since_last_increase = 0
        self.error_requests_since_last_increase = 0
        self.finished_requests = 0
        self.error_requests = 0

    def _increase_permits(self, additional_permits: int) -> None:
        """Increase the number of available permits by a specified amount."""
        for _ in range(additional_permits):
            self._semaphore.release()
        self._permits += additional_permits
        if self.max_tokens is not None:
            self._permits = min(self._permits, self.max_tokens)  # Ensure not to exceed max tokens if set

    async def __aenter__(self) -> "DynamicSemaphore":
        """Enter an asynchronous context after acquiring a semaphore."""
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the asynchronous context by releasing a semaphore."""
        self._semaphore.release()

    def try_increase_permits(self, error_threshold: int, increment: int) -> None:
        """Try to increase the number of permits based on the error threshold and requests since the last increase."""
        if (self.requests_since_last_increase >= self._permits and
                self.error_requests_since_last_increase <= error_threshold):
            self._increase_permits(increment)
            self.requests_since_last_increase = 0
            self.error_requests_since_last_increase = 0