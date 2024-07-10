import asyncio


class DynamicSemaphore:
    def __init__(self, initial_permits, batch_size, given_max_tokens):
        self.batch_size = batch_size
        self.given_max_tokens = given_max_tokens
        self.initial_permits = initial_permits
        self._permits = initial_permits
        self._semaphore = asyncio.Semaphore(initial_permits)
        self.computed_max_tokens = 0
        self.requests_since_last_increase = 0
        self.error_requests_since_last_increase = 0
        self.finished_requests = 0
        self.error_requests = 0

    def increase_permits(self, additional_permits):
        for _ in range(additional_permits):
            self._semaphore.release()
        self._permits += additional_permits

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._semaphore.release()

    def try_increase_permits(self, error_threshold, increment):
        if (
            self.requests_since_last_increase >= self._permits
            and self.error_requests_since_last_increase <= error_threshold
        ):
            self.increase_permits(increment)
            self.requests_since_last_increase = 0
            self.error_requests_since_last_increase = 0

    def update_computed_max_tokens(self, tokens):
        if self.finished_requests < self.initial_permits:
            self.computed_max_tokens = max(self.computed_max_tokens, tokens)

    def get_max_tokens(self):

        # If user provided max tokes, use that value
        if self.given_max_tokens != None:
            return self.given_max_tokens

        # If we are still computing max tokens, use models default value
        elif self.finished_requests < self.initial_permits:
            return 10

        # If we finished computing max tokens, return that value
        elif self.finished_requests >= self.initial_permits:
            return int(self.computed_max_tokens + (self.computed_max_tokens * 0.10))
