import asyncio
import hashlib
from collections import OrderedDict

from pydantic import BaseModel


class Message:
    def __init__(self, prompt: any, response: any):
        self.prompt = prompt
        self.response = response


class MessageCache:
    def __init__(self, cache_limit, save_input=True, save_output=True):
        self._cache = OrderedDict()
        self._lock = asyncio.Lock()
        self.cache_limit = cache_limit
        self.save_input = save_input
        self.save_output = save_output

    def get_cache_limit(self):
        return self.cache_limit

    def hash_key(self, prompt):
        return hashlib.sha256(prompt.encode()).hexdigest()

    async def add_message(self, prompt, response):
        async with self._lock:
            hashed_key = self.hash_key(
                prompt
            )  # Changed key to prompt as key is not defined
            if len(self._cache) >= self.cache_limit:
                print("Removed old message from cache")
                self._cache.popitem(last=False)
            if self.save_input and self.save_output:
                self._cache[hashed_key] = Message(prompt=prompt, response=response)
            elif self.save_input:
                self._cache[hashed_key] = Message(prompt=prompt, response=None)
            elif self.save_output:
                self._cache[hashed_key] = Message(prompt=None, response=response)
            print("Added message to memory")

    async def remove_oldest_message(self):
        async with self._lock:
            if self._cache:
                return self._cache.popitem(last=False)
            return None

    async def get_message(self, prompt):
        async with self._lock:
            hashed_key = self.hash_key(prompt)
            message = self._cache.get(hashed_key, None)

            if message:
                return message.response
            return message

    async def remove_message(self, key):
        async with self._lock:
            hashed_key = self.hash_key(key)
            return self._cache.pop(hashed_key, None)

    async def cache_size(self):
        async with self._lock:
            return len(self._cache)
