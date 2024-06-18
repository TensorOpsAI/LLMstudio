import aiohttp
import requests
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmstudio.cli import start_server
from llmstudio.config import ENGINE_HOST, ENGINE_PORT

from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Union, Dict
import random
import asyncio


# Batch Imports
import asyncio
import random
from typing import List, Union, Dict
from tqdm.asyncio import tqdm_asyncio


class LLM:

    class BatchTracker:
        def __init__(self):
            self.current_errors = 0
            self.target_error_rate = 0.05
            self.error_threshold = self.target_error_rate
            self.error_rate_smoothing = 0.01 

    def __init__(self, model_id: str, **kwargs):
        start_server()
        self.provider, self.model = model_id.split("/")
        self.session_id = kwargs.get("session_id")
        self.api_key = kwargs.get("api_key")
        self.api_endpoint = kwargs.get("api_endpoint")
        self.api_version = kwargs.get("api_version")
        self.base_url = kwargs.get("base_url")
        self.temperature = kwargs.get("temperature")
        self.top_p = kwargs.get("top_p")
        self.top_k = kwargs.get("top_k")
        self.max_tokens = kwargs.get("max_tokens")
        # self.failed_requests = 0
        # self.pause = False
        self.tracker = self.BatchTracker()

    def chat(self, input: str, is_stream: bool = False, **kwargs):
        response = requests.post(
            f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
            json={
                "model": self.model,
                "session_id": self.session_id,
                "api_key": self.api_key,
                "api_endpoint": self.api_endpoint,
                "api_version": self.api_version,
                "base_url": self.base_url,
                "chat_input": input,
                "is_stream": is_stream,
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "max_tokens": self.max_tokens,
                },
                **kwargs,
            },
            stream=is_stream,
            headers={"Content-Type": "application/json"},
        )

        response.raise_for_status()

        if is_stream:
            return self.generate_chat(response)
        else:
            return ChatCompletion(**response.json())

    def generate_chat(self, response):
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield ChatCompletionChunk(**chunk.decode("utf-8"))

    async def async_chat(self, input: str, is_stream=False, **kwargs):
        if is_stream:
            return self.async_stream(input)
        else:
            return await self.async_non_stream(input)

##################################### 1st BATCH #####################################

    # async def chat_coroutine(self, 
    #                          input: Union[str, List[Dict[str, str]]], 
    #                          semaphore: asyncio.Semaphore, 
    #                          max_retries: int = 5,
    #                          wait_time: int = 60,
    #                          fail_treshold: int = 5):
        
    #     async with semaphore:
            
    #         await asyncio.sleep(wait_time)
    #         for i in range(max_retries):
    #             try:

    #                 # Proceed with the request
    #                 response = await self.async_chat(input)
    #                 return response

    #             except Exception as e:
    #                 self.failed_requests += 1
    #                 if self.failed_requests >= fail_treshold:  # If 5 or more requests have failed
    #                     self.pause = True  # Set the pause flag
    #                 if i < max_retries - 1:  # i is zero indexed
    #                     wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
    #                     await asyncio.sleep(wait_time)
    #                 else:
    #                     return None

    # async def batch_chat_coroutine(self, inputs: List[Union[str, List[Dict[str, str]]]], 
    #                                num_coroutines: int = 5, 
    #                                max_retries: int = 5,
    #                                wait_time: int = 60,
    #                                fail_treshold: int = 5) -> List[str]:
    #     semaphore = asyncio.Semaphore(num_coroutines)
    #     responses = await tqdm_asyncio.gather(*[self.chat_coroutine(input, semaphore=semaphore, max_retries=max_retries, wait_time=wait_time, fail_treshold=fail_treshold) for input in inputs])
    #     return responses
    
    # def batch_chat(self, inputs: List[Union[str, List[Dict[str, str]]]], num_coroutines: int = 5, max_retries: int = 5, wait_time: int = 60, fail_treshold: int = 5) -> List[str]:
    #     return asyncio.run(self.batch_chat_coroutine(inputs, num_coroutines, max_retries, wait_time=wait_time, fail_treshold=fail_treshold))
    #####################################################################################

    ##################################### 2nd BATCH #####################################
    # class TimedBlockSemaphore:
    #     def __init__(self, count):
    #         self._semaphore = asyncio.Semaphore(count)
    #         self._block_event = asyncio.Event()
    #         self._block_event.set()  # Initially not blocked
    #         self._timer = None
    #         self.failed_requests = 0

    #     async def acquire(self):
    #         await self._block_event.wait()  # Wait until not blocked
    #         await self._semaphore.acquire()

    #     def release(self):
    #         self._semaphore.release()

    #     def block_for(self, seconds):
    #         if self._timer is not None:
    #             self._timer.cancel()  # Cancel existing timer
    #         self._block_event.clear()  # Block new acquisitions
    #         self._timer = asyncio.get_event_loop().call_later(seconds, self._unblock)

    #     def _unblock(self):
    #         self._block_event.set()  # Allow acquisitions
    #         self.failed_requests = 0
    #         self._timer = None

    # ###
    # async def chat_coroutine(self, 
    #                          input: Union[str, List[Dict[str, str]]], 
    #                          semaphore: TimedBlockSemaphore, 
    #                          max_retries: int = 5,
    #                          st: int = 60,
    #                          fail_treshold: int = 5):
    #     await semaphore.acquire()
    #     try:

    #         for i in range(max_retries):
    #             try:
    #                 # Proceed with the request
    #                 print(input)
    #                 response = await self.chat(input)
    #                 return response

    #             except Exception as e:
    #                 semaphore.failed_requests += 1
    #                 print(f'Failed requests:{semaphore.failed_requests}')
    #                 if semaphore.failed_requests >= fail_treshold:  # If 5 or more requests have failed
    #                     semaphore.block_for(st)  # Block the semaphore
    #                     print(f'Blocking semaphore for:{st} seconds')
    #                 if i < max_retries - 1:  # i is zero indexed
    #                     wait_time= (2 ** i) + random.random()  # Exponential backoff with jitter
    #                     await asyncio.sleep(wait_time)
    #                 else:
    #                     return None
    #     finally:
    #         semaphore.release()
    
    # async def batch_chat_coroutine(self, inputs: List[Union[str, List[Dict[str, str]]]], 
    #                                num_coroutines: int = 5, 
    #                                max_retries: int = 5,
    #                                stop_time: int = 60,
    #                                fail_treshold: int = 5) -> List[str]:
    #     semaphore = self.TimedBlockSemaphore(num_coroutines)
    #     responses = await asyncio.gather(*[self.chat_coroutine(input, semaphore=semaphore, max_retries=max_retries, st=stop_time, fail_treshold=fail_treshold) for input in inputs])
    #     return responses
    
    # def batch_chat(self, inputs: List[Union[str, List[Dict[str, str]]]], num_coroutines: int = 5, max_retries: int = 5, stop_time: int = 60, fail_treshold: int = 5) -> List[str]:
    #     return asyncio.run(self.batch_chat_coroutine(inputs, num_coroutines, max_retries, stop_time=stop_time, fail_treshold=fail_treshold))
    
    #####################################################################################

    ##################################### 3rd BATCH #####################################
    
    async def chat_coroutine(self, 
                             input: Union[str, List[Dict[str, str]]], 
                             semaphore: asyncio.Semaphore,
                             num_coroutines,
                             max_retries):
        
        async with semaphore:
            print('New coroutine added')
            
            # Make new coroutines wait while the error rate is too high
            while self.tracker.current_errors > (num_coroutines * self.tracker.error_threshold):
                await asyncio.sleep(1)  # Sleep for 1 second

            has_error = False # This flag is so one coroutine can ony increment the error count one time

            for i in range(max_retries):
                try:
                    # Proceed with the request
                    self.tracker.total_requests += 1  # Increment the total requests counter
                    response = await self.async_chat(input)
                    
                    # TODO
                    # Track tokens consume with max_tokens

                    # If thread had a error previously but was now successfull, decrease error count
                    if self.tracker.current_errors > 0 and has_error == True: 
                        self.tracker.current_errors -= 1
                        has_error = False
                    
                    return response

                
                except Exception as e:
                    print('Got error')

                    # Update error treshold if the coroutine faces an error.
                    if not has_error:
                        self.tracker.current_errors += 1  # Increment the count when an error occurs
                        has_error = True

                        # Compute error difference
                        current_error_rate = self.tracker.current_errors / num_coroutines
                        error_difference = current_error_rate - self.tracker.target_error_rate

                        # Update error treshold
                        self.error_threshold += self.error_rate_smoothing * error_difference  # Adjust error threshold based on error difference
                        self.error_threshold = max(0.0, min(1.0, self.error_threshold))
                    
                    # Perform exponential backoff with jitter
                    if i < max_retries - 1:  # i is zero indexed
                        wait_time = (2 ** i) + random.random()  # Exponential backoff with jitter
                        await asyncio.sleep(wait_time)

                    # Decrement the error count if the coroutine had an error
                    else:
                        if self.tracker.current_errors > 0 and has_error == True:  # Decrement the count if this coroutine had an error
                            self.tracker.current_errors -= 1
                        return None
                    
    # async def print_stats(self):
    #     while True:
    #         await asyncio.sleep(60)
    #         print(f"Requests sent in the last minute: {self.tracker.total_requests}")
    #         print(f"Tokens consumed in the last minute: {self.tracker.tokens_consumed}")
    #         self.tracker.total_requests = 0
    #         self.tracker.tokens_consumed = 0

    # async def batch_chat_coroutine(self, inputs: List[Union[str, List[Dict[str, str]]]], 
    #                                num_coroutines: int = 5, 
    #                                max_retries: int = 5) -> List[str]:
    #     semaphore = asyncio.Semaphore(num_coroutines)
    #     tasks = []
    #     for input in inputs:
    #         tasks.append(self.chat_coroutine(input, semaphore=semaphore, num_coroutines=num_coroutines, max_retries=max_retries))
    #         await asyncio.sleep(random.uniform(0.1, 1.0))  # Add a random delay before starting the next coroutine
    #     responses = await tqdm_asyncio.gather(*tasks)
    #     return responses, self.tracker.total_requests, self.tracker.error_requests
    
    async def batch_chat_coroutine(self, inputs: List[Union[str, List[Dict[str, str]]]], num_coroutines, max_retries) -> List[str]:

        # Set semaphore
        semaphore = asyncio.Semaphore(num_coroutines)

        # Get all responses in the same order of the input
        responses = await tqdm_asyncio.gather(*[self.chat_coroutine(input, semaphore=semaphore, max_retries=max_retries) for input in inputs])
        return responses
    
    def batch_chat(self, inputs: List[Union[str, List[Dict[str, str]]]],
                   max_retries: int = 5,
                   num_coroutines: int = 5,    
                   target_error_rate: float = 0.05,
                   error_smoothing: float = 0.01) -> List[str]:
        
        # Set tracker values
        self.tracker.target_error_rate = target_error_rate
        self.tracker.error_rate_smoothing = error_smoothing

        return asyncio.run(self.batch_chat_coroutine(inputs, num_coroutines, max_retries))
    #####################################################################################
    
    async def async_non_stream(self, input: str, **kwargs):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
                json={
                    "model": self.model,
                    "api_key": self.api_key,
                    "api_secret": self.api_endpoint,
                    "api_region": self.api_version,
                    "chat_input": input,
                    "is_stream": False,
                    **kwargs,
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                return ChatCompletion(**await response.json())

    async def async_stream(self, input: str, **kwargs):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
                json={
                    "model": self.model,
                    "api_key": self.api_key,
                    "api_secret": self.api_endpoint,
                    "api_region": self.api_version,
                    "chat_input": input,
                    "is_stream": True,
                    **kwargs,
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                async for chunk in response.content.iter_any():
                    if chunk:
                        yield ChatCompletionChunk(**await chunk.decode("utf-8"))
