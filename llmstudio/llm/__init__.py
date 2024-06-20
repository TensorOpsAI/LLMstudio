# Batch Imports
import asyncio
import random
from typing import Dict, List, Union

import aiohttp
import requests
from IPython.display import clear_output
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tqdm.asyncio import tqdm_asyncio

from llmstudio.cli import start_server
from llmstudio.config import ENGINE_HOST, ENGINE_PORT


class LLM:

    ########################################## BATCH STUFF ##########################################
    class DynamicSemaphore:
        def __init__(self, initial_permits):
            self._permits = initial_permits
            self._semaphore = asyncio.Semaphore(initial_permits)

            self.requests_since_last_increase = 0
            self.error_requests_since_last_increase = 0

        def increase_permits(self, additional_permits):
            for _ in range(additional_permits):
                self._semaphore.release()
            self._permits += additional_permits

        async def __aenter__(self):
            await self._semaphore.acquire()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self._semaphore.release()

        def try_increment(self, error_threshold, increment):
            if (
                self.requests_since_last_increase >= self._permits
                and self.error_requests_since_last_increase <= error_threshold
            ):
                self.increase_permits(increment)
                self.requests_since_last_increase = 0
                self.error_requests_since_last_increase = 0

        def increment_requests_since_last_increase(self):
            self.requests_since_last_increase += 1

        def increment_error_requests_since_last_increase(self):
            self.error_requests_since_last_increase += 1

        def get_permits(self):
            return self._permits

    class BatchTracker:
        def __init__(self, given_max_tokens, total_inputs):
            self.total_inputs = total_inputs
            self.given_max_tokens = given_max_tokens
            self.total_requests = 0
            self.total_tokens = 0
            self.total_error_requests = 0
            self.sample_size = 10
            self.finished_requests = 0
            self.current_requests_with_errors = 0
            self.computed_max_tokens = 0

        def process_finished_request(self, has_error):

            # Increment finished requests counter
            self.finished_requests += 1

            # Decrement error counter if request had an error
            if has_error == True:
                if self.current_requests_with_errors > 0:
                    self.current_requests_with_errors -= 1

        def increment_total_tokens(self, tokens_used):
            self.total_tokens += tokens_used

        def increment_total_requests(self):
            self.total_requests += 1

        def increment_current_requests_with_errors(self):
            self.current_requests_with_errors += 1

        def increment_total_error_requests(self):
            self.total_error_requests += 1

        def update_computed_max_tokens(self, tokens):
            if self.finished_requests < self.sample_size:
                self.computed_max_tokens = max(self.computed_max_tokens, tokens)

        def get_max_tokens(self):

            # If user provided max tokes, use that value
            if self.given_max_tokens != None:
                return self.given_max_tokens

            # If we are still computing max tokens, use models default value
            elif self.finished_requests < self.sample_size:
                return 1024

            # If we finished computing max tokens, return that value
            elif self.finished_requests >= self.sample_size:
                return int(self.computed_max_tokens + (self.computed_max_tokens * 0.10))

    #################################################################################################

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
        self.frequency_penalty = kwargs.get("frequency_penalty")
        self.presence_penalty = kwargs.get("presence_penalty")

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
                    "temperature": kwargs.get("temperature") or self.temperature,
                    "top_p": kwargs.get("top_p") or self.top_p,
                    "top_k": kwargs.get("top_k") or self.top_k,
                    "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
                    "max_output_tokens": kwargs.get("max_tokens") or self.max_tokens,
                    "frequency_penalty": kwargs.get("frequency_penalty")
                    or self.frequency_penalty,
                    "presence_penalty": kwargs.get("presence_penalty")
                    or self.presence_penalty,
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

    ##################################### BATCH #####################################

    async def chat_coroutine(
        self,
        tracker,
        input: Union[str, List[Dict[str, str]]],
        semaphore,
        max_retries,
        error_threshold,
        increment,
        verbose,
    ):

        async with semaphore:
            # Make new coroutines wait while the error rate is too high
            while tracker.current_requests_with_errors > error_threshold:
                await asyncio.sleep(1)  # Sleep for 1 second

            has_error = False  # This flag is so one coroutine can only increment the error count one time

            for i in range(max_retries):

                # Everytime we do a request, we incement this value to track how many requests we did by the end

                try:
                    # Try getting a response
                    response = await self.async_chat(
                        input, max_tokens=tracker.get_max_tokens()
                    )
                    tracker.increment_total_requests()

                    # Update the max tokens used so far.
                    tracker.update_computed_max_tokens(response.metrics["total_tokens"])
                    tracker.increment_total_tokens(tracker.get_max_tokens())

                    # Update tracker (increment finished requests, decrement current error requests if req had an error)
                    tracker.process_finished_request(has_error)
                    semaphore.increment_requests_since_last_increase()

                    # Try to increment the semaphore if possible
                    semaphore.try_increment(error_threshold, increment)
                    return response

                except Exception:

                    tracker.increment_total_requests()
                    # Update total error counts
                    tracker.increment_total_error_requests()

                    # Update error rate if the coroutine faces an error.
                    if not has_error:
                        has_error = True
                        tracker.increment_current_requests_with_errors()  # Increment the count when an error occurs
                        semaphore.increment_error_requests_since_last_increase()

                    # Perform exponential backoff with jitter
                    if i < max_retries - 1:  # i is zero indexed
                        wait_time = (
                            2**i
                        ) + random.random()  # Exponential backoff with jitter
                        await asyncio.sleep(wait_time)

                    else:
                        # Update tracker (incement finished requests, decrement current error requests)
                        tracker.process_finished_request(has_error)
                        return None
                finally:
                    if verbose > 0:
                        clear_output()
                        print(
                            f"Finished requests: {tracker.finished_requests}/{tracker.total_inputs}"
                        )
                        print(
                            f"Requests on retry: {tracker.current_requests_with_errors}"
                        )
                        print(f"Total requests: {tracker.total_requests}")
                        print(f"Total error requests: {tracker.total_error_requests}")
                        print(f"Paralel request ammount: {semaphore.get_permits()}")

    async def batch_chat_coroutine(
        self,
        tracker,
        inputs: List[Union[str, List[Dict[str, str]]]],
        num_coroutines,
        max_retries,
        error_threshold,
        increment,
        verbose,
    ) -> List[str]:

        # Set semaphore
        semaphore = self.DynamicSemaphore(num_coroutines)

        # Get all responses in the same order of the input
        if verbose > 0:
            responses = await asyncio.gather(
                *[
                    self.chat_coroutine(
                        tracker=tracker,
                        input=input,
                        semaphore=semaphore,
                        max_retries=max_retries,
                        error_threshold=error_threshold,
                        increment=increment,
                        verbose=verbose,
                    )
                    for input in inputs
                ],
            )
            return responses

        else:
            responses = await tqdm_asyncio.gather(
                *[
                    self.chat_coroutine(
                        tracker=tracker,
                        input=input,
                        semaphore=semaphore,
                        max_retries=max_retries,
                        error_threshold=error_threshold,
                        increment=increment,
                        verbose=verbose,
                    )
                    for input in inputs
                ],
                desc="Getting chat responses: ",
            )
            return responses

    def batch_chat(
        self,
        inputs: List[Union[str, List[Dict[str, str]]]],
        num_coroutines: int = 20,
        max_retries: int = 5,
        error_threshold: int = 5,
        max_tokens=None,
        increment: int = 5,
        verbose: int = 0,
    ) -> List[str]:

        """
        Perform chat requests in bulk with retry logic scaling up to API rate limits.

        Parameters
        ----------
        inputs : list of str or list of dict
            The inputs for the chat requests.
        num_coroutines : int, default=20
            Starting number of coroutines to use for the batch operation. Try choosing a value close to your models TPM rate limit.
        max_retries : int, default=5
            The maximum number of retries for each chat operation.
        error_threshold : int, default=5
            The maximum number of errors allowed before the batch operation considers it is being throttled.
        max_tokens : int, optional
            The maximum tokens per chat operation defaults to the highest token count in the first 10 requests, increased by 10%.
        increment : int, default=5
            The increment for the semaphore permits.
        verbose : int, default=0
            The level of verbosity. If greater than 0, progress information is printed.

        Returns
        -------
        list of str
            The responses from the chat operations.

        """

        # Error treshold can not be higher than the number of coroutines
        if error_threshold > num_coroutines:
            raise Exception(
                "error_treshold can not be higher than the num_couroutines."
            )

        if num_coroutines > len(inputs):
            raise Exception(
                "num_coroutines can not be higher than the amount of inputs."
            )

        tracker = self.BatchTracker(
            given_max_tokens=max_tokens, total_inputs=len(inputs)
        )

        responses = asyncio.run(
            self.batch_chat_coroutine(
                tracker,
                inputs,
                num_coroutines,
                max_retries,
                error_threshold,
                increment,
                verbose,
            )
        )
        return responses

    #####################################################################################

    async def async_non_stream(self, input: str, **kwargs):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
                json={
                    "model": self.model,
                    "session_id": self.session_id,
                    "api_key": self.api_key,
                    "api_endpoint": self.api_endpoint,
                    "api_version": self.api_version,
                    "base_url": self.base_url,
                    "chat_input": input,
                    "is_stream": False,
                    "parameters": {
                        "temperature": kwargs.get("temperature") or self.temperature,
                        "top_p": kwargs.get("top_p") or self.top_p,
                        "top_k": kwargs.get("top_k") or self.top_k,
                        "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
                        "max_output_tokens": kwargs.get("max_tokens")
                        or self.max_tokens,
                        "frequency_penalty": kwargs.get("frequency_penalty")
                        or self.frequency_penalty,
                        "presence_penalty": kwargs.get("presence_penalty")
                        or self.presence_penalty,
                    },
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
                    "session_id": self.session_id,
                    "api_key": self.api_key,
                    "api_endpoint": self.api_endpoint,
                    "api_version": self.api_version,
                    "base_url": self.base_url,
                    "chat_input": input,
                    "is_stream": True,
                    "parameters": {
                        "temperature": kwargs.get("temperature") or self.temperature,
                        "top_p": kwargs.get("top_p") or self.top_p,
                        "top_k": kwargs.get("top_k") or self.top_k,
                        "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
                        "max_output_tokens": kwargs.get("max_tokens")
                        or self.max_tokens,
                        "frequency_penalty": kwargs.get("frequency_penalty")
                        or self.frequency_penalty,
                        "presence_penalty": kwargs.get("presence_penalty")
                        or self.presence_penalty,
                    },
                    **kwargs,
                },
                stream=True,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                async for chunk in response.content.iter_any():
                    if chunk:
                        yield ChatCompletionChunk(**await chunk.decode("utf-8"))
