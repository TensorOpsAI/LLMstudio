import asyncio
import time
from abc import ABC, abstractmethod
from statistics import mean

import nest_asyncio
import numpy as np
import requests
from aiohttp import ClientSession, ClientTimeout
from pydantic import BaseModel
from requests.exceptions import RequestException
from sentence_transformers import SentenceTransformer, util

from llmstudio.engine.config import EngineConfig, RouteType
from llmstudio.utils.rest_utils import run_apis


class LLMModel(ABC):
    """
    Abstract base class for Large Language Models.

    This class should be inherited by concrete implementations of specific LLM models. It ensures
    that derived classes implement `__init__` and `chat` methods which are crucial for the operation
    of the model.

    Attributes:
        model (str): The name of the model being used.
        api_key (str, optional): The API key for authenticating with the model provider.
        api_secret (str, optional): The API secret for authenticating with the model provider.
        api_region (str, optional): The API region for interfacing with the model provider.

    Methods:
        chat: To be implemented in child classes for providing chatting functionality.
    """

    PROVIDER = None

    @abstractmethod
    def __init__(
        self,
        model: str,
        api_key: str = None,
        api_secret: str = None,
        api_region: str = None,
        tests: dict = {},
        engine_config: EngineConfig = EngineConfig(),
        parameters: BaseModel = None,
    ):
        """
        Initialize the LLMModel instance.

        Args:
            model (str): The name of the model to be used.
            api_key (str, optional): The API key for authentication. Default is None.
            api_secret (str, optional): The API secret for enhanced security. Default is None.
            tests (dict, optional): Dictionary of batch of tests to be used when running tests or evaluation. Default is Empty Dict.
            api_region (str, optional): The API region for interfacing. Default is None.
        """
        self.model = model
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_region = api_region
        self.tests = tests
        self.validation_url = f"{str(engine_config.routes_endpoint)}/{RouteType.LLM_VALIDATION.value}/{self.PROVIDER}"
        self.chat_url = (
            f"{str(engine_config.routes_endpoint)}/{RouteType.LLM_CHAT.value}/{self.PROVIDER}"
        )
        validated_params = self.validate_parameters(parameters)
        self.parameters = validated_params

    @staticmethod
    def _raise_api_key_error():
        raise ValueError(
            "Please provide api_key parameter or set the specific environment variable."
        )

    def _check_api_access(self, max_retries=3, delay=0.5):
        retries = 0
        while retries < max_retries:
            try:
                response = requests.post(
                    self.validation_url,
                    json={
                        "model": self.model,
                        "api_key": self.api_key,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                if not response.json():
                    raise ValueError(f"The API key doesn't have access to {self.model}")
                return  # Successful API check
            except RequestException:
                retries += 1
                time.sleep(delay)  # Wait before retrying

        raise ValueError("Max retries reached. The API key doesn't have access to {self.model}")

    @abstractmethod
    def validate_parameters(self, parameters: BaseModel) -> BaseModel:
        """
        Validate and possibly adjust the provided parameters.

        Args:
            parameters (BaseModel): Parameters to validate, encapsulated in a Pydantic model.

        Returns:
            BaseModel: Validated/adjusted parameters encapsulated in a Pydantic model.
        """

    def generate_chat(self, response):
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")

    def chat(
        self,
        chat_input: str,
        parameters: BaseModel = None,
        is_stream: bool = False,
        safety_margin=None,
        custom_max_tokens=None,
        timeout_s: int = 120,
    ):
        """
        Initiate a chat interaction with the language model.

        This method sends a request to the language model API, providing an input string and
        optionally some parameters to influence the model's responses. It then returns the
        model's output as received from the API.

        Args:
            chat_input (str): The input string to send to the model. This is typically a prompt
                          that you want the model to respond to.
            parameters (BaseModel, optional): A Pydantic model containing parameters that affect
                                          the model's responses, such as "temperature" or
                                          "max tokens". Defaults to None. If no parameters are specified the ones declared when instancianting the model will be used if available.
            is_stream (bool, optional): A boolean flag that indicates whether the request should
                                    be handled as a stream. Defaults to False.

        Returns:
            dict: The response from the API, typically containing the model's output.

        Raises:
            RequestException: If the API request fails.
            ValueError: If the API response cannot be parsed or contains error information.
        """
        if not parameters:
            parameters = self.parameters
        else:
            parameters = self.validate_parameters(parameters)
        response = requests.post(
            self.chat_url,
            json={
                "model": self.model,
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "api_region": self.api_region,
                "chat_input": chat_input,
                "parameters": parameters,
                "is_stream": is_stream,
                "safety_margin": safety_margin,
                "end_token": False,
                "custom_max_token": custom_max_tokens,
            },
            stream=is_stream,
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )

        if is_stream:
            return self.generate_chat(response)
        else:
            return response.json()

    async def chat_async(
        self,
        chat_input: str,
        parameters: BaseModel = None,
        is_stream: bool = False,
        timeout_s: int = 120,
    ):

        timeout = ClientTimeout(total=timeout_s)

        async with ClientSession(timeout=timeout) as session:
            if not parameters:
                parameters = self.parameters
            else:
                parameters = self.validate_parameters(parameters)
            async with session.post(
                self.chat_url,
                json={
                    "model": self.model,
                    "api_key": self.api_key,
                    "api_secret": self.api_secret,
                    "api_region": self.api_region,
                    "chat_input": chat_input,
                    "parameters": parameters,
                    "is_stream": is_stream,
                },
            ) as response:
                return await response.json()

    async def run_tests_async(
        self, tests: dict = {}, parameters: dict = {}, is_stream: bool = False
    ):
        if not tests:
            tests = self.tests
        tests = self.validate_tests(tests)

        async def run_single_test(key):
            test, gt_answer = tests[key].values()
            answer = await self.chat_async(test, parameters=parameters, is_stream=is_stream)
            return key, answer

        tasks = [run_single_test(key) for key in tests]
        test_responses = await asyncio.gather(*tasks)
        return {k: v for k, v in test_responses}

    def run_tests(self, tests: dict = {}, parameters: dict = {}, is_stream: bool = False):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Jupyter Notebook; use a workaround since loop is already running
            nest_asyncio.apply()
            return loop.run_until_complete(self.run_tests_async(tests, parameters, is_stream))
        else:
            # Standalone script; standard approach
            return asyncio.run(self.run_tests_async(tests, parameters, is_stream))

    def set_tests(self, tests: dict = {}):
        self.tests = tests

    def validate_tests(self, tests):
        """
        Validate that the tests are not empty

        Args:
            tests: The tests to validate.

        Returns:
            str: The validated tests.

        Raises:
            ValueError: If the tests are empty.
        """
        if not tests:
            raise ValueError(f"tests should not be empty")

        correct_format = (
            """'test_1': {'question': 'What is the capital of Portugal', 'answer': 'Lisbon'}"""
        )

        if not isinstance(tests, dict):
            raise ValueError(f"tests is not a dict")
        for key, value in tests.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"key of tests is not a string. Example of correct format: {correct_format}"
                )
            if not isinstance(value, dict):
                raise ValueError(
                    f"value of each test is not a Dictionary. Example of correct format: {correct_format}"
                )
            if "question" not in value or "answer" not in value:
                raise ValueError(f"tests value should be in format: {correct_format}")
            if not (isinstance(value["question"], str) and isinstance(value["answer"], str)):
                raise ValueError(f"question and answer should be strings")
        return tests


class LLMClient(ABC):
    """
    Abstract base class for Large Language Model Vendor Client.

    This class represents an abstract client to interact with various LLMs. Concrete
    implementations should realize the `get_model` method and utilize the `MODEL_MAPPING`
    to facilitate the retrieval of model instances.

    Attributes:
        MODEL_MAPPING (dict): A mapping from string model names to model class names.
        api_key (str, optional): The API key for authentication.
        api_secret (str, optional): The API secret for enhanced security.
        api_region (str, optional): The API region for interfacing.

    Methods:
        get_model: Retrieve an instance of an LLM model by name.
    """

    MODEL_MAPPING = {}

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        api_region: str = None,
        engine_config: EngineConfig = EngineConfig(),
    ):
        """
        Initialize the LLMClient instance.

        Args:
            api_key (str, optional): The API key for authentication. Default is None.
            api_secret (str, optional): The API secret for enhanced security. Default is None.
            api_region (str, optional): The API region for interfacing. Default is None.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_region = api_region
        self.engine_config = engine_config
        run_apis(engine_config=self.engine_config)

    def get_model(self, model: str, parameters: BaseModel = None):
        """
        Retrieve an instance of an LLM model by name.

        The method uses `MODEL_MAPPING` to locate and initialize the appropriate model class.

        Args:
            model (str): The name of the model to be retrieved.

        Returns:
            instance of the desired model class, initialized with the provided model name and API key.

        Raises:
            ValueError: If the model name is not found in `MODEL_MAPPING`.
        """
        model_class_name = self.MODEL_MAPPING.get(model)
        if not model_class_name:
            raise ValueError(f"Unknown model: {model}")

        model_class = getattr(self, model_class_name)
        return model_class(
            model=model,
            api_key=self.api_key,
            api_secret=self.api_secret,
            api_region=self.api_region,
            engine_config=self.engine_config,
            parameters=parameters,
        )


class LLMCompare(ABC):
    def __int__(self):
        pass

    async def _get_response_from_model(self, model: LLMModel, prompt: str, output_dict: dict):
        """
        Helper method to get response from a given model and store it in the output dictionary.

        Args:
            model (LLMModel): The language model to get a response from.
            prompt (str): The input prompt for the model.
            output_dict (dict): Dictionary to store the responses.

        Returns:
            dict: The updated output_dict.
        """
        output_dict[model.model] = model.chat(prompt)
        return output_dict

    def _compute_entrywise_average_similarity(self, list1, list2):
        """
        Computes cosine average_similarity for each pair of sentences at the same position in two lists.

        Parameters:
        - list1: List of sentences [s1, s2, ...]
        - list2: List of sentences [s1, s2, ...]

        Returns:
        - average_similarity_vector: 1D numpy array with the average_similarity for each respective entry.
        """

        # initiate an embedding model
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        # Ensure the two lists are of the same length
        assert len(list1) == len(list2), "The two lists must be of the same length"

        # Encode the sentences from both lists
        embeddings1 = model.encode(list1, convert_to_tensor=True)
        embeddings2 = model.encode(list2, convert_to_tensor=True)

        # Compute average_similarity for respective entries
        average_similarity_vector = [
            util.pytorch_cos_sim(emb1, emb2).item() for emb1, emb2 in zip(embeddings1, embeddings2)
        ]

        return np.array(average_similarity_vector)

    async def _get_llm_performance(self, model, prompt_list, expected_output_list, output_dict):
        latency_list = []
        cost_list = []
        out_tokens_list = []
        chat_output_list = []

        assert len(prompt_list) == len(
            expected_output_list
        ), "Prompt List and Expected List are not the same size"

        for prompt in prompt_list:
            model_response = model.chat(prompt)  # assuming the chat method is asynchronous

            chat_output_list.append(model_response["chatOutput"])
            latency_list.append(model_response["latency"])
            cost_list.append(model_response["cost"])
            out_tokens_list.append(model_response["outputTokens"])

        # now compute some metrics
        average_latency = mean(latency_list)
        average_output_token = mean(out_tokens_list)
        average_cost = mean(cost_list)

        # average_similarity performance
        average_similarity = mean(
            self._compute_entrywise_average_similarity(chat_output_list, expected_output_list)
        )

        statistics = {
            "average_latency": average_latency,
            "average_cost": average_cost,
            "average_output_token": average_output_token,
            "average_similarity": average_similarity,
        }
        output_dict[model.model] = statistics
        return output_dict

    async def single_prompt_compare(self, models: [LLMClient], prompt: str):
        """
        Compare multiple language models by obtaining their responses to a given prompt.

        Args:
            models (list[LLMClient]): List of language models to compare.
            prompt (str): Input prompt for the models.

        Returns:
            dict: A dictionary where keys are model names and values are their corresponding responses.
        """

        output_dict = {}

        tasks = [self._get_response_from_model(model, prompt, output_dict) for model in models]

        await asyncio.gather(*tasks)

        return output_dict

    async def dataset_prompt_compare(self, models, prompt_list, expected_output_list):
        output_dict = {}

        tasks = [
            self._get_llm_performance(model, prompt_list, expected_output_list, output_dict)
            for model in models
        ]

        await asyncio.gather(*tasks)

        return output_dict
