from abc import ABC, abstractmethod
import requests
from pydantic import BaseModel


class LLMModel(ABC):
    """
    Abstract base class for Large Language Models.

    This class should be inherited by concrete implementations of specific LLM models. It ensures
    that derived classes implement `__init__` and `chat` methods which are crucial for the operation
    of the model.

    Attributes:
    model_name (str): The name of the model being used.
    api_key (str, optional): The API key for authenticating with the model provider.
    api_secret (str, optional): The API secret for authenticating with the model provider.
    api_region (str, optional): The API region for interfacing with the model provider.

    Methods:
    chat: To be implemented in child classes for providing chatting functionality.
    """
    CHAT_URL = ""
    TEST_URL = ""

    @abstractmethod
    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        api_secret: str = None,
        api_region: str = None,
    ):
        """
        Initialize the LLMModel instance.

        Args:
        model_name (str): The name of the model to be used.
        api_key (str, optional): The API key for authentication. Default is None.
        api_secret (str, optional): The API secret for enhanced security. Default is None.
        api_region (str, optional): The API region for interfacing. Default is None.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_region = api_region

    @staticmethod
    def _raise_api_key_error():
        raise ValueError(
            "Please provide api_key parameter or set the specific environment variable."
        )

    def _check_api_access(self):
        response = requests.post(
            self.TEST_URL,
            json={
                "model_name": self.model_name,
                "api_key": self.api_key,
            },
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if not response.json():
            raise ValueError(
                f"The API key doesn't have access to {self.model_name}"
            )
    
    @abstractmethod
    def validate_parameters(self, parameters: BaseModel) -> BaseModel:
        """
        Validate and possibly adjust the provided parameters.

        Args:
        parameters (BaseModel): Parameters to validate, encapsulated in a Pydantic model.

        Returns:
        BaseModel: Validated/adjusted parameters encapsulated in a Pydantic model.
        """

    def chat(self, chat_input: str, parameters: BaseModel = None, is_stream: bool = False):
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
                                          "max tokens". Defaults to None.
        is_stream (bool, optional): A boolean flag that indicates whether the request should 
                                    be handled as a stream. Defaults to False.

        Returns:
        dict: The response from the API, typically containing the model's output.

        Raises:
        RequestException: If the API request fails.
        ValueError: If the API response cannot be parsed or contains error information.
        """
        validated_params = self.validate_parameters(parameters)

        response = requests.post(
            self.CHAT_URL,
            json={
                "model_name": self.model_name,
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "api_region": self.api_region,
                "chat_input": chat_input,
                "parameters": validated_params,
                "is_stream": is_stream,
            },
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        return response.json()


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
        self, api_key: str = None, api_secret: str = None, api_region: str = None
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

    def get_model(self, model_name: str):
        """
        Retrieve an instance of an LLM model by name.

        The method uses `MODEL_MAPPING` to locate and initialize the appropriate model class.

        Args:
        model_name (str): The name of the model to be retrieved.

        Returns:
        instance of the desired model class, initialized with the provided model name and API key.

        Raises:
        ValueError: If the model name is not found in `MODEL_MAPPING`.
        """
        model_class_name = self.MODEL_MAPPING.get(model_name)
        if not model_class_name:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = getattr(self, model_class_name)
        return model_class(
            model_name=model_name,
            api_key=self.api_key,
            api_secret=self.api_secret,
            api_region=self.api_region,
        )
