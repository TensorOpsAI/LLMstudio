import copy
from typing import Any, List, Optional, Union

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel


class Metrics(BaseModel):
    input_tokens: int
    """Number of tokens in the input."""

    output_tokens: int
    """Number of tokens in the output."""

    reasoning_tokens: int
    """Number of reasoning tokens used by the model."""

    total_tokens: int
    """Total token count (input + output + reasoning)."""

    cached_tokens: int
    """Number of cached tokens which will lower the price of input."""

    cost_usd: float
    """Total cost of the response in USD (input + output + reasoning - cached)."""

    latency_s: float
    """Total time taken for the response, in seconds."""

    time_to_first_token_s: Optional[float] = None
    """Time to receive the first token, in seconds."""

    inter_token_latency_s: Optional[float] = None
    """Average time between tokens, in seconds. Defaults to None if not provided."""

    tokens_per_second: Optional[float] = None
    """Processing rate of tokens per second. Defaults to None if not provided."""

    def __getitem__(self, key: str) -> Any:
        """
        Allows subscriptable access to class fields.

        Parameters
        ----------
        key : str
            The name of the field to retrieve.

        Returns
        -------
        Any
            The value of the specified field.

        Raises
        ------
        KeyError
            If the key does not exist.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' not found in MetricsStream.")

    def __iter__(self):
        """
        Allows iteration over the class fields as key-value pairs.
        """
        return iter(self.model_dump().items())

    def __len__(self):
        """
        Returns the number of fields in the class.
        """
        return len(self.model_fields)

    def keys(self):
        """
        Returns the keys of the fields.
        """
        return self.model_fields.keys()

    def values(self):
        """
        Returns the values of the fields.
        """
        return self.model_dump().values()

    def items(self):
        """
        Returns the key-value pairs of the fields.
        """
        return self.model_dump().items()


class ChatCompletionLLMstudioBase:
    """
    Base class to share the methods between different ChatCompletionLLMstudio classes.
    """

    def clean_print(self):
        """
        Custom representation of the class to prevent large fields from bloating the output.
        Ensures missing fields are handled gracefully without errors.
        """
        data = copy.deepcopy(self.model_dump())

        def clean_large_fields(d):
            """
            Recursively traverses the dictionary to replace large image Base64 data
            with a placeholder while ensuring missing fields do not cause errors.
            """
            for key, value in d.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            # Handle image_url directly under chat_input or context
                            if "image_url" in item and isinstance(
                                item["image_url"], dict
                            ):
                                if "url" in item["image_url"] and isinstance(
                                    item["image_url"]["url"], str
                                ):
                                    if item["image_url"]["url"].startswith(
                                        "data:image/"
                                    ):
                                        item["image_url"][
                                            "url"
                                        ] = "<large image base64 data hidden>"

                            # Handle nested content inside context
                            if "content" in item and isinstance(item["content"], list):
                                for content_item in item["content"]:
                                    if (
                                        isinstance(content_item, dict)
                                        and "image_url" in content_item
                                    ):
                                        if "url" in content_item[
                                            "image_url"
                                        ] and isinstance(
                                            content_item["image_url"]["url"], str
                                        ):
                                            if content_item["image_url"][
                                                "url"
                                            ].startswith("data:image/"):
                                                content_item["image_url"][
                                                    "url"
                                                ] = "<large image base64 data hidden>"

            return d

        cleaned_data = clean_large_fields(data)
        print(cleaned_data)


class ChatCompletionLLMstudio(ChatCompletion, ChatCompletionLLMstudioBase):
    chat_input: Union[str, List[dict]]
    """The input prompt for the chat completion."""

    chat_output: str
    """The final response generated by the model."""

    chat_output_stream: str
    """Incremental chunks of the response for streaming."""

    context: List[dict]
    """The conversation history or context provided to the model."""

    provider: str
    """Identifier for the backend service provider."""

    deployment: Optional[str]
    """Information about the deployment configuration used."""

    timestamp: float
    """The timestamp when the response was generated."""

    parameters: dict
    """The parameters used in the request."""

    metrics: Optional[Metrics]
    """Performance and usage metrics calculated for the response."""


class ChatCompletionChunkLLMstudio(ChatCompletionChunk, ChatCompletionLLMstudioBase):
    chat_input: Union[str, List[dict]]
    """The input prompt for the chat completion."""

    chat_output: Optional[str]
    """The final response generated by the model."""

    chat_output_stream: Optional[str]
    """Incremental chunks of the response for streaming."""

    context: List[dict]
    """The conversation history or context provided to the model."""

    provider: str
    """Identifier for the backend service provider."""

    deployment: Optional[str]
    """Information about the deployment configuration used."""

    timestamp: float
    """The timestamp when the chunk was generated."""

    parameters: dict
    """The parameters used in the request."""

    metrics: Optional[Metrics]
    """Performance and usage metrics calculated for the response chunk."""
