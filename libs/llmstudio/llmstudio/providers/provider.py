from typing import Any, Coroutine, Dict, Optional

from llmstudio.providers import LLMProxyProvider, ProxyConfig, Tracker, TrackingConfig
from llmstudio.utils import create_session_id
from llmstudio_core.providers import LLMCore
from llmstudio_core.providers.provider import Provider
from openai.types.chat import ChatCompletion, ChatCompletionChunk


class LLM(Provider):
    def __init__(
        self,
        provider: str,
        proxy_config: Optional[ProxyConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        base_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes an LLM provider instance to route your calls to the configured provider client.

        This constructor sets up the LLM provider with optional proxy and tracking configurations.
        If a proxy configuration is provided, it initializes an LLMProxyProvider; otherwise, it uses
        LLMCore. It also sets up a tracker if a tracking configuration is provided. The session_id
        is used to uniquely identify interactions within a session, and it requires a tracking
        configuration to be specified.

        Parameters:
        - provider (str): The name of the LLM provider (e.g., "openai", "vertexai").
        - proxy_config (Optional[ProxyConfig]): Configuration for proxy settings, applicable if proxy server is running.
        - tracking_config (Optional[TrackingConfig]): Configuration for tracking interactions with the LLM. Applicable if Tracking server is running.
        - session_id (Optional[str]): A unique identifier for the session, used for tracking purposes.

        # If running without a Proxy Server:
        # All providers:
        - api_key (Optional[str]): The API key for authenticating requests to the LLM provider.

        # Azure
        - api_endpoint (Optional[str]): The specific API endpoint to use for requests.
        - api_version (Optional[str]): The version of the API to use.
        - base_url (Optional[str]): The base URL for the API requests.

        - **kwargs: Additional keyword arguments for provider customization.

        Raises:
        - ValueError: If a session_id is provided without a tracking_config.
        """
        if proxy_config is not None:
            self._provider = LLMProxyProvider(
                provider=provider, proxy_config=proxy_config
            )
        else:
            self._provider = LLMCore(
                provider=provider,
                api_key=api_key,
                api_endpoint=api_endpoint,
                api_version=api_version,
                base_url=base_url,
                region=region,
                secret_key=secret_key,
                access_key=access_key,
                **kwargs,
            )

        self._session_id = None
        self._tracker = None
        if tracking_config is not None:
            self._tracker = Tracker(tracking_config=tracking_config)
            self._session_id = create_session_id() if session_id is None else session_id

        if (session_id is not None) and tracking_config is None:
            raise ValueError(
                f"'session_id' requires the 'tracking_config' specified and 'llmstudio[tracker]' installation."
            )

    def _provider_config_name(self):
        return self._provider._provider_config_name()

    def chat(
        self,
        chat_input: Any,
        model: str,
        is_stream: bool | None = False,
        retries: int | None = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ) -> ChatCompletionChunk | ChatCompletion:
        result = self._provider.chat(
            chat_input, model, is_stream, retries, parameters, **kwargs
        )

        if isinstance(result, (ChatCompletionChunk, ChatCompletion)):
            if self._tracker:
                result_dict = self._add_session_id(result, self._session_id)
                self._tracker.log(result_dict)
            return result
        else:

            def generator_wrapper():
                for item in result:
                    yield item
                    if self._tracker and item.metrics:
                        result_dict = self._add_session_id(item, self._session_id)
                        self._tracker.log(result_dict)

            return generator_wrapper()

    async def achat(
        self,
        chat_input: Any,
        model: str,
        is_stream: bool | None = False,
        retries: int | None = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ) -> Coroutine[Any, Any, Coroutine[Any, Any, ChatCompletionChunk | ChatCompletion]]:
        result = await self._provider.achat(
            chat_input, model, is_stream, retries, parameters, **kwargs
        )
        if isinstance(result, (ChatCompletionChunk, ChatCompletion)):
            if self._tracker:
                result_dict = self._add_session_id(result, self._session_id)
                self._tracker.log(result_dict)
            return result
        else:

            async def async_generator_wrapper():
                async for item in result:
                    yield item
                    if self._tracker and item.metrics:
                        result_dict = self._add_session_id(item, self._session_id)
                        self._tracker.log(result_dict)

            return async_generator_wrapper()

    @staticmethod
    def _add_session_id(
        result: ChatCompletionChunk | ChatCompletion, session_id: str
    ) -> Dict[str, Any]:
        result_dict = result.model_dump()
        result_dict["session_id"] = session_id
        return result_dict
