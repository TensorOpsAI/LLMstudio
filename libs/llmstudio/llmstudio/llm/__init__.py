from typing import Any, Coroutine, Optional
from llmstudio_core import LLMCore
from llmstudio_core.providers.provider import ProviderABC
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmstudio_proxy.provider import LLMProxyProvider, ProxyConfig
from llmstudio_tracker.tracker import Tracker, TrackingConfig


class LLM(ProviderABC):


    def __init__(self,
                 provider: str,
                 api_key: Optional[str] = None,
                 proxy_config: Optional[ProxyConfig] = None,
                 tracking_config: Optional[TrackingConfig] = None,
                 **kwargs):
        
        if proxy_config is not None:
            self._provider = LLMProxyProvider(provider=provider,
                                              proxy_config=proxy_config)
        else:
            self._provider = LLMCore(provider=provider,
                                     api_key=api_key, 
                                     **kwargs)

        self._tracker = None
        if tracking_config is not None:
            self._tracker = Tracker(tracking_config=tracking_config)

    def _provider_config_name(self):
        return self._provider._provider_config_name()

    def chat(self, chat_input: Any, 
             model: str, is_stream: bool | None = False, 
             retries: int | None = 0, 
             parameters: Optional[dict] = {},
             **kwargs) -> ChatCompletionChunk | ChatCompletion:
        result = self._provider.chat(chat_input, model, is_stream, retries, parameters, **kwargs)
        
        if isinstance(result, (ChatCompletionChunk, ChatCompletion)):
            if self._tracker:
                result_dict = result.model_dump()
                result_dict["session_id"] = kwargs.get("session_id")
                self._tracker.log(result_dict)
            return result
        else:
            def generator_wrapper():
                for item in result:
                    yield item
                    if self._tracker and item.metrics:
                        result_dict = item.model_dump()
                        result_dict["session_id"] = kwargs.get("session_id")
                        self._tracker.log(result_dict)           
            return generator_wrapper()
    
    async def achat(self, chat_input: Any, 
              model: str, 
              is_stream: bool | None = False, 
              retries: int | None = 0,
              parameters: Optional[dict] = {},
              **kwargs) -> Coroutine[Any, Any, Coroutine[Any, Any, ChatCompletionChunk | ChatCompletion]]:
        result = await self._provider.achat(chat_input, model, is_stream, retries, parameters, **kwargs)
        if isinstance(result, (ChatCompletionChunk, ChatCompletion)):
            if self._tracker:
                result_dict = result.model_dump()
                result_dict["session_id"] = kwargs.get("session_id")
                self._tracker.log(result_dict)
                return result
        else:
            async def async_generator_wrapper():
                async for item in result:
                    yield item
                    if self._tracker and item.metrics:
                        result_dict = item.model_dump()
                        result_dict["session_id"] = kwargs.get("session_id")
                        self._tracker.log(result_dict)            
            return async_generator_wrapper()
