from typing import Any, Coroutine, Optional
from llmstudio_core import LLMCore
from llmstudio_core.providers.provider import ProviderABC
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from llmstudio.engine.provider import LLMProxyProvider, ProxyConfig
from llmstudio.tracking.database import create_tracking_engine
from llmstudio.tracking.logs import crud, schemas

from sqlalchemy.orm import sessionmaker

class TrackingConfig(BaseModel):
    database_uri: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if (self.host and self.port) or self.url or self.database_uri:
            raise ValueError("You must provide either both 'host' and 'port', or 'url', or 'database_uri'.")


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

        self._session_local = None
        if tracking_config is not None:
            engine = create_tracking_engine(tracking_config.database_uri)
            self._session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _provider_config_name(self):
        return self._provider._provider_config_name()
    
    def get_db(self):
        db = self.session_local()
        try:
            yield db
        finally:
            db.close()

    def chat(self, chat_input: Any, 
             model: str, is_stream: bool | None = False, 
             retries: int | None = 0, 
             parameters: Optional[dict] = {},
             **kwargs) -> ChatCompletionChunk | ChatCompletion:
        result = self._provider.chat(chat_input, model, is_stream, retries, parameters, **kwargs)
        if self._session_local:
            log = schemas.LogDefaultCreate(**result.metrics)
            crud.add_log(db=self._session_local, log=log)
        return result
    
    async def achat(self, chat_input: Any, 
              model: str, 
              is_stream: bool | None = False, 
              retries: int | None = 0,
              parameters: Optional[dict] = {},
              **kwargs) -> Coroutine[Any, Any, Coroutine[Any, Any, ChatCompletionChunk | ChatCompletion]]:
        result = await self._provider.achat(chat_input, model, is_stream, retries, parameters, **kwargs)
        if self._session_local and result.metrics:
            log = schemas.LogDefaultCreate(**result.metrics)
            crud.add_log(db=self._session_local, log=log)
        return result
