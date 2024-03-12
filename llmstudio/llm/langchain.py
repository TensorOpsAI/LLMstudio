from typing import Any, Dict, List, Optional, Tuple

from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGeneration, ChatResult
from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_core.language_models.chat_models import BaseChatModel

from llmstudio.llm import LLM


class ChatLLMstudio(BaseChatModel):
    model_id: str
    llm: LLM = None

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.llm = LLM(model_id=self.model_id, **kwargs)

    @property
    def _llm_type(self):
        return "LLMstudio"

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _create_chat_result(self, response: Any) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": response["model"],
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        messages_dicts = self._create_message_dicts(messages, [])
        response = self.llm.chat(messages_dicts, **kwargs)
        return self._create_chat_result(response)
