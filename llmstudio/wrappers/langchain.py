from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from ..models import LLMModel


class LangchainLLMWrapper(LLM):
    model: LLMModel

    @property
    def _llm_type(self) -> str:
        return "Langchain Wrapper of " + self.model.model

    def _call(
        self,
        prompt: str,
        parameters: Optional[dict] = {},
        is_stream: Optional[bool] = False,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        response = self.model.chat(prompt, parameters=parameters)
        return response["chatOutput"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
