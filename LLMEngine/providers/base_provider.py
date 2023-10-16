from typing import Tuple
from LLMEngine.config import RouteConfig

from pydantic import BaseModel

class BaseProvider(BaseModel):
    """
    Base class for LLMStudio LLMEngine providers.
    """

    NAME: str
    SUPPORTED_ROUTE_TYPES: Tuple[str, ...]

    def __init__(self, config: RouteConfig):
        self.config = config

    async def chat(self, data) -> dict:
        raise NotImplementedError

    @staticmethod
    def validate_model_field(data, model_list):
        from fastapi import HTTPException

        if "model_name" not in data:
            raise HTTPException(
                status_code=422,
                detail="The parameter 'model_name' is mandatory to be passed in the request body.",
            )
        if data["model_name"] not in model_list:
            raise HTTPException(
                status_code=422,
                detail=f"The model '{data['model_name']}' does not exist.",
            )
