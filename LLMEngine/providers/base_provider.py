from abc import ABC


class BaseProvider(ABC):
    """
    Base class for LLMStudio LLMEngine providers.
    """
    def __init__(self):
        super().__init__()

    async def chat(self, data) -> dict:
        raise NotImplementedError

    def validate_model_field(self, data, model_list):
        from fastapi import HTTPException
        if not data.model_name:
            raise HTTPException(
                status_code=422,
                detail="The parameter 'model_name' is mandatory to be passed in the request body.",
            )
        if data.model_name not in model_list:
            raise HTTPException(
                status_code=422,
                detail=f"The model '{data['model_name']}' does not exist.",
            )
