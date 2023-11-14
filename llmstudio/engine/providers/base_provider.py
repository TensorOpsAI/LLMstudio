from abc import ABC
from concurrent.futures import ThreadPoolExecutor


class BaseProvider(ABC):
    """
    Abstract base class for LLMstudio engine providers.

    This class defines the core interface for providers that integrate
    with LLMstudio's engine. It is intended to be subclassed by specific
    provider implementations.
    """

    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def chat(self, data) -> dict:
        """
        Asynchronously handle a chat request.

        Parameters:
            data: The data payload for the chat operation.

        Returns:
            dict: A dictionary containing the chat response.

        Raises:
            NotImplementedError: This method is intended to be overridden by subclasses.
        """
        raise NotImplementedError

    async def test(test, data) -> dict:
        """
        Asynchronously handle a test request.

        Parameters:
            data: The data payload for the test operation.

        Returns:
            dict: A dictionary containing the test response.

        Raises:
            NotImplementedError: This method is intended to be overridden by subclasses.
        """
        raise NotImplementedError

    def validate_model_field(self, data, model_list):
        """
        Validate the 'model' field in the request data.

        Parameters:
            data: The data payload containing the 'model'.
            model_list: List of valid model names.

        Raises:
            HTTPException: If the 'model' is not provided or is not in the list of valid models.
        """
        from fastapi import HTTPException

        if not data.model:
            raise HTTPException(
                status_code=422,
                detail="The parameter 'model' is mandatory to be passed in the request body.",
            )
        if data.model not in model_list:
            raise HTTPException(
                status_code=422,
                detail=f"The model '{data.model}' does not exist.",
            )
