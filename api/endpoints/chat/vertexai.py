from typing import Optional
import json

from fastapi import APIRouter
from google.oauth2 import service_account
from pydantic import BaseModel, Field, validator, ValidationError
import vertexai
from vertexai.language_models import TextGenerationModel, ChatModel

from api.worker.config import celery_app
from api.utils import append_log

router = APIRouter()

class VertexAIParameters(BaseModel):
    """
    A Pydantic model that encapsulates parameters used for VertexAI API requests.

    Attributes:
    temperature (Optional[float]): Controls randomness in the model's output.
    max_tokens (Optional[int]): The maximum number of tokens in the output.
    top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
    top_k (Optional[float]): Sets the number of the most likely next tokens to filter for.
    """
    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(256, ge=1, le=1024)
    top_p: Optional[float] = Field(1, ge=0, le=1)
    top_k: Optional[float] = Field(40, ge=1, le=40)


class VertexAIRequest(BaseModel):
    """
    A Pydantic model for encapsulating data required to make a request to the VertexAI API.

    Attributes:
    api_key (dict): Authentication key for the VertexAI API, can be dict or JSON string.
    model_name (str): Identifier of the VertexAI model to use.
    chat_input (str): The input text to process by the model.
    parameters (Optional[VertexAIParameters]): Additional parameters to control the model's output.

    Methods:
    parse_api_key: Ensures that api_key is a dict and validates its JSON structure.
    validate_model_name: Validates that the model_name is one of the allowed options.
    """
    api_key: dict
    model_name: str
    chat_input: str
    parameters: Optional[VertexAIParameters] = VertexAIParameters()

    @validator("api_key", pre=True, always=True)
    def parse_api_key(cls, value):
        """
        Parse the API key, ensuring it is in a proper dictionary format.
        
        Args:
        value (Union[dict, str]): API key either as a JSON-formatted string or dictionary.
        
        Returns:
        dict: API key in dictionary format.
        
        Raises:
        ValueError: If the provided string is not valid JSON.
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception as exception:
                raise ValueError(f"Given string is not valid JSON: {value}") from exception
        return value

    @validator("model_name", always=True)
    def validate_model_name(cls, value):
        """
        Validate that the model name adheres to the allowed options.
        
        Args:
        value (str): The model name to validate.
        
        Returns:
        str: The validated model name.
        
        Raises:
        ValueError: If the model name is not an allowed value.
        """
        allowed_values = ["text-bison", "chat-bison"]
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value
    

def get_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost based on token usage.

    Args:
    input_tokens (int): Number of tokens in the input text.
    output_tokens (int): Number of tokens in the output text.

    Returns:
    float: The calculated cost for the API usage.
    """
    return 0.0000005 * (input_tokens + output_tokens)


@router.post("/vertexai")
async def get_vertexai_chat(data: VertexAIRequest):
    """
    FastAPI endpoint to interact with the VertexAI API for text generation or chat completions.
    
    Args:
    data (VertexAIRequest): Object containing necessary parameters for the API call.
    
    Returns:
    dict: A dictionary containing the chat input, chat output, tokens data, cost, and other metadata.
    """
    credentials = service_account.Credentials.from_service_account_info(data.api_key)
    vertexai.init(project=data.api_key["project_id"], credentials=credentials)

    if data.model_name == "text-bison":
        response = TextGenerationModel.from_pretrained(data.model_name).predict(
            prompt=data.chat_input,
            temperature=data.parameters.temperature,
            max_output_tokens=data.parameters.max_tokens,
            top_p=data.parameters.top_p,
            top_k=data.parameters.top_k,
        )
    elif data.model_name == "chat-bison":
        response = (
            ChatModel.from_pretrained(data.model_name)
            .start_chat()
            .send_message(
                message=data.chat_input,
                temperature=data.parameters.temperature,
                max_output_tokens=data.parameters.max_tokens,
                top_p=data.parameters.top_p,
                top_k=data.parameters.top_k,
            )
        )

    import random, time # delete

    data = {
        "id": random.randint(0, 1000),
        "chatInput": data.chat_input,
        "chatOutput": response.text,
        "inputTokens": len(data.chat_input),
        "outputTokens": len(response.text),
        "totalTokens": len(data.chat_input) + len(response.text),
        "cost": get_cost(len(data.chat_input), len(response.text)),
        "timestamp": time.time(),
        "modelName": data.model_name,
        "parameters": data.parameters.dict(),
    }

    append_log(data)

    return data