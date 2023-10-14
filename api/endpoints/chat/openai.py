from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import openai
from pydantic import BaseModel, Field, validator
import tiktoken

from api.worker.config import celery_app
from api.utils import append_log

end_token = "<END_TOKEN>"

router = APIRouter()


class OpenAIParameters(BaseModel):
    """
    A Pydantic model for encapsulating parameters used in OpenAI API requests.
    
    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
        frequency_penalty (Optional[float]): Modifies the likelihood of tokens appearing based on their frequency.
        presence_penalty (Optional[float]): Adjusts the likelihood of new tokens appearing.
    """
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=2048)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class OpenAIRequest(BaseModel):
    """
    A Pydantic model for encapsulating data needed to make a request to the OpenAI API.
    
    Attributes:
        api_key (str): Authentication key for the OpenAI API.
        model_name (str): The identifier of the GPT model to be used.
        chat_input (str): The input text for the model.
        parameters (Optional[OpenAIParameters]): Additional parameters to control the model’s output.
        is_stream (Optional[bool]): Flag to determine whether to receive streamed responses from the API.
    
    Methods:
        validate_model_name: Ensures that the chosen model_name is one of the allowed models.
    """
    api_key: str
    model_name: str
    chat_input: str
    parameters: Optional[OpenAIParameters] = OpenAIParameters()
    is_stream: Optional[bool] = False

    @validator("model_name", always=True)
    def validate_model_name(cls, value):
        """
        Validate that the model name is one of the allowed options.
        
        Args:
            value: The model name to validate.
        
        Returns:
            str: The validated model name.
        
        Raises:
            ValueError: If the model name is not an allowed value.
        """
        allowed_values = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value
    

def get_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate the cost of using the OpenAI API based on token usage and model.
    
    Args:
        input_tokens (int): Number of tokens in the input.
        output_tokens (int): Number of tokens in the output.
        model_name (str): Identifier of the model used.
    
    Returns:
        float: The calculated cost for the API usage.
    """
    if model_name == "gpt-3.5-turbo":
        return 0.0000015 * input_tokens + 0.000002 * output_tokens
    if model_name == "gpt-4":
        return 0.00003 * input_tokens + 0.00006 * output_tokens
    if model_name == "gpt-3.5-turbo-16k":
        return 0.00003 * input_tokens + 0.00004 * output_tokens
    return None

def get_tokens(chat_input: str, model_name: str) -> int:
    """
    Determine the number of tokens in a given input string using the specified model’s tokenizer.
    
    Args:
        chat_input (str): Text to be tokenized.
        model_name (str): Identifier of the model, determines tokenizer used.
    
    Returns:
        int: Number of tokens in the input string.
    """
    tokenizer = tiktoken.encoding_for_model(model_name)
    return len(tokenizer.encode(chat_input))
    
def generate_stream_response(response: dict, data: OpenAIRequest):
    """
    Generate stream responses, yielding chat output or tokens and cost information at stream end.
    
    Args:
        response (dict): Dictionary containing chunks of responses from the OpenAI API.
        data (OpenAIRequest): OpenAIRequest object containing necessary parameters for the API call.
    
    Yields:
        str: A chunk of chat output or, at stream end, tokens counts and cost information.
    """
    chat_output = ""
    for chunk in response:
        if (
            chunk["choices"][0]["finish_reason"] != "stop"
            and chunk["choices"][0]["finish_reason"] != "length"
        ):
            chunk_content = chunk["choices"][0]["delta"]["content"]
            chat_output += chunk_content
            yield chunk_content
        else:
            input_tokens = get_tokens(data.chat_input, data.model_name)
            output_tokens = get_tokens(chat_output, data.model_name)
            cost = get_cost(input_tokens, output_tokens, data.model_name)
            yield f"{end_token},{input_tokens},{output_tokens},{cost}"  # json


@router.post("/openai")
async def openai_chat_endpoint(data: OpenAIRequest):
    """
    FastAPI endpoint to interact with the OpenAI API for chat completions.
    
    Args:
        data (OpenAIRequest): OpenAIRequest object containing necessary parameters for the API call.
    
    Returns:
        Union[StreamingResponse, dict]: Streaming response if is_stream is True, otherwise a dict with chat and token data.
    """
    try:
        openai.api_key = data.api_key
        response = openai.ChatCompletion.create(
            model=data.model_name,
            messages=[
                {
                    "role": "user",
                    "content": data.chat_input,
                }
            ],
            temperature=data.parameters.temperature,
            max_tokens=data.parameters.max_tokens,
            top_p=data.parameters.top_p,
            frequency_penalty=data.parameters.frequency_penalty,
            presence_penalty=data.parameters.presence_penalty,
            stream=data.is_stream,
        )

        if data.is_stream:
            return StreamingResponse(generate_stream_response(response, data))
        
        input_tokens = get_tokens(data.chat_input, data.model_name)
        output_tokens = get_tokens(
            response["choices"][0]["message"]["content"], data.model_name
        )

        import random, time # delete

        data = {
            "id": random.randint(0, 1000),
            "chatInput": data.chat_input,
            "chatOutput": response["choices"][0]["message"]["content"],
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
            "cost": get_cost(input_tokens, output_tokens, data.model_name),
            "timestamp": time.time(),
            "modelName": data.model_name,
            "parameters": data.parameters.dict(),
        }
    except:
        openai.api_key = data["api_key"]

        response = openai.ChatCompletion.create(
            model=data["model_name"],
            messages=[
                {
                    "role": "user",
                    "content": data["chat_input"],
                }
            ],
            temperature=data["parameters"].temperature,
            max_tokens=data["parameters"].max_tokens,
            top_p=data["parameters"].top_p,
            frequency_penalty=data["parameters"].frequency_penalty,
            presence_penalty=data["parameters"].presence_penalty,
            stream=data["is_stream"],
        )

        if data["is_stream"]:
            return StreamingResponse(generate_stream_response(response, data))
        
        input_tokens = get_tokens(data["chat_input"], data["model_name"])
        output_tokens = get_tokens(
            response["choices"][0]["message"]["content"], data["model_name"]
        )

        import random, time # delete

        data = {
            "id": random.randint(0, 1000),
            "chatInput": data["chat_input"],
            "chatOutput": response["choices"][0]["message"]["content"],
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
            "cost": get_cost(input_tokens, output_tokens, data["model_name"]),
            "timestamp": time.time(),
            "modelName": data["model_name"],
            "parameters": data["parameters"].dict(),
        }

        append_log(data)
        return data