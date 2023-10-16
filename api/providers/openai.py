
from api.config import OpenAIConfig, RouteConfig
from pydantic import BaseModel, Field
from typing import Optional
import openai
import tiktoken
from fastapi.responses import StreamingResponse
import random, time

# TODO: Change to constants.py
end_token = "<END_TOKEN>"


# TODO: Change to constants.py
OPENAI_PRICING_DICT = {
            "gpt-3.5-turbo": {"input_tokens": 0.0000015, "output_tokens": 0.000002},
            "gpt-4": {"input_tokens": 0.00003, "output_tokens": 0.00006},
            "gpt-3.5-turbo-16k": {"input_tokens": 0.00003, "output_tokens": 0.00004},
        }

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

class OpenAIProvider(BaseModel):

    api_key: str
    model_name: str
    chat_input: str
    parameters: Optional[OpenAIParameters] = OpenAIParameters()
    is_stream: Optional[bool] = False


    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, OpenAIConfig):
            # Should be unreachable
            raise ValueError(
                "Invalid config type {config.model.config}"
            )
        self.openai_config: OpenAIConfig = config.model.config
    
    # TODO: Request base url and headers based on api_type (not implemented)

    async def chat(self, data) -> dict:
        from fastapi import HTTPException
        from fastapi.encoders import jsonable_encoder

        # TODO: Change method to base_provider.py
        if data.model_name not in ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_name: {data.model_name}",
            )
        data = jsonable_encoder(data, exclude_none=True)
        openai.api_key = self.openai_config.openai_api_key
        response = openai.ChatCompletion.create(
        model=data["model_name"],
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
        return data

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
        return OPENAI_PRICING_DICT[model_name]["input_tokens"] * input_tokens + OPENAI_PRICING_DICT[model_name]["output_tokens"] * output_tokens

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

def generate_stream_response(response: dict, data: OpenAIProvider):
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



