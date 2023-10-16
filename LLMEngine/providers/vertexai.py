
from LLMEngine.config import VertexAIConfig, RouteConfig
from pydantic import BaseModel, Field, validator
from typing import Optional
import tiktoken
from fastapi.responses import StreamingResponse
import random, time
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextGenerationModel, ChatModel, CodeGenerationModel, CodeChatModel
from LLMEngine.providers.base_provider import BaseProvider

# TODO: Change to constants.py
end_token = "<END_TOKEN>"


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
    api_key: str
    model_name: str
    chat_input: str
    parameters: Optional[VertexAIParameters] = VertexAIParameters()
    is_stream: Optional[bool] = False

class VertexAIProvider(BaseProvider):

    def __init__(self, config: VertexAIConfig, api_key: dict):
         super().__init__()
         self.vertexai_config = validate_provider_config(config, api_key)

    async def chat(self, data: VertexAIRequest) -> dict:
        """
        FastAPI endpoint to interact with the VertexAI API for text generation or chat completions.
        Args:
            data (VertexAIRequest): Object containing necessary parameters for the API call.
        Returns:
            dict: A dictionary containing the chat input, chat output, tokens data, cost, and other metadata.
        """
        credentials = service_account.Credentials.from_service_account_info(self.vertexai_config['vertexai_api_key'])
        vertexai.init(project=data['api_key']["project_id"], credentials=credentials)

        # TODO: Change to constants.py
        model_map = {
            "text-bison": TextGenerationModel,
            "chat-bison": ChatModel,
            "code-bison": CodeGenerationModel,
            "codechat-bison": CodeChatModel
        }
        self.validate_model_field(data, model_map.keys())
        model_class = model_map.get(data['model_name'])
    
        # TODO: Change to constants.py
        input_arg_name_map = {
            TextGenerationModel: 'prompt',
            CodeGenerationModel: 'prefix',
            ChatModel: 'message',
            CodeChatModel: 'message'
        }

        input_arg_name = input_arg_name_map.get(model_class)

        kwargs = {
            "temperature": data['parameters']['temperature'],
            "max_output_tokens": data['parameters']['max_tokens'],
        }
        if data['model_name'] not in {"code-bison", "codechat-bison"}:
            kwargs.update({
                "top_p": data['parameters']['top_p'],
                "top_k": data['parameters']['top_k'],
            })

        if model_class in {TextGenerationModel, CodeGenerationModel}:
            model = model_class.from_pretrained(data['model_name'])
            response = predict(model, input_arg_name,
                            data['chat_input'], data['is_stream'], **kwargs)
        else:
            model = model_class.from_pretrained(data.model_name)
            response = chat_predict(model, data['chat_input'],
                                    data['is_stream'], **kwargs)

        if data['is_stream']:
            return StreamingResponse(generate_stream_response(response, data))

        data = {
            "id": random.randint(0, 1000),
            "chatInput": data['chat_input'],
            "chatOutput": response['text'],
            "inputTokens": len(data['chat_input']),
            "outputTokens": len(response['text']),
            "totalTokens": len(data['chat_input']) + len(response['text']),
            "cost": get_cost(len(data['chat_input']), len(response['text'])),
            "timestamp": time.time(),
            "modelName": data['model_name'],
            "parameters": data['parameters'],
        }

        return data


    # TODO: Request base url and headers based on api_type (not implemented)

def predict(model, input_str: str, chat_input: str, is_stream: bool, **kwargs):
    """
    Makes a prediction using the specified model.
    Args:
        model: The model to use for making predictions.
        arg_name (str): The name of the argument to pass the chat input as.
        chat_input (str): The input string for the chat.
        is_stream (bool): Whether to stream the response.
        **kwargs: Additional parameters for the prediction function.
    Returns:
        The prediction response.
    """
    args = {input_str: chat_input, **kwargs}
    if is_stream:
        return model.predict_streaming(**args)
    return model.predict(**args)


def chat_predict(model, input_str: str, chat_input: str, is_stream: bool, **kwargs):
    """
    Makes a prediction using the specified chat model.
    Args:
        model: The chat model to use for making predictions.
        arg_name (str): The name of the argument to pass the chat input as.
        chat_input (str): The input string for the chat.
        is_stream (bool): Whether to stream the response.
        **kwargs: Additional parameters for the chat prediction function.
    Returns:
        The chat prediction response.
    """
    args = {input_str: chat_input, **kwargs}
    chat = model.start_chat()
    if is_stream:
        return chat.send_message_streaming(**args)
    return model.send_message(**args)
def get_cost(input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost of using the OpenAI API based on token usage and model.
        
        Args:
            input_tokens (int): Number of tokens in the input.
            output_tokens (int): Number of tokens in the output.
            model_name (str): Identifier of the model used.
        
        Returns:
            float: The calculated cost for the API usage.
        """
        return 0.0000005 * (input_tokens + output_tokens)

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

def generate_stream_response(response: dict, data: VertexAIProvider):
        """
        Generate stream responses, yielding chat output or tokens and cost information at stream end.
        
        Args:
            response (dict): Dictionary containing chunks of responses from the Vertex AI API.
            data (VertexAIRequest): Object containing necessary parameters for the API call.
        
        Yields:
            str: A chunk of chat output or, at stream end, tokens counts and cost information.
        """
        chat_output = ""

        for chunk in response:
            print(chunk)
            print(type(chunk))
            chat_output += ""
            yield ""

        input_tokens = len(data.chat_input)
        output_tokens = len(chat_output)
        cost = get_cost(input_tokens, output_tokens)
        yield f"{end_token},{input_tokens},{output_tokens},{cost}"  # json



# TODO: Send to utils.py
def validate_provider_config(vertexai_config, api_key):
    if not (vertexai_config or api_key):
        raise ValueError(
            f"Config was not specified neither an api_key was provided."
        )
    if vertexai_config is None:
        vertexai_config = {}
    vertexai_config.setdefault('vertexai_api_key', api_key)
    return vertexai_config

