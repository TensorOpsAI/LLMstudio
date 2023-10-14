from enum import Enum
from typing import Union, List
from pydantic import BaseModel
from pydantic import ValidationError, root_validator, validator
from utils import is_valid_endpoint_name
# TODO: Change this section
from endpoints.chat.openai import OpenAIRequest
from endpoints.chat.vertexai import VertexAIRequest
from endpoints.chat.bedrock import BedrockRequest



class RouteType(str, Enum):
    LLM_CHAT = "api/llm/chat"
    LLM_VALIDATION = "api/llm/validation"


class Provider(str, Enum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    BEDROCK = "bedrock"

    @classmethod
    def values(cls):
        return {p.value for p in cls}
    
Providers = {
    Provider.OPENAI: OpenAIRequest,
    Provider.VERTEXAI: VertexAIRequest,
    Provider.BEDROCK: BedrockRequest,
}

class RouteConfig(BaseModel):
    name: str
    route_type: RouteType
    model: str
    model_instance = Providers[Provider(model)]
    
    @validator("name", pre=True)
    def validate_endpoint_name(cls, route_name):
        if not is_valid_endpoint_name(route_name):
            raise ValueError(
                "The route name provided contains disallowed characters for a url endpoint. "
                f"'{route_name}' is invalid. Names cannot contain spaces or any non "
                "alphanumeric characters other than hyphen and underscore."
            )
        return route_name

    @validator("model", pre=True)
    def validate_model(cls, model):
        if model not in Provider.values() :
            raise ValueError(
                "The model name provided is not valid. "
            )
        return model

# TODO
#    def to_route(self) -> "Route":
#        return Route(
#            name=self.name,
#            route_type=self.route_type,
#            model=RouteModelInfo(
#               name=self.model.name,
#                provider=self.model.provider,
#            ),
#            route_url=f"{MLFLOW_GATEWAY_ROUTE_BASE}{self.name}{MLFLOW_QUERY_SUFFIX}",
#        )
    

class GatewayConfig(BaseModel):
    routes: List[RouteConfig]
