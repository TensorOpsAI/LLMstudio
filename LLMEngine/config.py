from enum import Enum
import pathlib
import os
import pydantic
from packaging import version
from pathlib import Path
import yaml
import json
from typing import Union, List, Optional
from pydantic.json import pydantic_encoder
from pydantic import BaseModel
from pydantic import ValidationError, root_validator, validator
from utils import is_valid_endpoint_name


# TODO: Change to constants
LLM_ENGINE_ROUTE_BASE = "/gateway/"


IS_PYDANTIC_V2 = version.parse(pydantic.version.VERSION) >= version.parse("2.0")

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

class OpenAIConfig(BaseModel):
    openai_api_key: str
    openai_api_type: Optional[str] = 'openai'
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None
    openai_deployment_name: Optional[str] = None
    openai_organization: Optional[str] = None

    # pylint: disable=no-self-argument
    @validator("openai_api_key", pre=True)
    def validate_openai_api_key(cls, value):
        return _resolve_api_key_from_input(value)

class VertexAIConfig(BaseModel):
    vertexai_api_key: str

    # pylint: disable=no-self-argument
    @validator("vertexai_api_key", pre=True)
    def validate_vertexai_api_key(cls, value):
        return _resolve_api_key_from_input(value)
    

class BedrockConfig(BaseModel):
    bedrock_api_key: str

    # pylint: disable=no-self-argument
    @validator("bedrock_api_key", pre=True)
    def validate_bedrock_api_key(cls, value):
        return _resolve_api_key_from_input(value)
    
    
provider_configs = {
    Provider.OPENAI: OpenAIConfig,
    Provider.VERTEXAI: VertexAIConfig,
    Provider.BEDROCK: BedrockConfig,
}

def _resolve_api_key_from_input(api_key_input):
    """
    Resolves the provided API key.

    Input formats accepted:

    - Path to a file as a string which will have the key loaded from it
    - environment variable name that stores the api key
    - the api key itself
    """

    if not isinstance(api_key_input, str):
        raise ValueError(
            "The api key provided is not a string. Please provide either an environment "
            "variable key, a path to a file containing the api key, or the api key itself"
        )

    # try reading as an environment variable
    if api_key_input.startswith("$"):
        env_var_name = api_key_input[1:]
        if env_var := os.getenv(env_var_name):
            return env_var
        else:
            raise ValueError(
                f"Environment variable {env_var_name!r} is not set"
            )

    # try reading from a local path
    file = pathlib.Path(api_key_input)
    if file.is_file():
        return file.read_text()

    # if the key itself is passed, return
    return api_key_input

class ModelProvider(BaseModel):
    provider: Union[str, Provider]
    config: Optional[
        Union[
            OpenAIConfig,
            VertexAIConfig,
            BedrockConfig,
        ]
    ] = None

    @validator("provider", pre=True)
    def validate_provider(cls, value):
        if isinstance(value, Provider):
            return value
        formatted_value = value.replace("-", "_").upper()
        if formatted_value in Provider.__members__:
            return Provider[formatted_value]
        raise ValueError(f"The provider '{value}' is not supported.")

    @classmethod
    def _validate_config(cls, info, values):
        if provider := values.get("provider"):
            config_type = provider_configs[provider]
            return config_type(**info)

        raise ValueError(
            "A provider must be provided for each gateway route."
        )

    if IS_PYDANTIC_V2:
        @validator("config", pre=True)
        def validate_config(cls, info, values):
            return cls._validate_config(info, values)
    else:
        @validator("config", pre=True)
        def validate_config(cls, config, values):
            return cls._validate_config(config, values)
        


class RouteConfig(BaseModel):
    name: str
    route_type: RouteType
    model_providers: List[ModelProvider]
    
    @validator("name", pre=True)
    def validate_endpoint_name(cls, route_name):
        if not is_valid_endpoint_name(route_name):
            raise ValueError(
                "The route name provided contains disallowed characters for a url endpoint. "
                f"'{route_name}' is invalid. Names cannot contain spaces or any non "
                "alphanumeric characters other than hyphen and underscore."
            )
        return route_name

    @validator("model_providers", pre=True)
    def validate_model(cls, model_providers):
        if not model_providers:
            raise ValueError(
                "No model providers were provided for the route. Please provide at least one model provider."
            )
        for model_provider in model_providers:
            if model_provider:
                model_instance = ModelProvider(**model_provider)
                if model_instance.provider not in Provider.values():
                    raise ValueError(
                        f"The provider entry for {model_instance.provider} is incorrect. Providers accepted are {Provider.values()}"
                    )
        return model_providers
    
    @validator("route_type", pre=True)
    def validate_route_type(cls, value):
        if value in RouteType._value2member_map_:
            return value
        raise ValueError(f"The route_type '{value}' is not supported. Please use one of {RouteType._value2member_map_}")
    
    def to_routes(self) -> List["Route"]:
        return [model_provider.config.to_route() for model_provider in self.model_providers]

    def to_route(self) -> "Route":
            return Route(
                name=self.name,
                route_type=self.route_type,
                route_url=f"{LLM_ENGINE_ROUTE_BASE}{self.name}",
            )

class Route(BaseModel):
    name: str
    route_type: str
    route_url: str

    class Config:
        schema_extra = {
            "example": {
                "name": "openai",
                "route_type": "llm/v1/completions",
                "route_url": "/llmengine/openai",
            }
        }
    

class LLMEngineConfig(BaseModel):
    routes: List[RouteConfig]


def _load_route_config(path: Union[str, Path]) -> LLMEngineConfig:
    """
    Reads the gateway configuration yaml file from the storage location and returns an instance
    of the configuration RouteConfig class
    """
    if isinstance(path, str):
        path = Path(path)
    try:
        configuration = yaml.safe_load(path.read_text())
    except Exception as e:
        raise ValueError(
            f"The file at {path} is not a valid yaml file"
        ) from e
    check_configuration_route_name_collisions(configuration)
    try:
        return LLMEngineConfig(**configuration)
    except ValidationError as e:
        raise ValueError(
            f"The gateway configuration is invalid: {e}"
        ) from e
    

def _save_route_config(config: LLMEngineConfig, path: Union[str, Path]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.write_text(yaml.safe_dump(json.loads(json.dumps(config.dict(), default=pydantic_encoder))))


def _validate_config(config_path: str) -> LLMEngineConfig:
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist")

    try:
        return _load_route_config(config_path)
    except ValidationError as e:
        raise ValueError(f"Invalid gateway configuration: {e}") from e


# TODO: Add to utils.py
def check_configuration_route_name_collisions(config):
    if len(config["routes"]) < 2:
        return
    names = [route["name"] for route in config["routes"]]
    if len(names) != len(set(names)):
        raise ValueError(
            "Duplicate names found in route configurations. Please remove the duplicate route "
            "name from the configuration to ensure that route endpoints are created properly."
        )