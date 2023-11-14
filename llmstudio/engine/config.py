import json
import os
import pathlib
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import pydantic
import yaml
from packaging import version
from pydantic import BaseModel, ValidationError, validator
from pydantic.json import pydantic_encoder

from llmstudio.engine.utils import (
    check_configuration_route_name_collisions,
    is_valid_endpoint_name,
)

IS_PYDANTIC_V2 = version.parse(pydantic.version.VERSION) >= version.parse("2.0")


class EngineConfig:
    def __init__(
        self,
        api_name="Engine",
        host="localhost",
        port=8000,
        localhost=True,
        config_path=os.path.join(os.path.dirname(__file__), "config.yaml"),
        health_endpoint="health",
        routes_endpoint="api/engine",
    ):
        self.api_name = api_name
        self.host = host
        self.port = port
        self.config_path = config_path
        self.localhost = localhost
        self.update_url()
        self.update_endpoints(health_endpoint, routes_endpoint)

    def update_url(self):
        """Update the URL based on the current host, port and localhost values."""
        self.url = f"http://{self.host}:{self.port}" if self.localhost else self.host

    def update_endpoints(self, health_endpoint, routes_endpoint):
        """Update the health and routes endpoints based on the current url."""
        self.health_endpoint = f"{self.url}/{health_endpoint}"
        self.routes_endpoint = f"{self.url}/{routes_endpoint}"


class RouteType(str, Enum):
    """
    Used for specifying various types of routes in an API.
    """

    LLM_CHAT = "chat"
    LLM_VALIDATION = "validation"


class Provider(str, Enum):
    """
    Enum class to represent different AI service providers.
    """

    OPENAI = "openai"
    VERTEXAI = "vertexai"
    BEDROCK = "bedrock"
    ANTHROPIC = "anthropic"

    @classmethod
    def values(cls):
        return {p.value for p in cls}


class OpenAIConfig(BaseModel):
    """
    OpenAIConfig is a class derived from BaseModel for handling the configuration needed for OpenAI API calls.

    Attributes:
    api_key (str): The API key for authentication.
    openai_api_type (str, optional): Type of the OpenAI API to use, default is 'openai'.
    openai_api_base (str, optional): Base URL for the OpenAI API, default is None.
    openai_api_version (str, optional): API version, default is None.
    openai_deployment_name (str, optional): Name of the deployment, default is None.
    openai_organization (str, optional): Name of the organization, default is None.
    """

    api_key: str
    openai_api_type: Optional[str] = "openai"
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None
    openai_deployment_name: Optional[str] = None
    openai_organization: Optional[str] = None

    @validator("api_key", pre=True)
    def validate_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class VertexAIConfig(BaseModel):
    """
    VertexAIConfig is a class derived from BaseModel for handling the configuration needed for VertexAI API calls.

    Attributes:
    api_key (dict): The JSON API key for authentication.
    """

    api_key: dict

    @validator("api_key", pre=True)
    def validate_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class BedrockConfig(BaseModel):
    """
    BedrockConfig is a class derived from BaseModel for handling the configuration needed for Bedrock API calls.

    Attributes:
    api_key (dict): The JSON API key for authentication.
    """

    api_key: dict

    @validator("api_key", pre=True)
    def validate_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class AnthropicConfig(BaseModel):
    api_key: dict

    @validator("api_key", pre=True)
    def validate_api_key(cls, value):
        return _resolve_api_key_from_input(value)


provider_configs = {
    Provider.OPENAI: OpenAIConfig,
    Provider.VERTEXAI: VertexAIConfig,
    Provider.BEDROCK: BedrockConfig,
    Provider.ANTHROPIC: AnthropicConfig,
}


def _resolve_api_key_from_input(api_key_input):
    """
    Resolves the provided API key.

    Input formats accepted:

    - Path to a file as a string which will have the key loaded from it
    - environment variable name that stores the api key
    - the api key itself
    """

    if isinstance(api_key_input, dict):
        api_key_input = api_key_input.get("api_key")

    # try reading as an environment variable
    if api_key_input.startswith("$"):
        env_var_name = api_key_input[1:]
        if env_var := os.getenv(env_var_name):
            return env_var
        else:
            raise ValueError(f"Environment variable {env_var_name!r} is not set")

    # try reading from a local path
    file = pathlib.Path(api_key_input)
    if file.is_file():
        return file.read_text()

    # if the key itself is passed, return
    return api_key_input


class ModelProvider(BaseModel):
    """
    Represents a provider for machine learning models along with its configuration.

    Attributes:
        provider (Union[str, Provider]): The provider of the machine learning model.
        config (Optional[Union[OpenAIConfig, ...]]): The configuration for the selected provider.
    """

    provider: Union[str, Provider]
    config: Optional[
        Union[
            OpenAIConfig,
            VertexAIConfig,
            BedrockConfig,
            AnthropicConfig,
        ]
    ] = None

    @validator("provider", pre=True)
    def validate_provider(cls, value):
        """
        Validates the 'provider' field.

        Parameters:
            value: The value to validate, can be either a string or an instance of the Provider enum.

        Returns:
            Provider: A valid Provider enum instance.

        Raises:
            ValueError: If the provided value is not a valid provider.
        """
        if isinstance(value, Provider):
            return value
        formatted_value = value.replace("-", "_").upper()
        if formatted_value in Provider.__members__:
            return Provider[formatted_value]
        raise ValueError(f"The provider '{value}' is not supported.")

    @classmethod
    def _validate_config(cls, info, values):
        """
        Internal method to validate the 'config' field based on the provided 'provider'.

        Parameters:
            info: The configuration information to validate.
            values: Dictionary containing other field values of the class, primarily used to fetch 'provider'.

        Returns:
            The validated configuration as an instance of the appropriate configuration class.

        Raises:
            ValueError: If a valid 'provider' is not provided.
        """
        if provider := values.get("provider"):
            config_type = provider_configs[provider]
            return config_type(**info)

        raise ValueError("A provider must be provided for each gateway route.")

    if IS_PYDANTIC_V2:

        @validator("config", pre=True)
        def validate_config(cls, info, values):
            return cls._validate_config(info, values)

    else:

        @validator("config", pre=True)
        def validate_config(cls, config, values):
            return cls._validate_config(config, values)


class RouteConfig(BaseModel):
    """
    Represents the configuration for a single route.

    Attributes:
        name (str): The name of the route.
        route_type (RouteType): The type of the route, as defined in the RouteType enum.
        providers (List[ModelProvider]): A list of model providers for the route.
    """

    name: str
    route_type: RouteType
    providers: List[ModelProvider]

    @validator("name", pre=True)
    def validate_endpoint_name(cls, route_name):
        """
        Validates that the provided route name is a valid endpoint name.

        Args:
            route_name (str): The name of the route.

        Returns:
            str: The validated name.

        Raises:
            ValueError: If the name contains invalid characters.
        """
        if not is_valid_endpoint_name(route_name):
            raise ValueError(
                "The route name provided contains disallowed characters for a url endpoint. "
                f"'{route_name}' is invalid. Names cannot contain spaces or any non "
                "alphanumeric characters other than hyphen and underscore."
            )
        return route_name

    @validator("providers", pre=True)
    def validate_model(cls, providers):
        """
        Validates that the provided model providers list is not empty and contains valid entries.

        Args:
            providers (List[Dict]): A list of dictionaries containing model provider information.

        Returns:
            List[Dict]: The validated list of model providers.

        Raises:
            ValueError: If the list is empty or contains invalid providers.
        """
        if not providers:
            raise ValueError(
                "No model providers were provided for the route. Please provide at least one model provider."
            )
        for provider in providers:
            if provider:
                model_instance = ModelProvider(**provider)
                if model_instance.provider not in Provider.values():
                    raise ValueError(
                        f"The provider entry for {model_instance.provider} is incorrect. Providers accepted are {Provider.values()}"
                    )
        return providers

    @validator("route_type", pre=True)
    def validate_route_type(cls, value):
        """
        Validates that the provided route type is a valid RouteType enum value.

        Args:
            value (str): The type of the route.

        Returns:
            str: The validated route type.

        Raises:
            ValueError: If the route_type is not a valid RouteType enum value.
        """
        if value in RouteType._value2member_map_:
            return value
        raise ValueError(
            f"The route_type '{value}' is not supported. Please use one of {RouteType._value2member_map_}"
        )

    def to_route(self, name, route_url) -> "Route":
        """
        Converts the configuration to a Route object.

        Args:
            name (str): The name of the route.
            route_url (str): The URL for the route.

        Returns:
            Route: A Route object containing the route's configuration.
        """
        return Route(
            name=name,
            route_type=self.route_type,
            route_url=route_url,
        )


class Route(BaseModel):
    """
    A class to represent a routing information.

    Attributes:
        name (str): The name of the route.
        route_type (str): The type of the route, usually represented as a string-based identifier.
        route_url (str): The URL pattern for the route.
    """

    name: str
    route_type: str
    route_url: str


class EngineRouteConfig(BaseModel):
    routes: List[RouteConfig]


def _load_route_config(path: Union[str, Path]) -> EngineRouteConfig:
    """
    Reads the gateway configuration yaml file from the storage location and returns an instance
    of the configuration RouteConfig class
    """
    if isinstance(path, str):
        path = Path(path)
    try:
        configuration = yaml.safe_load(path.read_text())
    except Exception as e:
        raise ValueError(f"The file at {path} is not a valid yaml file") from e
    check_configuration_route_name_collisions(configuration)
    try:
        return EngineRouteConfig(**configuration)
    except ValidationError as e:
        raise ValueError(f"The gateway configuration is invalid: {e}") from e


def _save_route_config(config: EngineRouteConfig, path: Union[str, Path]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.write_text(
        yaml.safe_dump(json.loads(json.dumps(config.dict(), default=pydantic_encoder)))
    )


def _validate_config(config_path: str) -> EngineRouteConfig:
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist")

    try:
        return _load_route_config(config_path)
    except ValidationError as e:
        raise ValueError(f"Invalid gateway configuration: {e}") from e
