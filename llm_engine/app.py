from fastapi import FastAPI, HTTPException, Request
from typing import Any, Dict, Optional, Callable

from llm_engine.config import LlmEngineConfig, Route, RouteType
from llm_engine.constants import LLM_ENGINE_HEALTH_ENDPOINT, LLM_ENGINE_ROUTE_BASE, VERSION
from llm_engine.providers import get_provider


class LlmEngineAPI(FastAPI):
    """
    Extends FastAPI to provide an API engine with dynamic routes based on the given configuration.

    Attributes:
        dynamic_routes (Dict[str, Route]): A dictionary mapping from route names to Route objects.
    """

    def __init__(self, config: LlmEngineConfig, *args: Any, **kwargs: Any):
        """
        Initialize the LlmEngineAPI instance.

        Args:
            config (LlmEngineConfig): The configuration object containing routes and other settings.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.dynamic_routes: Dict[str, Route] = {}
        self.set_dynamic_routes(config)
    
    def set_dynamic_routes(self, config: LlmEngineConfig) -> None:
        """
        Clears existing dynamic routes and sets new ones based on the provided configuration.

        Args:
            config (LlmEngineConfig): The configuration object containing routes and other settings.
        """
        self.dynamic_routes.clear()
        for route in config.routes:
            for provider in route.model_providers:
                self._add_dynamic_route(route, provider)

    def _add_dynamic_route(self, route: dict, provider: dict):
        """
        Internal method to add a dynamic route based on route and provider information.

        Args:
            route (dict): Dictionary containing information about the route.
            provider (dict): Dictionary containing information about the provider.
        """
        provider_name = provider.provider
        provider_config = provider.config or {}
        route_and_provider_name = f"{provider_name}-{route.name}"
        path = f"{LLM_ENGINE_ROUTE_BASE}{route_and_provider_name}"
        
        self.add_api_route(
            path=path,
            endpoint=self._route_type_to_endpoint(
                provider_name, provider_config, route.route_type
            ),
            methods=["POST"],
        )
        self.dynamic_routes[route_and_provider_name] = route.to_route(provider_name, path)

    def _route_type_to_endpoint(self, provider_name: str, provider_config: dict, route_type: RouteType) -> Callable:
        """
        Maps a route type to its corresponding endpoint callable based on provider information.

        Args:
            provider_name (str): The name of the provider.
            provider_config (dict): Configuration specific to the provider.
            route_type (RouteType): The type of the route.

        Returns:
            Callable: The callable endpoint to be used for the route.
        
        Raises:
            HTTPException: If the route type is unexpected for the given provider.
        """
        provider_to_factory = {
            RouteType.LLM_CHAT: "chat",
            RouteType.LLM_VALIDATION: "test",
        }
        
        factory = provider_to_factory.get(route_type)
        if factory:
            return self._create_generic_endpoint(factory,provider_name, provider_config)
        
        raise HTTPException(
            status_code=404,
            detail=f"Unexpected route type {route_type!r} for provider {provider_name!r}.",
        )

    def _create_generic_endpoint(self, method_name: str, provider_name: str, provider_config: str) -> Callable:
        """
        Creates a generic endpoint for the given method name and provider.

        Args:
            method_name (str): The method name that should be invoked on the provider.
            provider_name (str): The name of the provider.
            provider_config (str): Configuration specific to the provider.

        Returns:
            Callable: A generic endpoint that invokes the specified method on the provider.
        
        Raises:
            HTTPException: If the specified method is not found for the given provider.
        """
        async def _generic_endpoint(request: Request):
            payload = await request.json()
            api_key = payload.get('api_key', None)
            provider_instance = get_provider(provider_name)(provider_config, api_key)
            method = getattr(provider_instance, method_name, None)
            
            if not method:
                raise HTTPException(
                    status_code=404,
                    detail=f"Method {method_name!r} not found for provider {provider_name!r}.",
                )
            
            return await method(payload)
        
        return _generic_endpoint
    

    def get_dynamic_route(self, route_name: str) -> Optional[Route]:
        """
        Retrieves the dynamic route by its name.

        Args:
            route_name (str): The name of the route to retrieve.

        Returns:
            Optional[Route]: The Route object if found, None otherwise.
        """
        return self.dynamic_routes.get(route_name)
    
    def get_dynamic_routes(self):
        """
        Retrieves all dynamic routes.

        Returns:
            Dict[str, Route]: A dictionary of all dynamic routes, keyed by route name.
        """
        return self.dynamic_routes



def create_app_from_config(config: LlmEngineConfig) -> LlmEngineAPI:
    """
    Initializes and returns an LlmEngineAPI application based on the given configuration.

    Parameters:
    config (LlmEngineConfig): The configuration settings for initializing the LlmEngineAPI.

    Returns:
    LlmEngineAPI: An initialized LlmEngineAPI application.
    """
    app = LlmEngineAPI(
        config=config,
        title="llm_engine API",
        description="The core API for llm_engine.",
        version=VERSION,
    )

    @app.get(LLM_ENGINE_HEALTH_ENDPOINT)
    async def health():
        return {"status": "OK"}

    @app.get(LLM_ENGINE_ROUTE_BASE + "{route_name}")
    async def get_route(route_name: str) -> Route:
        if matched := app.get_dynamic_route(route_name):
            return matched

        raise HTTPException(
            status_code=404,
            detail=f"The route '{route_name}' is not present or active on the server. Please "
            "verify the route name.",
        )

    @app.get(LLM_ENGINE_ROUTE_BASE)
    async def search_routes(page_token: Optional[str] = None):
        # TODO: Implement better function
        routes = app.get_dynamic_routes()
        return routes

    return app
