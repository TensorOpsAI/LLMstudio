from fastapi import FastAPI, HTTPException, Request
from typing import Any, Dict, Optional, Callable

from LLMEngine.config import LLMEngineConfig, Route, RouteType
from LLMEngine.constants import LLM_ENGINE_HEALTH_ENDPOINT, LLM_ENGINE_ROUTE_BASE, VERSION
from LLMEngine.providers import get_provider


class LLMEngineAPI(FastAPI):
    def __init__(self, config: LLMEngineConfig, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.dynamic_routes: Dict[str, Route] = {}
        self.set_dynamic_routes(config)
    
    def set_dynamic_routes(self, config: LLMEngineConfig) -> None:
        self.dynamic_routes.clear()
        for route in config.routes:
            for provider in route.model_providers:
                self._add_dynamic_route(route, provider)

    def _add_dynamic_route(self, route: dict, provider: dict):
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
        return self.dynamic_routes.get(route_name)
    
    def get_dynamic_routes(self):
        return self.dynamic_routes

def _create_chat_endpoint(provider_name, provider_config):
        async def _chat_endpoint(request: Request):
            payload = await request.json()
            api_key = payload.get('api_key', None)
            prov = get_provider(provider_name)(provider_config, api_key)
            return await prov.chat(payload)
        
        return _chat_endpoint
    
def _create_validation_endpoint(provider_name, provider_config):
        async def _validation_endpoint(request: Request):
            payload = await request.json()
            api_key = payload.get('api_key', None)
            prov = get_provider(provider_name)(provider_config, api_key)
            return await prov.test(payload)
        
        return _validation_endpoint

def _route_type_to_endpoint(provider_name : str, provider_config : dict, route_type : RouteType):
    provider_to_factory = {
        RouteType.LLM_CHAT: _create_chat_endpoint,
        RouteType.LLM_VALIDATION: _create_validation_endpoint,
    }
    if factory := provider_to_factory.get(route_type):
        return factory(provider_name, provider_config)

    raise HTTPException(
        status_code=404,
        detail=f"Unexpected route type {route_type!r} for provider {provider_name!r}.",
    )


def create_app_from_config(config: LLMEngineConfig) -> LLMEngineAPI:
    """
    Create the GatewayAPI app from the gateway configuration.
    """
    app = LLMEngineAPI(
        config=config,
        title="LLMEngine API",
        description="The core API for LLMEngine.",
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
