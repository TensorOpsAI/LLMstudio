from fastapi import FastAPI, HTTPException, Request
from typing import Any, Dict, Optional

from LLMEngine.config import LLMEngineConfig, Route
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
                provider_name = provider.provider
                provider_config = provider.config if provider.config else {}
                route_and_provider_name = f"{provider_name}-{route.name}"
                path=f"{LLM_ENGINE_ROUTE_BASE}{route_and_provider_name}"
                self.add_api_route(
                    path=path,
                    endpoint=self._create_dynamic_chat_endpoint(provider_name, provider_config),
                    methods=["POST"],
                )
                self.dynamic_routes[route_and_provider_name] = route.to_route(provider_name, path)

    def get_dynamic_route(self, route_name: str) -> Optional[Route]:
        return self.dynamic_routes.get(route_name)
    
    def get_dynamic_routes(self):
        return self.dynamic_routes

    def _create_dynamic_chat_endpoint(self, provider_name, provider_config):
            async def _dynamic_chat_endpoint(request: Request):
                payload = await request.json()
                api_key = payload.get('api_key', None)
                prov = get_provider(provider_name)(provider_config, api_key)
                return await prov.chat(payload)
            
            return _dynamic_chat_endpoint

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
