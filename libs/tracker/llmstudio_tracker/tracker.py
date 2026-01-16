import json
from typing import Optional

import requests
from llmstudio_tracker.db.schemas.prompt_manager import PromptDefaultBase
from pydantic import BaseModel


class TrackingConfig(BaseModel):
    database_uri: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.url is None:
            if self.host is not None and self.port is not None:
                self.url = f"http://{self.host}:{self.port}"
            else:
                raise ValueError(
                    "You must provide either both 'host' and 'port', or 'url'."
                )


class Tracker:
    def __init__(self, tracking_config: TrackingConfig):
        self.tracking_url = tracking_config.url
        self._session = requests.Session()

    def log(self, data: dict):
        req = self._session.post(
            f"{self.tracking_url}/api/tracking/logs",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=100,
        )
        return req

    def get_logs(self):
        req = self._session.get(
            f"{self.tracking_url}/api/tracking/logs",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req

    def get_session_logs(self, session_id: str):
        req = self._session.get(
            f"{self.tracking_url}/api/tracking/logs/{session_id}",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req

    def update_session(self, data: dict):
        req = self._session.post(
            f"{self.tracking_url}/api/tracking/session",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=100,
        )
        return req

    def get_session(self, session_id: str):
        req = self._session.get(
            f"{self.tracking_url}/api/tracking/session/{session_id}",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req

    def add_extras(self, message_id: int):
        req = self._session.patch(
            f"{self.tracking_url}/api/tracking/session/{message_id}",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req


class PromptManager:
    def __init__(self, tracking_config: TrackingConfig):
        self.tracking_url = tracking_config.url
        self._session = requests.Session()

    def add_prompt(self, prompt: PromptDefaultBase):
        req = self._session.post(
            f"{self.tracking_url}/api/tracking/add/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump_json(),
            timeout=100,
        )
        return req

    def delete_prompt(self, prompt: PromptDefaultBase):
        req = self._session.delete(
            f"{self.tracking_url}/api/tracking/delete/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump_json(),
            timeout=100,
        )
        return req

    def update_prompt(self, prompt: PromptDefaultBase):
        req = self._session.patch(
            f"{self.tracking_url}/api/tracking/update/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump_json(),
            timeout=100,
        )
        return req

    def get_prompt(
        self,
        prompt_id: str = None,
        name: str = None,
        model: str = None,
        provider: str = None,
    ):

        data = {
            "prompt_id": prompt_id,
            "name": name,
            "model": model,
            "provider": provider,
        }

        req = self._session.get(
            f"{self.tracking_url}/api/tracking/get/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
            data=json.dumps(data),
        )
        return req
