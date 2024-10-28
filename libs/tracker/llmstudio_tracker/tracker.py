import json
from typing import Optional

import requests
from pydantic import BaseModel


class TrackingConfig(BaseModel):
    database_uri: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if (self.host is None and self.port is None) and self.url is None:
            raise ValueError(
                "You must provide either both 'host' and 'port', or 'url', or 'database_uri'."
            )


class Tracker:
    def __init__(self, tracking_config: TrackingConfig):
        self.tracking_host = tracking_config.host
        self.tracking_port = tracking_config.port
        self._session = requests.Session()

    def log(self, data: dict):
        req = self._session.post(
            f"http://{self.tracking_host}:{self.tracking_port}/api/tracking/logs",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=100,
        )
        return req

    def get_logs(self):
        req = self._session.get(
            f"http://{self.tracking_host}:{self.tracking_port}/api/tracking/logs",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req

    def get_session_logs(self, session_id: str):
        req = self._session.get(
            f"http://{self.tracking_host}:{self.tracking_port}/api/tracking/logs/{session_id}",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req

    def update_session(self, data: dict):
        req = self._session.post(
            f"http://{self.tracking_host}:{self.tracking_port}/api/tracking/session",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=100,
        )
        return req

    def get_session(self, session_id: str):
        req = self._session.get(
            f"http://{self.tracking_host}:{self.tracking_port}/api/tracking/session/{session_id}",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req

    def add_extras(self, message_id: int):
        req = self._session.patch(
            f"http://{self.tracking_host}:{self.tracking_port}/api/tracking/session/{message_id}",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req
