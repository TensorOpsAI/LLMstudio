import json

import requests

from llmstudio.config import TRACKING_HOST, TRACKING_PORT


class Tracker:
    def __init__(self):
        self._session = requests.Session()

    def log(self, data: dict):
        req = self._session.post(
            f"http://{TRACKING_HOST}:{TRACKING_PORT}/api/tracking/logs",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=100,
        )
        return req

    def update_session(self, data: dict):
        req = self._session.post(
            f"http://{TRACKING_HOST}:{TRACKING_PORT}/api/tracking/session",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=100,
        )
        return req

    def get_session(self, session_id: str):
        req = self._session.get(
            f"http://{TRACKING_HOST}:{TRACKING_PORT}/api/tracking/session/{session_id}",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req

    def add_extras(self, message_id: int):
        req = self._session.patch(
            f"http://{TRACKING_HOST}:{TRACKING_PORT}/api/tracking/session/{message_id}",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
        )
        return req


tracker = Tracker()
