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

    def session(self, data: dict):
        req = self._session.post(
            f"http://{TRACKING_HOST}:{TRACKING_PORT}/api/tracking/session",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=100,
        )
        return req


tracker = Tracker()
