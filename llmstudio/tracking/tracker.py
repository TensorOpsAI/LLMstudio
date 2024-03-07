import json
import os

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))


class Tracker:
    def __init__(self):
        self._session = requests.Session()

    def log(self, data: dict):
        req = self._session.post(
            f"http://{os.getenv('LLMSTUDIO_TRACKING_HOST')}:{os.getenv('LLMSTUDIO_TRACKING_PORT')}/api/tracking/logs",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=100,
        )
        return req


tracker = Tracker()
