import json

import requests
from llmstudio_tracker.prompt_management.schemas import PromptDefault
from llmstudio_tracker.tracker import TrackingConfig


class PromptManagement:
    def __init__(self, tracking_config: TrackingConfig):
        self.tracking_url = tracking_config.url
        self._session = requests.Session()

    def add_prompt(self, prompt: PromptDefault):
        req = self._session.post(
            f"{self.tracking_url}/api/tracking/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump(),
            timeout=100,
        )
        return req

    def delete_prompt(self, prompt: PromptDefault):
        req = self._session.delete(
            f"{self.tracking_url}/api/tracking/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump(),
            timeout=100,
        )
        return req

    def update_prompt(self, prompt: PromptDefault):
        req = self._session.get(
            f"{self.tracking_url}/api/tracking/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            data=prompt.model_dump(),
            timeout=100,
        )
        return req

    def get_prompt(self, prompt_id: str = None, name: str = None, label=None):

        data = {"prompt_id": prompt_id, "name": name, "label": label}

        req = self._session.get(
            f"{self.tracking_url}/api/tracking/prompt",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=100,
            data=json.dumps(data),
        )
        return req
