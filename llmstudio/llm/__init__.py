import requests


class LLM:
    def __init__(self, model_id: str, **kwargs):
        self.provider, self.model = model_id.split("/")
        self.api_key = kwargs.get("api_key")
        self.api_endpoint = kwargs.get("api_endpoint")
        self.api_version = kwargs.get("api_version")

    def chat(self, input: str, is_stream: bool = False, **kwargs):
        response = requests.post(
            f"http://localhost:8000/api/engine/chat/{self.provider}",
            json={
                "model": self.model,
                "api_key": self.api_key,
                "api_secret": self.api_endpoint,
                "api_region": self.api_version,
                "chat_input": input,
                "is_stream": is_stream,
                **kwargs,
            },
            stream=is_stream,
            headers={"Content-Type": "application/json"},
        )

        response.raise_for_status()

        if is_stream:
            return self.generate_chat(response)
        else:
            return response.json()

    def generate_chat(self, response):
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")
