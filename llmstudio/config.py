import os
import socket

from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))


def assign_port(default_port=None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            if default_port is not None:
                s.bind(("", default_port))
                return default_port
            else:
                s.bind(("", 0))
                return s.getsockname()[1]
        except OSError:
            s.bind(("", 0))
            return s.getsockname()[1]


defaults = {
    "LLMSTUDIO_ENGINE_HOST": "localhost",
    "LLMSTUDIO_TRACKING_HOST": "localhost",
    "LLMSTUDIO_UI_HOST": "localhost",
    "LLMSTUDIO_ENGINE_PORT": str(assign_port(50001)),
    "LLMSTUDIO_TRACKING_PORT": str(assign_port(50002)),
    "LLMSTUDIO_UI_PORT": str(assign_port(50003)),
    "LLMSTUDIO_TRACKING_URI": "sqlite:///./llmstudio_mgmt.db",
}

for key, default in defaults.items():
    os.environ[key] = os.getenv(key, default)

ENGINE_PORT = os.environ["LLMSTUDIO_ENGINE_PORT"]
TRACKING_PORT = os.environ["LLMSTUDIO_TRACKING_PORT"]
UI_PORT = os.environ["LLMSTUDIO_UI_PORT"]
ENGINE_HOST = os.environ["LLMSTUDIO_ENGINE_HOST"]
TRACKING_HOST = os.environ["LLMSTUDIO_TRACKING_HOST"]
UI_HOST = os.environ["LLMSTUDIO_UI_HOST"]
TRACKING_URI = os.environ["LLMSTUDIO_TRACKING_URI"]

os.environ["NEXT_PUBLIC_LLMSTUDIO_ENGINE_PORT"] = ENGINE_PORT
os.environ["NEXT_PUBLIC_LLMSTUDIO_TRACKING_PORT"] = TRACKING_PORT
os.environ["NEXT_PUBLIC_LLMSTUDIO_ENGINE_HOST"] = ENGINE_HOST
os.environ["NEXT_PUBLIC_LLMSTUDIO_TRACKING_HOST"] = TRACKING_HOST
