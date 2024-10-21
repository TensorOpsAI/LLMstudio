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
    "LLMSTUDIO_TRACKING_HOST": "localhost",
    "LLMSTUDIO_TRACKING_PORT": str(assign_port(50002)),
    "LLMSTUDIO_TRACKING_URI": "sqlite:///./llmstudio_mgmt.db",
}

for key, default in defaults.items():
    os.environ[key] = os.getenv(key, default)

TRACKING_HOST = os.environ["LLMSTUDIO_TRACKING_HOST"]
TRACKING_PORT = os.environ["LLMSTUDIO_TRACKING_PORT"]
TRACKING_URI = os.environ["LLMSTUDIO_TRACKING_URI"]
