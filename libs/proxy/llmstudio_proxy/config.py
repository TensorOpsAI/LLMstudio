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
    "LLMSTUDIO_ENGINE_PORT": str(assign_port(50001)),
}

for key, default in defaults.items():
    os.environ[key] = os.getenv(key, default)

ENGINE_HOST = os.environ["LLMSTUDIO_ENGINE_HOST"]
ENGINE_PORT = os.environ["LLMSTUDIO_ENGINE_PORT"]
