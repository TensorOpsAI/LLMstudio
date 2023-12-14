import os
import subprocess
from pathlib import Path
import threading

UI_HOST = os.getenv("ENGINE_HOST", "localhost")
UI_PORT = int(os.getenv("UI_PORT", 8000))
UI_URL = f"http://{UI_HOST}:{UI_PORT}"
LOG_LEVEL = os.getenv("LOG_LEVEL", "critical")


def run_bun_in_thread():
    ui_dir = Path(os.path.join(os.path.dirname(__file__)))
    try:
        subprocess.run(["bun", "update"], cwd=ui_dir, check=True)
        subprocess.run(["bun", "run", "dev"], cwd=ui_dir, check=True)
    except Exception as e:
        print(f"Error running the UI app: {e}")


def run_ui_app():
    print(f"Running UI on {UI_HOST}:{UI_PORT}")  # Print statement added
    thread = threading.Thread(target=run_bun_in_thread)
    thread.start()
