import os
import subprocess
from pathlib import Path
import threading
import webbrowser

from llmstudio.config import UI_PORT


def run_ui_app():
    ui_dir = Path(os.path.join(os.path.dirname(__file__)))
    try:
        subprocess.run(["bun", "install", "--silent"], cwd=ui_dir, check=True)
        subprocess.run(
            ["bun", "dev", "--port", UI_PORT],
            cwd=ui_dir,
            check=True,
        )
    except Exception as e:
        print(f"Error running LLMstudio UI: {e}")
