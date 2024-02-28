import os
import subprocess
from pathlib import Path
import threading


def run_bun_in_thread():
    ui_dir = Path(os.path.join(os.path.dirname(__file__)))
    try:
        subprocess.run(["bun", "update"], cwd=ui_dir, check=True)
        subprocess.run(
            ["bun", "run", "dev", "--port", os.getenv("LLMSTUDIO_UI_PORT")],
            cwd=ui_dir,
            check=True,
        )
    except Exception as e:
        print(f"Error running the UI app: {e}")


def run_ui_app():
    print(
        f"Running UI on http://localhost:{os.getenv('LLMSTUDIO_UI_PORT')}"
    )  # Print statement added
    thread = threading.Thread(target=run_bun_in_thread)
    thread.start()
