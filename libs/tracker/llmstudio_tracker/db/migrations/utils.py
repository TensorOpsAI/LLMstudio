import logging
import subprocess
from pathlib import Path

from alembic import command
from alembic.config import Config

logging.basicConfig(level=logging.INFO)


def run_alembic_upgrade():
    try:
        logging.info("Running LLMstudio Tracker Migrations Alembic upgrade...")

        alembic_cfg = Config()
        this_dir = Path(__file__).resolve().parent
        alembic_cfg.set_main_option("script_location", str(this_dir))

        command.upgrade(alembic_cfg, "head")

        logging.info("Alembic: LLMstudio Tracker Migrations upgrade successful.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Alembic: LLMstudio Tracker Migrations upgrade failed: {e}")
        raise
