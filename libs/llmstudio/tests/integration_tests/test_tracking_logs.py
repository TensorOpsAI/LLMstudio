import os
import uuid

import pytest

# Load .env
from dotenv import load_dotenv
from llmstudio.providers import LLM
from llmstudio.server import start_servers
from llmstudio_tracker.db.models.logs import LogDefault
from llmstudio_tracker.tracker import TrackingConfig
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

load_dotenv()


DATABASE_URL = os.environ["LLMSTUDIO_TRACKING_URI"]
LLMSTUDIO_TRACKING_HOST = os.environ["LLMSTUDIO_TRACKING_HOST"]
LLMSTUDIO_TRACKING_PORT = os.environ["LLMSTUDIO_TRACKING_PORT"]

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


@pytest.mark.parametrize(
    "provider, model, api_key_name",
    [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
    ],
)
def test_llm_tracking_logs(provider, model, api_key_name):
    session_id = str(uuid.uuid4())

    start_servers(proxy=False, tracker=True)

    tracking_config = TrackingConfig(
        host=LLMSTUDIO_TRACKING_HOST, port=LLMSTUDIO_TRACKING_PORT
    )

    llm = LLM(
        provider=provider,
        api_key=os.environ[api_key_name],
        session_id=session_id,
        tracking_config=tracking_config,
    )

    chat_request = {
        "chat_input": f"Hello, my name is Alice - session {session_id}",
        "model": model,
        "is_stream": False,
        "retries": 0,
        "parameters": {"temperature": 0, "max_tokens": 1000},
    }

    response = llm.chat(**chat_request)
    print(response)

    assert hasattr(response, "chat_output"), "Missing 'chat_output'"
    assert response.chat_output is not None, "'chat_output' is None"

    # DB: Check if row was logged
    db = Session()
    logs = (
        db.execute(select(LogDefault).where(LogDefault.session_id == session_id))
        .scalars()
        .all()
    )

    assert len(logs) == 1, "No log entry found for session"
    log = logs[0]

    assert log.chat_input == f"Hello, my name is Alice - session {session_id}"
    assert log.model == "gpt-4o-mini"
    assert log.session_id == session_id
    assert log.chat_output is not None
    assert isinstance(log.parameters, dict)
    db.close()
