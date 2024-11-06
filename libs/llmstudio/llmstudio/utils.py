import datetime
from uuid import uuid4


def create_session_id() -> str:
    hash = str(uuid4())
    try:
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    except AttributeError:  # python < 3.12
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{hash}"
