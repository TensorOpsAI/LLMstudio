import datetime
from uuid import uuid4


def create_session_id() -> str:
    hash = str(uuid4())
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{hash}"