from datetime import datetime, timezone

from llmstudio_tracker.config import DB_TYPE
from llmstudio_tracker.database import Base
from llmstudio_tracker.db_utils import JSONEncodedDict
from sqlalchemy import JSON, Column, DateTime, Integer, String


class SessionDefault(Base):
    __tablename__ = "sessions"

    if DB_TYPE == "bigquery":
        message_id = Column(
            Integer,
            primary_key=True,
            default=lambda: int(
                datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-1]
            ),
        )
        session_id = Column(String)
        chat_history = Column(JSONEncodedDict)
        extras = Column(JSONEncodedDict)

    else:
        message_id = Column(Integer, primary_key=True, index=True)
        session_id = Column(String, index=True)
        chat_history = Column(JSON)
        extras = Column(JSON)

    updated_at = Column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(timezone.utc),
        default=lambda: datetime.now(timezone.utc),
    )
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
