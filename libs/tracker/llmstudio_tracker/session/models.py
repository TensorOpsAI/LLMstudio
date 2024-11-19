from datetime import datetime
import uuid
from sqlalchemy import JSON, Column, DateTime, Integer, String, func

from llmstudio_tracker.config import DB_TYPE
from llmstudio_tracker.database import Base
from llmstudio_tracker.db_utils import JSONEncodedDict


class SessionDefault(Base):
    __tablename__ = "sessions"
    message_id = Column(
        Integer,
        primary_key=True,
        default=lambda: uuid.uuid4().int % 10**12 if DB_TYPE == "bigquery" else None,
        index=True if DB_TYPE != "bigquery" else False
    )
    session_id = Column(String, index=True if DB_TYPE != "bigquery" else False)
    chat_history = Column(JSONEncodedDict if DB_TYPE == "bigquery" else JSON)
    extras = Column(JSONEncodedDict if DB_TYPE == "bigquery" else JSON)
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=datetime.now if DB_TYPE == "bigquery" else func.now(),
        default=datetime.now if DB_TYPE == "bigquery" else None,
        server_default=func.now() if DB_TYPE != "bigquery" else None
    )
    created_at = Column(
        DateTime(timezone=True),
        default=datetime.now if DB_TYPE == "bigquery" else None,
        server_default=func.now() if DB_TYPE != "bigquery" else None
    )
