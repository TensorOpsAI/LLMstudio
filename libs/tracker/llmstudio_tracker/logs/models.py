from datetime import datetime, timezone

from llmstudio_tracker.config import DB_TYPE
from llmstudio_tracker.database import Base
from llmstudio_tracker.db_utils import JSONEncodedDict
from sqlalchemy import JSON, Column, DateTime, Integer, String


class LogDefault(Base):
    __tablename__ = "logs_default"

    if DB_TYPE == "bigquery":
        log_id = Column(
            Integer,
            primary_key=True,
            default=lambda: int(
                datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-1]
            ),
        )
        created_at = Column(
            DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
        )
        session_id = Column(String)
        chat_input = Column(String)
        chat_output = Column(String)
        context = Column(JSONEncodedDict)
        provider = Column(String)
        model = Column(String)
        deployment = Column(String)
        parameters = Column(JSONEncodedDict)
        metrics = Column(JSONEncodedDict)
    else:
        log_id = Column(Integer, primary_key=True, index=True)
        created_at = Column(
            DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
        )
        session_id = Column(String)
        chat_input = Column(String)
        chat_output = Column(String)
        context = Column(JSON)
        provider = Column(String)
        model = Column(String)
        deployment = Column(String)
        parameters = Column(JSON)
        metrics = Column(JSON)
