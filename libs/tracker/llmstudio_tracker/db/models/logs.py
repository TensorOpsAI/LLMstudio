from datetime import datetime, timezone

from llmstudio_tracker.base_class import Base
from llmstudio_tracker.config import DB_TYPE
from llmstudio_tracker.db_utils import JSONEncodedDict
from sqlalchemy import JSON, Column, DateTime, Integer, String, Text


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
        extras = Column(JSONEncodedDict)
    else:
        log_id = Column(Integer, primary_key=True, index=True)
        created_at = Column(
            DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
        )
        session_id = Column(String(191))
        chat_input = Column(Text)
        chat_output = Column(Text)
        context = Column(JSON)
        provider = Column(String(191))
        model = Column(String(191))
        deployment = Column(String(191))
        parameters = Column(JSON)
        metrics = Column(JSON)
        extras = Column(JSON)
