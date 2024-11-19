from datetime import datetime
import uuid
from sqlalchemy import Column, Integer, String, DateTime, JSON, func

from llmstudio_tracker.config import DB_TYPE
from llmstudio_tracker.db_utils import JSONEncodedDict
from llmstudio_tracker.database import Base


class LogDefault(Base):
    __tablename__ = "logs_default"

    log_id = Column(
        Integer,
        primary_key=True,
        default=lambda: uuid.uuid4().int % 10**12 if DB_TYPE == "bigquery" else None,
        index=True if DB_TYPE != "bigquery" else False
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now() if DB_TYPE != "bigquery" else None,
        default=datetime.now if DB_TYPE == "bigquery" else None
    )

    session_id = Column(String)
    chat_input = Column(String)
    chat_output = Column(String)
    context = Column(JSONEncodedDict if DB_TYPE == "bigquery" else JSON)
    provider = Column(String)
    model = Column(String)
    deployment = Column(String)
    parameters = Column(JSONEncodedDict if DB_TYPE == "bigquery" else JSON)
    metrics = Column(JSONEncodedDict if DB_TYPE == "bigquery" else JSON)
