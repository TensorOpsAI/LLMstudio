from llmstudio_tracker.database import Base
from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.sql import func


class LogDefault(Base):
    __tablename__ = "logs_default"

    log_id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session_id = Column(String)

    chat_input = Column(String)
    chat_output = Column(String)
    context = Column(JSON)
    provider = Column(String)
    model = Column(String)
    deployment = Column(String)
    parameters = Column(JSON)
    metrics = Column(JSON)
