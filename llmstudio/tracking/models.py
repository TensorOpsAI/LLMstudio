from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from llmstudio.tracking.database import Base


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
    parameters = Column(JSON)
    metrics = Column(JSON)
