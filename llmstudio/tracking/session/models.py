from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.sql import func

from llmstudio.tracking.database import Base


class SessionDefault(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    chat_history = Column(JSON)
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
