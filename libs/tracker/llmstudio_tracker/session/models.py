from llmstudio_tracker.database import Base
from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.sql import func


class SessionDefault(Base):
    __tablename__ = "sessions"
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    chat_history = Column(JSON)
    extras = Column(JSON)
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), server_default=func.now()
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
