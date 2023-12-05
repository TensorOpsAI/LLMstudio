from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from llmstudio.tracking.database import Base


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(Integer, primary_key=True, index=True)

    name = Column(String, unique=True, index=True)
    description = Column(String, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sessions = relationship("Session", back_populates="owner")


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(Integer, primary_key=True, index=True)

    name = Column(String, index=True)
    description = Column(String, index=True)
    project_id = Column(Integer, ForeignKey("projects.project_id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("Project", back_populates="sessions")
    logs = relationship("Log", back_populates="owner")


class Log(Base):
    __tablename__ = "logs"

    log_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.session_id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    input = Column(String)
    output = Column(String)
    provider = Column(String)
    model = Column(String)
    parameters = Column(JSON)
    metrics = Column(JSON)

    owner = relationship("Session", back_populates="logs")


class LogDefault(Base):
    __tablename__ = "logs_default"

    log_id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chat_input = Column(String)
    chat_output = Column(String)
    provider = Column(String)
    model = Column(String)
    parameters = Column(JSON)
    metrics = Column(JSON)
