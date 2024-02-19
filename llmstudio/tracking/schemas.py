from datetime import datetime
from typing import List, Dict

from pydantic import BaseModel

# BaseModels


class ProjectBase(BaseModel):
    name: str
    description: str = None


class ProjectCreate(ProjectBase):
    pass


class SessionBase(BaseModel):
    name: str
    description: str = None


class SessionCreate(SessionBase):
    pass


class LogBase(BaseModel):
    input: str = None
    output: str = None
    provider: str = None
    model: str = None
    parameters: dict = None
    metrics: dict = None


class LogCreate(LogBase):
    pass


class Log(LogBase):
    log_id: int
    session_id: int
    created_at: datetime


class Session(SessionBase):
    session_id: int
    project_id: int
    created_at: datetime
    logs: list[Log] = []


class Project(ProjectBase):
    project_id: int
    created_at: datetime
    sessions: list[Session] = []


class LogDefaultBase(BaseModel):
    chat_input: str = None
    chat_output: str = None
    provider: str = None
    model: str = None
    parameters: dict = None
    metrics: dict = None


class LogDefault(LogDefaultBase):
    log_id: int
    created_at: datetime


class LogDefaultCreate(LogDefaultBase):
    pass


class DashboardMetrics(BaseModel):
    request_by_provider: List[Dict[str, int]]
    request_by_model: List[Dict[str, int]]
    total_cost_by_provider: List[Dict]
    total_cost_by_model: List[Dict]
    average_latency: List[Dict]
    average_ttft: List[Dict]
    average_itl: List[Dict]
    average_tps: List[Dict]
