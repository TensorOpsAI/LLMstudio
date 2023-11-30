from datetime import datetime

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

    class Config:
        orm_mode = True


class Session(SessionBase):
    session_id: int
    project_id: int
    created_at: datetime
    logs: list[Log] = []

    class Config:
        orm_mode = True


class Project(ProjectBase):
    project_id: int
    created_at: datetime
    sessions: list[Session] = []

    class Config:
        orm_mode = True
