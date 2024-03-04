from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# BaseModels
class LogDefaultBase(BaseModel):
    chat_input: str = None
    chat_output: str = None
    session_id: Optional[str] = None
    context: list[dict[str, str]] = None
    provider: str = None
    model: str = None
    parameters: dict = None
    metrics: dict = None


class LogDefault(LogDefaultBase):
    log_id: int
    created_at: datetime


class LogDefaultCreate(LogDefaultBase):
    pass
