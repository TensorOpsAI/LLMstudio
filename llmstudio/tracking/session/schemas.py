from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class SessionDefaultBase(BaseModel):
    session_id: str
    chat_history: List[Dict[str, Any]] = None


class SessionDefault(SessionDefaultBase):
    created_at: datetime
    updated_at: datetime


class SessionDefaultCreate(SessionDefaultBase):
    pass
