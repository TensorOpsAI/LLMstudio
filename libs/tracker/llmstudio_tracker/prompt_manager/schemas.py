from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel


class PromptInfo(BaseModel):
    prompt_id: Optional[str] = None
    name: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class PromptDefault(BaseModel):
    prompt_id: Optional[str] = None
    config: Optional[Dict] = {}
    prompt: str
    is_active: Optional[bool] = None
    name: str
    version: Optional[int] = None
    label: Optional[str] = "production"
    model: str
    provider: str
    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
