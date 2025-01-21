from datetime import datetime
from typing import Dict

from pydantic import BaseModel


class PromptDefault(BaseModel):
    prompt_id: str
    config: Dict
    prompt: str
    is_active: bool
    name: str
    version: int
    label: str
    updated_at: datetime
    created_at: datetime
