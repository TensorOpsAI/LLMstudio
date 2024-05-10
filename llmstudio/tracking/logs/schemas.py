from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# BaseModels
class LogDefaultBase(BaseModel):
    chat_input: str = None
    chat_output: str = None
    session_id: Optional[str] = None
    context: List[Dict[str, Any]] = None
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
