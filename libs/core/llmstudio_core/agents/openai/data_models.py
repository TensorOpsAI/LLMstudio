from enum import Enum
from typing import Dict, List, Optional

from llmstudio_core.agents.data_models import (
    AgentBase,
    Attachment,
    InputMessageBase,
    RunBase,
    ToolOutput,
    ToolResources,
)
from pydantic import BaseModel


class ResponseFormatType(str, Enum):
    """Enum for response format types."""

    AUTO = "auto"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ResponseFormat(BaseModel):
    """Defines how the model should format its output."""

    type: ResponseFormatType
    json_schema: Optional[Dict] = None  # Required if type=json_schema


class OpenAIAgent(AgentBase):
    thread_id: Optional[str] = None
    tool_resources: Optional[ToolResources] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    response_format: Optional[ResponseFormat] = None


class OpenAIRun(RunBase):
    run_id: str
    tool_outputs: Optional[List[ToolOutput]] = []


class OpenAIInputMessage(InputMessageBase):
    attachments: Optional[List[Attachment]] = []
