from typing import Optional

from pydantic import BaseModel, Field


class AnthropicParameters(BaseModel):
    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(300, ge=1, le=2048)
    top_p: Optional[float] = Field(0.999, ge=0, le=1)
    top_k: Optional[int] = Field(250, ge=1, le=500)
