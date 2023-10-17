from typing import Optional

from pydantic import BaseModel, Field


class ClaudeParameters(BaseModel):
    """
    Model for validating and storing parameters specific to Claude model.

    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
        top_k (Optional[float]): Sets the number of the most likely next tokens to filter for.
    """

    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(300, ge=1, le=2048)
    top_p: Optional[float] = Field(0.999, ge=0, le=1)
    top_k: Optional[int] = Field(250, ge=1, le=500)


class TitanParameters(BaseModel):
    """
    Model for validating and storing parameters specific to Titan model.

    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
    """

    temperature: Optional[float] = Field(0, ge=0, le=1)
    max_tokens: Optional[int] = Field(512, ge=1, le=4096)
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1)
