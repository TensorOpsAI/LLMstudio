from typing import Optional

from pydantic import BaseModel, Field


class VertexAIParameters(BaseModel):
    """
    A Pydantic model that encapsulates parameters used for VertexAI API requests.

    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
        top_k (Optional[float]): Sets the number of the most likely next tokens to filter for.
    """

    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(256, ge=1, le=1024)
    top_p: Optional[float] = Field(1, ge=0, le=1)
    top_k: Optional[float] = Field(40, ge=1, le=40)
