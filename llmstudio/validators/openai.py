from typing import Optional

from pydantic import BaseModel, Field


class OpenAIParameters(BaseModel):
    """
    A Pydantic model for encapsulating parameters used in OpenAI API requests.

    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
        frequency_penalty (Optional[float]): Modifies the likelihood of tokens appearing based on their frequency.
        presence_penalty (Optional[float]): Adjusts the likelihood of new tokens appearing.
    """

    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=2048)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)
