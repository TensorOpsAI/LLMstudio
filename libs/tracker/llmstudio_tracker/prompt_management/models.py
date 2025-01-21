from datetime import datetime, timezone

from llmstudio_tracker.config import DB_TYPE
from llmstudio_tracker.database import Base
from llmstudio_tracker.db_utils import JSONEncodedDict
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
)


class PromptDefault(Base):
    __tablename__ = "prompts"

    if DB_TYPE == "bigquery":
        prompt_id = Column(
            Integer,
            primary_key=True,
            default=lambda: int(
                datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-1]
            ),
        )
        config = Column(JSONEncodedDict)
    else:
        prompt_id = Column(Integer, primary_key=True, index=True)
        config = Column(JSON)

    prompt = Column(String)
    is_active = Column(Boolean)
    name = Column(String)
    version = Column(Integer)
    label = Column(String)
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(timezone.utc),
        default=lambda: datetime.now(timezone.utc),
    )
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (UniqueConstraint("name", "label", name="uq_name_label"),)
