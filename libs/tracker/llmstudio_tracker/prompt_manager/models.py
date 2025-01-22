import uuid
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
    func,
)


class PromptDefault(Base):
    __tablename__ = "prompts"

    if DB_TYPE == "bigquery":
        prompt_id = Column(
            String,
            primary_key=True,
            default=lambda: str(uuid.uuid4()),
        )
        config = Column(JSONEncodedDict, nullable=True)
    else:
        prompt_id = Column(
            String, primary_key=True, default=lambda: str(uuid.uuid4())
        )  # Generate UUID as a string
        config = Column(JSON, nullable=True)

    prompt = Column(String)
    is_active = Column(Boolean, default=True)
    name = Column(String)
    model = Column(String)
    provider = Column(String)
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

    __table_args__ = (
        UniqueConstraint("name", "provider", "model", "version", name="uq_name_label"),
    )

    @staticmethod
    def get_next_version(session, name, model, provider):
        """
        Get the next version number for a combination of name, model, and provider.
        """
        max_version = (
            session.query(func.max(PromptDefault.version))
            .filter_by(name=name, model=model, provider=provider)
            .scalar()
        )
        return (max_version or 0) + 1

    @classmethod
    def create_with_incremental_version(cls, session, **kwargs):
        """
        Create a new PromptDefault entry with an incremental version.
        """
        name = kwargs.get("name")
        model = kwargs.get("model")
        provider = kwargs.get("provider")
        if not all([name, model, provider]):
            raise ValueError("name, model, and provider must be provided")

        # Determine the next version
        kwargs["version"] = cls.get_next_version(session, name, model, provider)

        # Create and add the new instance
        instance = cls(**kwargs)
        session.add(instance)
        return instance
