from llmstudio_tracker.db.models.session import SessionDefault
from llmstudio_tracker.db.schemas.session import SessionDefaultCreate
from sqlalchemy.orm import Session


def get_session_by_session_id(
    db: Session, session_id: str, skip: int = 0, limit: int = 100
):
    return (
        db.query(SessionDefault)
        .filter(SessionDefault.session_id == session_id)
        .order_by(SessionDefault.created_at.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_session_by_message_id(db: Session, message_id: int):
    return (
        db.query(SessionDefault).filter(SessionDefault.message_id == message_id).first()
    )


def add_session(db: Session, session: SessionDefaultCreate):
    db_session = session.SessionDefault(**session.model_dump())

    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def update_session(db: Session, message_id: int, extras: dict):
    existing_session = get_session_by_message_id(db, message_id)
    existing_session.extras = extras
    db.commit()
    db.refresh(existing_session)
    return existing_session


def upsert_session(db: Session, session: SessionDefaultCreate):
    return add_session(db, session)
