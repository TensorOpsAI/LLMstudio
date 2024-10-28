from llmstudio_tracker.session import models, schemas
from sqlalchemy.orm import Session


def get_project_by_name(db: Session, name: str):
    return db.query(models.Project).filter(models.Project.name == name).first()


def get_session_by_session_id(
    db: Session, session_id: str, skip: int = 0, limit: int = 100
):
    return (
        db.query(models.SessionDefault)
        .filter(models.SessionDefault.session_id == session_id)
        .order_by(models.SessionDefault.created_at.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_session_by_message_id(db: Session, message_id: int):
    return (
        db.query(models.SessionDefault)
        .filter(models.SessionDefault.message_id == message_id)
        .first()
    )


def add_session(db: Session, session: schemas.SessionDefaultCreate):
    db_session = models.SessionDefault(**session.dict())

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


def upsert_session(db: Session, session: schemas.SessionDefaultCreate):
    return add_session(db, session)
