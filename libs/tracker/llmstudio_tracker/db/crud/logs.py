from llmstudio_tracker.db.models.logs import LogDefault
from llmstudio_tracker.db.schemas.logs import LogDefaultCreate
from sqlalchemy.orm import Session


def get_project_by_name(db: Session, name: str):
    return db.query(LogDefault).filter(LogDefault.name == name).first()


def get_logs_by_session(db: Session, session_id: str, skip: int = 0, limit: int = 100):
    return (
        db.query(LogDefault)
        .filter(LogDefault.session_id == session_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def add_log(db: Session, log: LogDefaultCreate):
    db_log = LogDefault(**log.model_dump())
    db.add(db_log)
    db.commit()
    db.refresh(db_log)

    return db_log


def get_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(LogDefault).offset(skip).limit(limit).all()
