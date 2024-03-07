from sqlalchemy.orm import Session

from llmstudio.tracking import models, schemas


def get_project_by_name(db: Session, name: str):
    return db.query(models.Project).filter(models.Project.name == name).first()


def get_logs_by_session(db: Session, session_id: str, skip: int = 0, limit: int = 100):
    return (
        db.query(models.LogDefault)
        .filter(models.LogDefault.session_id == session_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def add_log(db: Session, log: schemas.LogDefaultCreate):
    db_log = models.LogDefault(**log.dict())
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log


def get_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.LogDefault).offset(skip).limit(limit).all()
