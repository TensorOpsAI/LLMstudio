from sqlalchemy.orm import Session

from llmstudio.tracking.session import models, schemas


def get_project_by_name(db: Session, name: str):
    return db.query(models.Project).filter(models.Project.name == name).first()


def get_session_by_id(db: Session, session_id: str):
    return (
        db.query(models.SessionDefault)
        .filter(models.SessionDefault.session_id == session_id)
        .first()
    )


def add_session(db: Session, session: schemas.SessionDefaultCreate):
    db_session = models.SessionDefault(**session.dict())

    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def update_session(db: Session, session: schemas.SessionDefaultCreate):
    existing_session = get_session_by_id(db, session.session_id)
    for key, value in session.dict().items():
        setattr(existing_session, key, value)

    db.commit()
    db.refresh(existing_session)
    return existing_session


def upsert_session(db: Session, session: schemas.SessionDefaultCreate):
    try:
        return update_session(db, session)
    except:
        return add_session(db, session)
