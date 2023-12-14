from sqlalchemy.orm import Session

from llmstudio.tracking import models, schemas


def get_project(db: Session, project_id: int):
    return (
        db.query(models.Project).filter(models.Project.project_id == project_id).first()
    )


def get_project_by_name(db: Session, name: str):
    return db.query(models.Project).filter(models.Project.name == name).first()


def get_projects(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Project).offset(skip).limit(limit).all()


def create_project(db: Session, project: schemas.ProjectCreate):
    db_project = models.Project(name=project.name, description=project.description)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


def get_sessions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Session).offset(skip).limit(limit).all()


def create_session(db: Session, session: schemas.SessionCreate, project_id: int):
    db_session = models.Session(**session.dict(), project_id=project_id)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def add_log_to_session(db: Session, log: schemas.LogCreate, session_id: int):
    db_log = models.Log(**log.dict(), session_id=session_id)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log


def get_session_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Log).offset(skip).limit(limit).all()


def add_log(db: Session, log: schemas.LogDefaultCreate):
    db_log = models.LogDefault(**log.dict())
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log


def get_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.LogDefault).offset(skip).limit(limit).all()
