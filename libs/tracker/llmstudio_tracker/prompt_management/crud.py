from llmstudio_tracker.prompt_management import models, schemas
from sqlalchemy.orm import Session


def get_project_by_name(db: Session, name: str):
    return db.query(models.Project).filter(models.Project.name == name).first()


def get_prompt_by_name_and_label(
    db: Session, name: str, label: str = "production", skip: int = 0, limit: int = 100
):
    return (
        db.query(models.PromptDefault)
        .filter(models.PromptDefault.name == name, models.PromptDefault.label == label)
        .order_by(models.PromptDefault.created_at.asc())
        .offset(skip)
        .limit(limit)
        .first()
    )


def get_prompt_by_id(db: Session, prompt_id: int):
    return (
        db.query(models.PromptDefault)
        .filter(models.PromptDefault.prompt_id == prompt_id)
        .first()
    )


def get_prompt(db: Session, prompt_id: int = None, name: str = None, label: str = None):
    if prompt_id:
        return get_prompt_by_id(db, prompt_id)
    else:
        return get_prompt_by_name_and_label(db, name, label)


def add_prompt(db: Session, prompt: schemas.PromptDefault):
    db_session = models.PromptDefault(**prompt.dict())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def update_prompt(db: Session, prompt: schemas.PromptDefault):
    if prompt.prompt_id:
        existing_prompt = get_prompt_by_id(db, prompt.prompt_id)
    else:
        existing_prompt = get_prompt_by_name_and_label(db, prompt.name, prompt.label)

    existing_prompt.config = prompt.config
    existing_prompt.prompt = prompt.prompt
    existing_prompt.is_active = prompt.is_active
    existing_prompt.name = prompt.name
    existing_prompt.version = prompt.version
    existing_prompt.label = prompt.label

    db.commit()
    db.refresh(existing_prompt)
    return existing_prompt


def delete_prompt(db: Session, prompt: schemas.PromptDefault):
    db_prompt = (
        db.query(schemas.PromptDefault)
        .filter(
            models.PromptDefault.name == prompt.name,
            models.PromptDefault.label == prompt.label,
        )
        .one()
    )

    db.delete(db_prompt)
    db.commit()
