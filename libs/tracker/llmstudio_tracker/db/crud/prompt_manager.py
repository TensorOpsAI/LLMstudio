from llmstudio_tracker.db.models.prompt_manager import PromptDefault
from llmstudio_tracker.db.schemas.prompt_manager import PromptDefaultBase
from sqlalchemy.orm import Session


def get_prompt_by_name_model_provider(
    db: Session, name: str, model: str, provider: str
):
    return (
        db.query(PromptDefault)
        .filter(
            PromptDefault.name == name,
            PromptDefault.model == model,
            PromptDefault.provider == provider,
            PromptDefault.is_active == True,
        )
        .order_by(PromptDefault.version.desc())
        .first()
    )


def get_prompt_by_id(db: Session, prompt_id: str):
    return db.query(PromptDefault).filter(PromptDefault.prompt_id == prompt_id).first()


def get_prompt(
    db: Session,
    prompt_id: str = None,
    name: str = None,
    model: str = None,
    provider: str = None,
):
    prompt = (
        get_prompt_by_id(db, prompt_id)
        if prompt_id
        else get_prompt_by_name_model_provider(db, name, model, provider)
    )

    if not prompt:
        return PromptDefaultBase()

    return prompt


def add_prompt(db: Session, prompt: PromptDefaultBase):

    prompt_created = PromptDefault.create_with_incremental_version(
        db,
        config=prompt.config,
        prompt=prompt.prompt,
        is_active=prompt.is_active,
        name=prompt.name,
        label=prompt.label,
        model=prompt.model,
        provider=prompt.provider,
    )
    db.add(prompt_created)
    db.commit()
    db.refresh(prompt_created)
    return prompt_created


def update_prompt(db: Session, prompt: PromptDefaultBase):
    if prompt.prompt_id:
        existing_prompt = get_prompt_by_id(db, prompt.prompt_id)
    else:
        existing_prompt = get_prompt_by_name_model_provider(
            db, prompt.name, prompt.model, prompt.provider
        )

    existing_prompt.config = prompt.config
    existing_prompt.prompt = prompt.prompt
    existing_prompt.is_active = prompt.is_active
    existing_prompt.name = prompt.name
    existing_prompt.model = prompt.model
    existing_prompt.provider = prompt.provider
    existing_prompt.version = prompt.version
    existing_prompt.label = prompt.label

    db.commit()
    db.refresh(existing_prompt)
    return existing_prompt


def delete_prompt(db: Session, prompt: PromptDefaultBase):
    if prompt.prompt_id:
        existing_prompt = get_prompt_by_id(db, prompt.prompt_id)
    else:
        existing_prompt = get_prompt_by_name_model_provider(
            db, prompt.name, prompt.model, prompt.provider
        )

    db.delete(existing_prompt)
    db.commit()
