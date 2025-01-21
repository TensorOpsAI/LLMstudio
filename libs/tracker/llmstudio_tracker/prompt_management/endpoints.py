from typing import List

from fastapi import APIRouter, Depends
from llmstudio_tracker.database import engine, get_db
from llmstudio_tracker.prompt_management import crud, models, schemas
from sqlalchemy.orm import Session

models.Base.metadata.create_all(bind=engine)


class PromptsRoutes:
    def __init__(self, router: APIRouter):
        self.router = router
        self.define_routes()

    def define_routes(self):
        # Add session
        self.router.post(
            "/prompt",
            response_model=schemas.PromptDefault,
        )(self.add_prompt)

        self.router.get("/prompt", response_model=List[schemas.PromptDefault])(
            self.get_prompt
        )

        self.router.patch("/prompt", response_model=schemas.PromptDefault)(
            self.update_prompt
        )

        self.router.delete("/prompt")(self.delete_prompt)

    async def add_prompt(
        self, prompt: schemas.PromptDefault, db: Session = Depends(get_db)
    ):
        return crud.add_prompt(db=db, prompt=prompt)

    async def update_prompt(
        self, prompt: schemas.PromptDefault, db: Session = Depends(get_db)
    ):
        return crud.update_prompt(db, prompt)

    async def get_prompt(
        self,
        prompt_id: int = None,
        name: str = None,
        label: str = None,
        db: Session = Depends(get_db),
    ):
        return crud.get_prompt(db, prompt_id=prompt_id, name=name, label=label)

    async def delete_prompt(
        self, prompt: schemas.PromptDefault, db: Session = Depends(get_db)
    ):
        return crud.delete_prompt(db, prompt)
