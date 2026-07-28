from fastapi import APIRouter, Depends
from llmstudio_tracker.database import get_db
from llmstudio_tracker.db.crud import prompt_manager as promt_manager_crud
from llmstudio_tracker.db.schemas.prompt_manager import (
    PromptDefaultResponse,
    PromptInfo,
)
from sqlalchemy.orm import Session


class PromptsRoutes:
    def __init__(self, router: APIRouter):
        self.router = router
        self.define_routes()

    def define_routes(self):
        self.router.post(
            "/add/prompt",
            response_model=PromptDefaultResponse,
        )(self.add_prompt)

        self.router.get("/get/prompt", response_model=PromptDefaultResponse)(
            self.get_prompt
        )

        self.router.patch("/update/prompt", response_model=PromptDefaultResponse)(
            self.update_prompt
        )

        self.router.delete("/delete/prompt")(self.delete_prompt)

    async def add_prompt(
        self, prompt: PromptDefaultResponse, db: Session = Depends(get_db)
    ):
        return promt_manager_crud.add_prompt(db=db, prompt=prompt)

    async def update_prompt(
        self, prompt: PromptDefaultResponse, db: Session = Depends(get_db)
    ):
        return promt_manager_crud.update_prompt(db, prompt)

    async def get_prompt(
        self,
        prompt_info: PromptInfo,
        db: Session = Depends(get_db),
    ):
        return promt_manager_crud.get_prompt(
            db,
            prompt_id=prompt_info.prompt_id,
            name=prompt_info.name,
            model=prompt_info.model,
            provider=prompt_info.provider,
        )

    async def delete_prompt(
        self, prompt: PromptDefaultResponse, db: Session = Depends(get_db)
    ):
        return promt_manager_crud.delete_prompt(db, prompt)
