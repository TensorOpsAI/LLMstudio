from typing import List

from fastapi import APIRouter, Depends
from llmstudio_tracker.database import engine, get_db
from llmstudio_tracker.session import crud, models, schemas
from sqlalchemy.orm import Session

models.Base.metadata.create_all(bind=engine)


class SessionsRoutes:
    def __init__(self, router: APIRouter):
        self.router = router
        self.define_routes()

    def define_routes(self):
        # Add session
        self.router.post(
            "/session",
            response_model=schemas.SessionDefault,
        )(self.add_session)

        # Read session
        self.router.get(
            "/session/{session_id}", response_model=List[schemas.SessionDefault]
        )(self.get_session)

        self.router.patch(
            "/session/{message_id}", response_model=schemas.SessionDefault
        )(self.update_session)

    async def add_session(
        self, session: schemas.SessionDefaultCreate, db: Session = Depends(get_db)
    ):
        return crud.upsert_session(db=db, session=session)

    async def update_session(
        self, message_id: int, extras: dict, db: Session = Depends(get_db)
    ):
        sessions = crud.update_session(db, message_id=message_id, extras=extras)
        return sessions

    async def get_session(self, session_id: str, db: Session = Depends(get_db)):
        sessions = crud.get_session_by_session_id(db, session_id=session_id)
        return sessions
